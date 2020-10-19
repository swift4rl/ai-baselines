// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow
import environments
/// Agent that uses the Proximal Policy Optimization (PPO).
///
/// Proximal Policy Optimization is an algorithm that trains an actor (policy) and a critic (value
/// function) using a clipped objective function. The clipped objective function simplifies the
/// update equation from its predecessor Trust Region Policy Optimization (TRPO). For more
/// information, check Proximal Policy Optimization Algorithms (Schulman et al., 2017).
open class PPOAgent {
    // Cache for trajectory segments for minibatch updates.
    var memory: PPOMemory
    /// The learning rate for both the actor and the critic.
    let learningRate: Float
    /// The discount factor that measures how much to weight to give to future
    /// rewards when calculating the action value.
    let discount: Float
    /// Number of epochs to run minibatch updates once enough trajectory segments are collected.
    let epochs: Int
    /// Parameter to clip the probability ratio.
    let clipEpsilon: Float
    /// Coefficient for the entropy bonus added to the objective.
    let entropyCoefficient: Float

    var actorCritic: ActorCritic
    var oldActorCritic: ActorCritic
    var actorOptimizer: AMSGrad<ActorNetwork>
    var criticOptimizer: AMSGrad<CriticNetwork>

    public init(
        observationSize: Int,
        hiddenSize: Int,
        actionCount: Int,
        learningRate: Float,
        discount: Float,
        epochs: Int,
        clipEpsilon: Float,
        entropyCoefficient: Float
    ) {
        self.learningRate = learningRate
        self.discount = discount
        self.epochs = epochs
        self.clipEpsilon = clipEpsilon
        self.entropyCoefficient = entropyCoefficient

        self.memory = PPOMemory()

        self.actorCritic = ActorCritic(
            observationSize: observationSize,
            hiddenSize: hiddenSize,
            actionCount: actionCount
        )
        self.oldActorCritic = self.actorCritic
        self.actorOptimizer = AMSGrad(for: actorCritic.actorNetwork, learningRate: learningRate)
        self.criticOptimizer = AMSGrad(for: actorCritic.criticNetwork, learningRate: learningRate)
    }

    open func step(env: UnityToGymWrapper, state: Tensor<Float32>) throws -> (Tensor<Float32>, Bool, Float) {
        let dist: DiagGaussianProbabilityDistribution = oldActorCritic(state)
        let action = dist.sample()
        var ret: (Tensor<Float32>, Bool, Float)
        let value = self.oldActorCritic.criticNetwork(state).flattened()
        //TODO change this env.step(Tensor<Float32>(action)) with proper Float actions
        if case let StepResult.SingleStepResult(observation, reward, done, _) = try env.step(action) {
            memory.append(
                state: state,
                action: action,
                value: value,
                reward: reward,
                logProb: dist.neglogp(of: action),
                isDone: done
            )
            ret = (observation, done, reward)
        } else {
            throw UnityException.UnityEnvironmentException(reason: "error occred during step call")
        }
        return ret
    }

    open func update() {
        // Discount rewards for advantage estimation
        var rewards: [Float32] = []
        var discountedReward: Float32 = 0
        for i in (0..<memory.rewards.count).reversed() {
            if memory.isDones[i] {
                discountedReward = 0
            }
            discountedReward = memory.rewards[i] + (discount * discountedReward)
            rewards.insert(discountedReward, at: 0)
        }
        var tfRewards = Tensor<Float32>(rewards)
        tfRewards = (tfRewards - tfRewards.mean()) / (tfRewards.standardDeviation() + 1e-8)

        // Retrieve stored states, actions, and log probabilities
        let oldStates: Tensor<Float32> = Tensor(memory.states)
        let oldActions: Tensor<Float32> = Tensor(memory.actions)
        let oldLogProbs: Tensor<Float32> = Tensor(memory.logProbs)
        let oldValues: Tensor<Float32> = Tensor(memory.values)
        // Optimize actor and critic
        var actorLosses: [Float32] = []
        var criticLosses: [Float32] = []
        for _ in 0..<epochs {
            // Optimize policy network (actor)
            let (actorLoss, actorGradients) = valueWithGradient(at: self.actorCritic.actorNetwork) { actorNetwork -> Tensor<Float32> in
                let sh = TensorShape(oldActions.shape[0])
                let range = Tensor<Int32>(rangeFrom: 0, to: Int32(oldActions.shape[0]), stride: 1)
                let zeros = Tensor<Int32>(zeros: sh)
                let tfIndices: Tensor<Int32> =
                    Tensor(stacking: [
                        range.concatenated(with: range),
                        zeros.concatenated(with: zeros),
                        zeros.concatenated(with: Tensor(ones: sh))
                    ], alongAxis: 1)

                let actionProbs: Tensor<Float32> = actorNetwork(oldStates).dimensionGathering(atIndices: tfIndices)
                let dist = DiagGaussianProbabilityDistribution(flat: actionProbs)
                
                let vpred = self.actorCritic.criticNetwork(oldStates).flattened()
                let vpredclipped = oldValues + (vpred - oldValues).clipped(min: -self.clipEpsilon, max: self.clipEpsilon)
                
                let logProbs = dist.neglogp(of: dist.sample())
                
                let vfLosses1 = (vpred - tfRewards).squared()
                let vfLosses2 = (vpredclipped - tfRewards).squared()
                let vfLoss = 0.5 * max(vfLosses1, vfLosses2).mean()
                
                var advantages: Tensor<Float> = tfRewards - vpred
                advantages = (advantages - advantages.mean()) / (advantages.standardDeviation() + 1e-8)
                let ratios: Tensor<Float32> = exp(oldLogProbs - logProbs)
                let pgLosses = -advantages * ratios
                let pgLosses2 = -advantages * ratios.clipped(min: 1 - self.clipEpsilon, max: 1 + self.clipEpsilon)
                
                let pgLoss = max(pgLosses, pgLosses2).mean()
                
                let entropy = dist.entropy()
                let entropyBonus: Tensor<Float> = Tensor<Float>(self.entropyCoefficient * entropy)

                let vfCoef: Float32 = 0.5
                
                let loss: Tensor<Float> =  pgLoss - entropyBonus + vfLoss * vfCoef
                print("loss => \(loss)")
                return loss.mean()
            }
            //print("gradients: \(actorGradients)")
            self.actorOptimizer.update(&self.actorCritic.actorNetwork, along: actorGradients)
            actorLosses.append(actorLoss.scalarized())

            // Optimize value network (critic)
            let (criticLoss, criticGradients) = valueWithGradient(at: self.actorCritic.criticNetwork) { criticNetwork -> Tensor<Float> in
                let stateValues = criticNetwork(oldStates).flattened()
                let loss: Tensor<Float> = 0.5 * pow(stateValues - tfRewards, 2)

                return loss.mean()
            }
            self.criticOptimizer.update(&self.actorCritic.criticNetwork, along: criticGradients)
            criticLosses.append(criticLoss.scalarized())
        }
        self.oldActorCritic = self.actorCritic
        memory.removeAll()
    }
}
