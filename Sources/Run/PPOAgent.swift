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
    var actorOptimizer: Adam<ActorNetwork>
    var criticOptimizer: Adam<CriticNetwork>

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
        self.actorOptimizer = Adam(for: actorCritic.actorNetwork, learningRate: learningRate)
        self.criticOptimizer = Adam(for: actorCritic.criticNetwork, learningRate: learningRate)
    }

    open func step(env: UnityToGymWrapper, state: Tensor<Float32>) throws -> (Tensor<Float32>, Bool, Float) {
        let actionProbs: Tensor<Float32> = oldActorCritic(state)
        var ret: (Tensor<Float32>, Bool, Float)
        //TODO change this env.step(Tensor<Float32>(action)) with proper Float actions
        if case let StepResult.SingleStepResult(observation, reward, done, _) = try env.step(actionProbs) {
            memory.append(
                state: state,
                action: actionProbs,
                reward: reward,
                logProb: -(actionProbs * exp(actionProbs)).sum(squeezingAxes: -1),
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
        tfRewards = (tfRewards - tfRewards.mean()) / (tfRewards.standardDeviation() + 1e-5)

        // Retrieve stored states, actions, and log probabilities
        let oldStates: Tensor<Float32> = Tensor(memory.states)
        let oldActions: Tensor<Float32> = Tensor(memory.actions)
        let oldLogProbs: Tensor<Float32> = Tensor(memory.logProbs)
        // Optimize actor and critic
        var actorLosses: [Float32] = []
        var criticLosses: [Float32] = []
        for _ in 0..<epochs {
            // Optimize policy network (actor)
            let (actorLoss, actorGradients) = valueWithGradient(at: self.actorCritic.actorNetwork) { actorNetwork -> Tensor<Float32> in
                let actionProbs = actorNetwork(oldStates)
                let logProbs = -(actionProbs * exp(actionProbs)).sum(squeezingAxes: -1)
                let stateValues = self.actorCritic.criticNetwork(oldStates).flattened()
                let ratios: Tensor<Float> = exp(logProbs - oldLogProbs)
                let advantages: Tensor<Float> = (tfRewards - stateValues)
                let surrogateObjective = Tensor(stacking: [
                    ratios * advantages,
                    ratios.clipped(min:1 - self.clipEpsilon, max: 1 + self.clipEpsilon) * advantages
                ]).min(alongAxes: 0).flattened()
                let entropy = -(actionProbs * exp(actionProbs)).sum(squeezingAxes: -1)
                let entropyBonus: Tensor<Float> = Tensor<Float>(self.entropyCoefficient * entropy)
                let loss: Tensor<Float> = -1 * (surrogateObjective + entropyBonus)

                return loss.mean()
            }
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
