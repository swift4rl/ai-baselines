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
    var trajectory: Trajectory
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
    
    let valueLossCoefficient: Float32
    
    let maxGradNorm: Float32
    
    let lam: Float32

    let nSteps: Int
    
    let nMiniBatches: Int
    
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
        entropyCoefficient: Float,
        valueLossCoefficient: Float32 = 0.5,
        maxGradNorm: Float32 = 0.5,
        lam: Float32 = 0.95,
        nSteps: Int = 128,
        nMiniBatches: Int = 4
    ) {
        self.learningRate = learningRate
        self.discount = discount
        self.epochs = epochs
        self.clipEpsilon = clipEpsilon
        self.entropyCoefficient = entropyCoefficient
        self.valueLossCoefficient = valueLossCoefficient
        self.maxGradNorm = maxGradNorm
        self.lam = lam
        self.nSteps = nSteps
        self.nMiniBatches = nMiniBatches
        self.trajectory = Trajectory()
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
        let dist: DiagGaussianProbabilityDistribution = oldActorCritic(state)
        let action = dist.sample()
        var ret: (Tensor<Float32>, Bool, Float)
        let value = self.oldActorCritic.criticNetwork(state).flattened()
        
        if case let StepResult.SingleStepResult(observation, reward, done, _) = try env.step(action) {
            trajectory.append(
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
        let returns = trajectory.returns(discount: self.discount, lam: self.lam)
        Array(0..<self.nSteps).inBatches(of: self.nSteps / self.nMiniBatches).forEach({batch in
            let index = batch.shuffled()
            let ret = returns.scalars
            update(using: self.trajectory.batch(index: index), rewards: Tensor(index.map{ret[$0]}))
        })
        trajectory.removeAll()
    }
    
    func update(using traj: Trajectory, rewards: Tensor<Float32>) {
        let oldStates: Tensor<Float32> = Tensor(traj.states)
        let oldActions: Tensor<Float32> = Tensor(traj.actions)
        let oldLogProbs: Tensor<Float32> = Tensor(traj.logProbs).flattened()
        let values: Tensor<Float32> = Tensor(traj.values).flattened()
        
        let returns = (rewards - rewards.mean()) / (rewards.standardDeviation() + 1e-10)

        var actorLosses: [Float32] = []
        var criticLosses: [Float32] = []
        for _ in 0..<epochs {
            // Optimize policy network (actor)
            var (actorLoss, actorGradients) = valueWithGradient(at: self.actorCritic.actorNetwork) { actorNetwork -> Tensor<Float32> in
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
                let vpredclipped = values + (vpred - values).clipped(min: -self.clipEpsilon, max: self.clipEpsilon)
                
                let logProbs = dist.neglogp(of: dist.sample())
//              let logProbs = dist.logProbability(of: dist.sample())
//              print("oldActions \(oldActions)")
//              print("oldValues \(oldValues)")
//              print("oldValues Shape \(oldValues.shape)")
//              print("vpred \(vpred.shape)")
//              print("vpred \(vpred)")
//              print("vpredclipped \(vpredclipped)")
//              print("logProbs \(logProbs)")
//              print("tfRewards \(tfRewards)")
                let vfLosses1 = (vpred - returns).squared()
//              print("vfLosses1 \(vfLosses1)")
                let vfLosses2 = (vpredclipped - returns).squared()
//              print("vfLosses2 \(vfLosses2)")
                let valueLoss = self.valueLossCoefficient * max(vfLosses1, vfLosses2).mean()
//              print("valueLoss \(valueLoss)")
                var advantages: Tensor<Float> = returns - vpred
//              print("advantages \(advantages)")
                advantages = (advantages - advantages.mean()) / (advantages.standardDeviation() + 1e-8)
//              print("normalized advantages \(advantages)")
            
//              print("oldLogProbs \(oldLogProbs)")
                let ratios: Tensor<Float32> = exp(oldLogProbs - logProbs)
//              print("ratios \(ratios)")
                
                let pgLosses = -advantages * ratios
//              print("pgLosses \(pgLosses)")
                
                let pgLosses2 = -advantages * ratios.clipped(min: 1 - self.clipEpsilon, max: 1 + self.clipEpsilon)
//              print("pgLosses2 \(pgLosses)")
                
                let policyLoss = max(pgLosses, pgLosses2).mean()
//              print("policyLoss \(policyLoss)")
                
                let entropy = dist.entropy().mean()
//              let entropy = -logProbs.mean()
//              print("entropy \(entropy)")

                let loss: Tensor<Float> =  policyLoss + self.valueLossCoefficient * valueLoss - self.entropyCoefficient * entropy
                let meanLoss = loss.mean()
                return meanLoss
            }
            //print("gradients: \(actorGradients)")
            actorGradients.clipByGlobalNorm(clipNorm: maxGradNorm)
            self.actorOptimizer.update(&self.actorCritic.actorNetwork, along: actorGradients)
            actorLosses.append(actorLoss.scalarized())
                
                // Optimize value network (critic)
            
            var (criticLoss, criticGradients) = valueWithGradient(at: self.actorCritic.criticNetwork) { criticNetwork -> Tensor<Float> in
                    let stateValues = criticNetwork(oldStates).flattened()
                    let loss: Tensor<Float> = (stateValues - returns).squared().mean()
                    return loss
            }
            criticGradients.clipByGlobalNorm(clipNorm: maxGradNorm)
            self.criticOptimizer.update(&self.actorCritic.criticNetwork, along: criticGradients)
            criticLosses.append(criticLoss.scalarized())
            
        }
        print("actor loss => \(actorLosses.reduce(Float32(0), +) / Float32(actorLosses.count))")
        print("critic loss => \(criticLosses.reduce(Float32(0), +) / Float32(criticLosses.count))")
        self.oldActorCritic = self.actorCritic
        
    }
}
