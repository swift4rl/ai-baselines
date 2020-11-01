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

open class PPO: Model {
    var trajectory: Trajectory
    let learningRate: Float
    let discount: Float
    let epochs: Int
    let clipEpsilon: Float
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

    public func predict(state: Tensor<Float32>) -> Tensor<Float32> {
        let dist: DiagGaussianProbabilityDistribution = oldActorCritic(state)
        return dist.sample()
    }
    
    public func updateTrajectory(action: Tensor<BehaviorSpecContinousAction.Scalar>, observation: Observation<Float32>) {
        if case let Observation.SingleObservation(state, reward, done, _) = observation {
            trajectory.append(
                state: state,
                action: action,
                reward: reward,
                isDone: done
            )
        }
    }

    open func update() {
        let values = trajectory.states.map{ self.oldActorCritic.criticNetwork($0).flattened() }
        let returns = trajectory.returns(discount: self.discount, lam: self.lam, values: values)
        Array(0..<self.nSteps).inBatches(of: self.nSteps / self.nMiniBatches).forEach({batch in
            let index = batch.shuffled()
            let ret = returns.scalars
            let subsample = self.trajectory.batch(index: index)
            let rewardsBatch = Tensor(index.map{ret[$0]})
            let valuesBatch = Tensor(index.map{values[$0]})
            update(using: subsample,
                   rewards: rewardsBatch,
                   values: valuesBatch)
        })
        trajectory.removeAll()
    }
    
    static func feedForward(with actorNetwork: ActorNetwork, for actions: Tensor<Float32>, given state: Tensor<Float32>) -> DiagGaussianProbabilityDistribution {
        let sh = TensorShape(actions.shape[0])
        let range = Tensor<Int32>(rangeFrom: 0, to: Int32(actions.shape[0]), stride: 1)
        let zeros = Tensor<Int32>(zeros: sh)
        let indices: Tensor<Int32> =
            Tensor(stacking: [
                range.concatenated(with: range),
                zeros.concatenated(with: zeros),
                zeros.concatenated(with: Tensor(ones: sh))
            ], alongAxis: 1)

        let actionProbs: Tensor<Float32> = actorNetwork(state).dimensionGathering(atIndices: indices)
        return DiagGaussianProbabilityDistribution(flat: actionProbs)
    }
    
    func update(using traj: Trajectory, rewards: Tensor<Float32>, values: Tensor<Float32>) {
        let oldStates: Tensor<Float32> = Tensor(traj.states)
        let oldActions: Tensor<Float32> = Tensor(traj.actions)
        let dist = Self.feedForward(with: oldActorCritic.actorNetwork, for: oldActions, given: oldStates)
        let oldLogProbs = dist.neglogp(of: dist.sample())
        
        let returns = (rewards - rewards.mean()) / (rewards.standardDeviation() + 1e-10)

        var actorLosses: [Float32] = []
        var criticLosses: [Float32] = []
        for _ in 0..<epochs {
            // Optimize policy network (actor)
            var (actorLoss, actorGradients) = valueWithGradient(at: self.actorCritic.actorNetwork) { actorNetwork -> Tensor<Float32> in
                let dist = Self.feedForward(with: actorNetwork, for: oldActions, given: oldStates)
                let logProbs = dist.neglogp(of: dist.sample())
                let vpred = self.actorCritic.criticNetwork(oldStates).flattened()
                let vpredclipped = values + (vpred - values).clipped(min: -self.clipEpsilon, max: self.clipEpsilon)
                
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
