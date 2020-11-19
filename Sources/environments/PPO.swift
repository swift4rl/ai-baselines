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
import PythonKit

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
    var actorOptimizer: Adam<ActorCritic>
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
        self.actorOptimizer = Adam(for: actorCritic, learningRate: learningRate)
        self.criticOptimizer = Adam(for: actorCritic.criticNetwork, learningRate: learningRate)
    }
    
    public func predict(state: Tensor<Float32>) -> DiagGaussianProbabilityDistribution { oldActorCritic(state) }
    
    public func updateTrajectory(action: Tensor<Float32>, logProb: Tensor<Float32>, observation: Observation<Float32>) {
        if case let Observation.SingleObservation(state, reward, done, _) = observation {
            trajectory.append(
                state: state,
                action: action,
                reward: reward,
                logProb: logProb,
                isDone: done
            )
        }
    }
    
    open func update() {
        for epoch in 0..<epochs {
            var actorLosses: [Float32] = []
            var criticLosses: [Float32] = []
            let batches = Array(Array(0..<self.nSteps).inBatches(of: self.nSteps / self.nMiniBatches))
            for i in 1..<batches.count {
                let oldBatch = batches[i-1]
                let oldTraj = self.trajectory.batch(index: Array(oldBatch))
                
                let batch = batches[i]
                let traj = self.trajectory.batch(index: Array(batch))
                
                let oldStates: Tensor<Float32> = Tensor(oldTraj.states).squeezingShape(at: 1)
                let states: Tensor<Float32> = Tensor(traj.states).squeezingShape(at: 1)
                let oldLogProbs = Tensor(oldTraj.logProbs).flattened()
                
                // Optimize policy network (actor)
                let oldVPred = self.actorCritic.criticNetwork(oldStates).flattened()
                let vPred = self.actorCritic.criticNetwork(states).flattened()
                //let rewards = traj.returns(discount: self.discount, lam: self.lam, values: oldVPred.scalars)
                var rewards = traj.returns(discount: self.discount)
                rewards = (rewards - rewards.mean()) / (rewards.standardDeviation() + 1e-5)

                
                var (actorLoss, actorGradients) = valueWithGradient(at: self.actorCritic) { actorNetwork -> Tensor<Float32> in
                    //let dist = Self.feedForward(with: actorNetwork, for: oldActions, given: oldStates)
                    let dist: DiagGaussianProbabilityDistribution = actorNetwork(states)

                    let action = dist.mode()

                    let logProbs = dist.neglogp(of: action).flattened()

                    let ratios: Tensor<Float32> = exp(logProbs - oldLogProbs)

                    let advantages: Tensor<Float> = rewards - vPred

                    let pgLosses = advantages * ratios

                    let pgLosses2 = advantages * ratios.clipped(min: 1 - self.clipEpsilon, max: 1 + self.clipEpsilon)

                    let actorLoss = Tensor(stacking: [pgLosses, pgLosses2]).min(alongAxes: 0).flattened().mean()
                    //let entropyBonus: Tensor<Float> = Tensor<Float>(self.entropyCoefficient * dist.entropy().flattened()).mean()
                    var (criticLoss, criticGradients) = valueWithGradient(at: self.actorCritic.criticNetwork) { criticNetwork -> Tensor<Float> in
                        let vpred = criticNetwork(states).flattened()
                        let vpredClipped = oldVPred + (vpred -  oldVPred).clipped(min: -self.clipEpsilon, max: self.clipEpsilon)
                        let vfLoss1 = (vpred - rewards).squared()
                        let vfLoss2 = (vpredClipped - rewards).squared()
                        let vfLoss = 0.5 * Tensor(stacking: [vfLoss1, vfLoss2]).max(alongAxes: 0).flattened()
                        return vfLoss.mean() * self.valueLossCoefficient
                    }
                    
                    criticGradients.clipByGlobalNorm(clipNorm: self.maxGradNorm)
                    self.criticOptimizer.update(&self.actorCritic.criticNetwork, along: criticGradients)
                    criticLosses.append(criticLoss.scalarized())
                    return criticLoss - actorLoss //- entropyBonus
                }
                
                actorGradients.clipByGlobalNorm(clipNorm: self.maxGradNorm)
                self.actorOptimizer.update(&self.actorCritic, along: actorGradients)
                actorLosses.append(actorLoss.scalarized())
                self.oldActorCritic = self.actorCritic
            }
           
            print("epoch \(epoch) actor loss => \(actorLosses.reduce(Float32(0), +) / Float32(actorLosses.count)) critic loss => \(criticLosses.reduce(Float32(0), +) / Float32(criticLosses.count))")
            actorLosses.removeAll()
            criticLosses.removeAll()
        }
        trajectory.removeAll()
    }
    
//    static func feedForward(with actorNetwork: ActorNetwork, for actions: Tensor<Float32>, given state: Tensor<Float32>) -> DiagGaussianProbabilityDistribution {
//        let sh = TensorShape(actions.shape[0])
//        let range = Tensor<Int32>(rangeFrom: 0, to: Int32(actions.shape[0]), stride: 1)
//        let zeros = Tensor<Int32>(zeros: sh)
//        let indices: Tensor<Int32> =
//            Tensor(stacking: [
//                range.concatenated(with: range),
//                zeros.concatenated(with: Tensor(ones: sh))
//            ], alongAxis: 1)
//
//        var actionProbs: Tensor<Float32> = actorNetwork(state)
//        actionProbs = actionProbs.dimensionGathering(atIndices: indices)
//
//        return DiagGaussianProbabilityDistribution(flat: actionProbs)
//    }
//
}
