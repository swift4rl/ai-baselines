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
            Array(0..<self.nSteps).inBatches(of: self.nSteps / self.nMiniBatches).forEach({batch in
                let traj = self.trajectory.batch(index: Array(batch))
                let oldStates: Tensor<Float32> = Tensor(traj.states).squeezingShape(at: 1)
                let oldLogProbs = Tensor(traj.logProbs).flattened()
                
                // Optimize policy network (actor)
                let oldStateValues = self.actorCritic.criticNetwork(oldStates).flattened()
                let rewards = traj.returns(discount: self.discount, lam: self.lam, values: oldStateValues.scalars)
                
                let (actorLoss, actorGradients) = valueWithGradient(at: self.actorCritic.actorNetwork) { actorNetwork -> Tensor<Float32> in
                    //let dist = Self.feedForward(with: actorNetwork, for: oldActions, given: oldStates)
                    let actionProbs = actorNetwork(oldStates)
                    
                    let dist = DiagGaussianProbabilityDistribution(flat: actionProbs)
                    
                    let action = dist.sample()
                    
                    let logProbs = dist.neglogp(of: action).flattened()
                    
                    let ratios: Tensor<Float32> = exp(logProbs - oldLogProbs)
                    
                    let advantages: Tensor<Float> = rewards - oldStateValues
                    let pgLosses = advantages * ratios
                    
                    let pgLosses2 = advantages * ratios.clipped(min: 1 - self.clipEpsilon, max: 1 + self.clipEpsilon)
                    
                    let actorLoss = Tensor(stacking: [pgLosses, pgLosses2]).min(alongAxes: 0).flattened()
                    
                    //let entropyBonus: Tensor<Float> = Tensor<Float>(self.entropyCoefficient * dist.entropy().flattened())
                    //let loss: Tensor<Float> =  -1 * (pgLoss + entropyBonus)
                    
                    return -actorLoss.mean()
                }
                
                let (criticLoss, criticGradients) = valueWithGradient(at: self.actorCritic.criticNetwork) { criticNetwork -> Tensor<Float> in
                    let stateValues = criticNetwork(oldStates).flattened()
                    let loss: Tensor<Float> = 0.5 * pow(stateValues - rewards, 2)
                    //loss.clipped(min: stateValues -  oldStateValues, max: <#T##Tensor<Float>#>)
                    return loss.mean()
                }
                // criticGradients.clipByGlobalNorm(clipNorm: maxGradNorm)
                //actorGradients.clipByGlobalNorm(clipNorm: maxGradNorm)
                self.criticOptimizer.update(&self.actorCritic.criticNetwork, along: criticGradients)
                criticLosses.append(criticLoss.scalarized())
                //print("gradients: \(actorGradients)")
                
                self.actorOptimizer.update(&self.actorCritic.actorNetwork, along: actorGradients)
                actorLosses.append(actorLoss.scalarized())
                self.oldActorCritic = self.actorCritic
            })
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
