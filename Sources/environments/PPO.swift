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
    let learningRate: Float32
    let discount: Float32
    let epochs: Int
    let clipEpsilon: Float32
    let valueClipEpsilon: Float32
    let entropyCoefficient: Float32
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
        learningRate: Float32,
        discount: Float32,
        epochs: Int,
        clipEpsilon: Float32,
        valueClipEpsilon: Float32,
        entropyCoefficient: Float32,
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
        self.valueClipEpsilon = valueClipEpsilon
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
            
            let oldBatch = batches[0]
            let oldTraj = self.trajectory.batch(index: Array(oldBatch))
            let oldStates: Tensor<Float32> = Tensor(oldTraj.states).squeezingShape(at: 1)
            let dist: DiagGaussianProbabilityDistribution = self.oldActorCritic(oldStates)
            let action = dist.sample()
            let oldLogProbs = dist.neglogp(of: action).flattened()
            let oldVPred = self.oldActorCritic.criticNetwork(oldStates).flattened()
            var rewards = self.trajectory.returns(discount: self.discount)
            rewards = (rewards - rewards.mean()) / (rewards.standardDeviation() + 1e-5)
            
            for i in 0..<batches.count {
                let batch = batches[i]
                let index = Array(batch)
                let traj = self.trajectory.batch(index: index)
                let states: Tensor<Float32> = Tensor(traj.states).squeezingShape(at: 1)
                // Optimize policy network (actor)
                let vPred = self.actorCritic.criticNetwork(states).flattened()
                let indices = Tensor<Int32>(rangeFrom: Int32(index[0]), to: Int32(index[index.count-1]+1), stride: 1)
                let rewardMiniBatch = rewards.gathering(atIndices: indices)
                
                //var rewards = traj.returns(discount: self.discount)
                
                let (actorLoss, actorGradients) = valueWithGradient(at: self.actorCritic) { actorNetwork -> Tensor<Float32> in
                    //let dist = Self.feedForward(with: actorNetwork, for: oldActions, given: oldStates)
                    let dist: DiagGaussianProbabilityDistribution = actorNetwork.forward(states)
                    
                    let action = dist.sample()
                    
                    let logProbs = dist.neglogp(of: action).flattened()
                    
                    let ratios: Tensor<Float32> = exp(logProbs - oldLogProbs)
                    
                    let advantages: Tensor<Float> = rewardMiniBatch - vPred
                    
                    let pgLosses = advantages * ratios
                    
                    let pgLosses2 = advantages * ratios.clipped(min: 1 - self.clipEpsilon, max: 1 + self.clipEpsilon)
                    
                    let actorLoss = -Tensor(stacking: [pgLosses, pgLosses2]).min(alongAxes: 0).flattened().mean()
                    let entropyBonus: Tensor<Float> = Tensor<Float>(self.entropyCoefficient * dist.entropy().flattened()).mean()
                    let (criticLoss, criticGradients) = valueWithGradient(at: self.actorCritic.criticNetwork) { criticNetwork -> Tensor<Float> in
                        let vpred = criticNetwork.forward(states).flattened()
                        let vpredClipped = oldVPred + (vpred -  oldVPred).clipped(min: -self.valueClipEpsilon, max: self.valueClipEpsilon)
                        let vfLoss1 = squaredDifference(vpred, rewardMiniBatch)
                        let vfLoss2 = squaredDifference(vpredClipped, rewardMiniBatch)
                        let vfLoss = Tensor(stacking: [vfLoss1, vfLoss2]).max(alongAxes: 0).flattened()
                        
                        return vfLoss.mean() * self.valueLossCoefficient
                    }
                    
                    //criticGradients.clipByGlobalNorm(clipNorm: self.maxGradNorm)
                    self.criticOptimizer.update(&self.actorCritic.criticNetwork, along: criticGradients)
                    criticLosses.append(criticLoss.scalarized())
                    return actorLoss + criticLoss - entropyBonus
                }
                
                //actorGradients.clipByGlobalNorm(clipNorm: self.maxGradNorm)
                self.actorOptimizer.update(&self.actorCritic, along: actorGradients)
                actorLosses.append(actorLoss.scalarized())
            }
            
            print("epoch \(epoch) actor loss => \(actorLosses.reduce(Float32(0), +) / Float32(actorLosses.count)) critic loss => \(criticLosses.reduce(Float32(0), +) / Float32(criticLosses.count))")
            actorLosses.removeAll()
            criticLosses.removeAll()
        }
        self.oldActorCritic = self.actorCritic
        trajectory.removeAll()
    }
}
