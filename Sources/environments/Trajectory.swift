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

/// A cache saving all rollouts for batch updates.
///
/// PPO first collects fixed-length trajectory segments then updates weights. All the trajectory
/// segments are discarded after the update.

import TensorFlow

struct Trajectory {
    var states: [Tensor<Float32>] = []
    var actions: [Tensor<Float32>] = []
    var rewards: [Float] = []
    var logProbs: [Tensor<Float32>] = []
    var isDones: [Bool] = []
    
    var nSteps: Int
    
    init() {
        self.nSteps = 0
    }
    
    init(
        states: [Tensor<Float32>],
        actions: [Tensor<Float32>],
        rewards: [Float],
        logProbs: [Tensor<Float32>],
        isDones: [Bool]) {
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.isDones = isDones
        self.logProbs = logProbs
        self.nSteps = isDones.count
    }
    
    public func batch(index: Array<Int>) -> Trajectory{
        return Trajectory(states: index.map{states[$0]},
                          actions: index.map{actions[$0]},
                          rewards: index.map{rewards[$0]},
                          logProbs: index.map{logProbs[$0]},
                          isDones: index.map{isDones[$0]}
        )
    }
    @noDerivative
    public func returns(discount: Float32, lam: Float32, values: [Float32]) -> Tensor<Float32> {
        var lastGaeLam: Float32 = 0
        var advantages = Array<Float32>(repeating: 0, count: nSteps)
        for step in (0 ..< self.rewards.count).reversed() {
            var nextnonterminal: Float32
            let nextValues: Float32
            if step == nSteps - 1 {
                nextnonterminal = 1.0 - (self.isDones[nSteps - 1] ? 1.0 : 0.0)
                nextValues = values[nSteps - 1]
            } else {
                nextnonterminal = 1.0 - (self.isDones[step + 1] ? 1.0 : 0.0)
                nextValues = values[step + 1]
            }
            let delta = self.rewards[step] + discount * nextValues * nextnonterminal - values[step]
            lastGaeLam = delta + discount * lam * nextnonterminal * lastGaeLam
            advantages[step] = lastGaeLam
        }
        return Tensor<Float32>(advantages) + Tensor(values)
    }
    
    @noDerivative
    public func returns(discount: Float32) -> Tensor<Float32>{
        var rewards : [ Float32 ] = []
        var discountedReward: Float32 = 0
        for i in (0..<self.rewards.count).reversed() {
            if self.isDones[i] {
                discountedReward = 0
            }
            discountedReward = self.rewards[i] + (discount * discountedReward)
            rewards.insert(discountedReward, at: 0)
        }
        var returns = Tensor<Float32>(rewards)
        return returns;
    }
    
    mutating func append(state: Tensor<Float32>, action: Tensor<Float32>, reward: Float, logProb: Tensor<Float32>, isDone: Bool) {
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        isDones.append(isDone)
        logProbs.append(logProb)
        self.nSteps += 1
    }
    
    mutating func removeAll() {
        states.removeAll()
        actions.removeAll()
        rewards.removeAll()
        isDones.removeAll()
        logProbs.removeAll()
        self.nSteps = 0
    }
    
    
}
