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

struct ActorNetwork: Layer {
    typealias Input = Tensor<Float32>
    typealias Output = Tensor<Float32>

    var l1, l2: Dense<Float>

    init(l1: Dense<Float>, hiddenSize: Int, actionCount: Int) {
        self.l1 = l1
        self.l2 = Dense<Float>(
            inputSize: hiddenSize,
            outputSize: actionCount,
            activation: tanh,
            weightInitializer: heNormal(seed: TensorFlowSeed(1,1))
        )
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: l1, l2)
    }
}

struct CriticNetwork: Layer {
    typealias Input = Tensor<Float32>
    typealias Output = Tensor<Float32>

    var l1, l2: Dense<Float>

    init(l1: Dense<Float>, hiddenSize: Int) {
        self.l1 = l1
        l2 = Dense<Float>(
            inputSize: hiddenSize,
            outputSize: 1,
            activation: tanh,
            weightInitializer: heNormal(seed: TensorFlowSeed(1,1))
        )
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: l1, l2)
    }
}

struct ActorCritic: Layer {
    typealias Input = Tensor<Float32>
    typealias Output = DiagGaussianProbabilityDistribution
    
    var actorNetwork: ActorNetwork
    var criticNetwork: CriticNetwork

    init(observationSize: Int, hiddenSize: Int, actionCount: Int) {
        let l1 = Dense<Float>(
            inputSize: observationSize,
            outputSize: hiddenSize,
            activation: tanh,
            weightInitializer: heUniform(seed: TensorFlowSeed(1,1))
        )
        
        self.actorNetwork = ActorNetwork(
            l1: l1,
            hiddenSize: hiddenSize,
            actionCount: actionCount
        )
        
        self.criticNetwork = CriticNetwork(
            l1: l1,
            hiddenSize: hiddenSize
        )
    }

    @differentiable
    func callAsFunction(_ state: Input) -> Output {
        precondition(state.rank == 2, "The input must be 2-D ([batch size, state size]).")
        return DiagGaussianProbabilityDistribution(flat: self.actorNetwork(state))
    }
}
