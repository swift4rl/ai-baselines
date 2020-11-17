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

    var l1, l2, l3: Dense<Float>

    init(inputSize: Int, hiddenSize: Int, actionCount: Int) {
        self.l1 = Dense<Float>(
            inputSize: inputSize,
            outputSize: hiddenSize,
            activation: tanh,
            weightInitializer: heNormal()
        )
        self.l2 = Dense<Float>(
            inputSize: hiddenSize,
            outputSize: hiddenSize,
            activation: tanh,
            weightInitializer: heNormal()
        )
        l3 = Dense<Float>(
            inputSize: hiddenSize,
            outputSize: actionCount,
            activation: tanh,
            weightInitializer: heNormal()
        )
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: l1, l2, l3)
    }
}

struct CriticNetwork: Layer {
    typealias Input = Tensor<Float32>
    typealias Output = Tensor<Float32>

    var l1, l2, l3: Dense<Float>

    init(inputSize: Int, hiddenSize: Int) {
        self.l1 = Dense<Float>(
            inputSize: inputSize,
            outputSize: hiddenSize,
            activation: tanh,
            weightInitializer: heNormal()
        )
        self.l2 = Dense<Float>(
            inputSize: hiddenSize,
            outputSize: hiddenSize,
            activation: tanh,
            weightInitializer: heNormal()
        )
        l3 = Dense<Float>(
            inputSize: hiddenSize,
            outputSize: 1,
            activation: tanh,
            weightInitializer: heNormal()
        )
    }

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        return input.sequenced(through: l1, l2, l3)
    }
}

struct ActorCritic: Layer {
    typealias Input = Tensor<Float32>
    typealias Output = DiagGaussianProbabilityDistribution
    
    var actorNetwork: ActorNetwork
    var criticNetwork: CriticNetwork

    init(observationSize: Int, hiddenSize: Int, actionCount: Int) {
        self.actorNetwork = ActorNetwork(
            inputSize: observationSize,
            hiddenSize: hiddenSize,
            actionCount: actionCount
        )
        self.criticNetwork = CriticNetwork(
            inputSize: observationSize,
            hiddenSize: hiddenSize
        )
    }

    @differentiable
    func callAsFunction(_ state: Input) -> Output {
        precondition(state.rank == 2, "The input must be 2-D ([batch size, state size]).")
        return DiagGaussianProbabilityDistribution(flat: self.actorNetwork(state).flattened())
    }
}
