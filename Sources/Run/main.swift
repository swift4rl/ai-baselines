////
////  main.swift
////  App
////
////  Created by Sercan Karaoglu on 01/08/2020.
////
import TensorFlow
import Foundation
import environments

let observationSize: Int =  8 //Int(env.observationSpace?.shape[0] ?? 0)
let actionCount: Int = 2

// Hyperparameters
/// The !size of the hidden layer of the 2-layer actor network and critic network. The actor network
/// has the shape observationSize - hiddenSize - actionCount, and the critic network has the same
/// shape but with a single output node.

let hiddenSize: Int = 128
/// The learning rate for both the actor and the critic.
let learningRate: Float32 = 0.003
/// The discount factor. This measures how much to "discount" the future rewards
/// that the agent will receive. The discount factor must be from 0 to 1
/// (inclusive). Discount factor of 0 means that the agent only considers the
/// immediate reward and disregards all future rewards. Discount factor of 1
/// means that the agent values all rewards equally, no matter how distant
/// in the future they may be. Denoted gamma in the PPO paper.
let discount: Float32 = 0.99
/// Number of epochs to run minibatch updates once enough trajectory segments are collected. Denoted
/// K in the PPO paper.
let epochs: Int = 3
/// Parameter to clip the probability ratio. The ratio is clipped to [1-clipEpsilon, 1+clipEpsilon].
/// Denoted epsilon in the PPO paper.
let clipEpsilon: Float32 = 0.03

let valueClipEpsilon: Float32 = 0.7

let valueLossCoefficient: Float32 = 0.3
/// Coefficient for the entropy bonus added to the objective. Denoted c_2 in the PPO paper.
let entropyCoefficient: Float32 = 1e-3
/// Maximum number of episodes to train the agent. The training is terminated
/// early if maximum score is achieved consecutively 10 times.
let maxEpisodes: Int = Int.max
/// Maximum timestep per episode.
let maxTimesteps: Int = Int.max
/// The length of the trajectory segment. Denoted T in the PPO paper.
let updateStep = 4

let batchSize = 128
//
//let updateTimestep = 10
//
//let nMiniBatches = 2

// Training loop
var episode: Int = 0
var episodeReturn: Float = 0
var episodeReturns: [Float] = []
var maxEpisodeReturn: Float = -1
var totalStepCount = 0


var agent: PPO = PPO(
    observationSize: observationSize,
    hiddenSize: hiddenSize,
    actionCount: actionCount,
    learningRate: learningRate,
    discount: discount,
    epochs: epochs,
    clipEpsilon: clipEpsilon,
    valueClipEpsilon: valueClipEpsilon,
    entropyCoefficient: entropyCoefficient,
    valueLossCoefficient: valueLossCoefficient,
    batchSize: batchSize
)

let appPath = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Desktop")
    .appendingPathComponent("cartpole.app")
    .appendingPathComponent("Contents")
    .appendingPathComponent("MacOS")
    .appendingPathComponent("ML experiment")

print(appPath)

let channel = EngineConfigurationChannel()

try channel.setConfigurationParameters(
    width: 600,
    height: 600,
    qualityLevel: 5,
    timeScale: 20,
    targetFrameRate: -1,
    captureFrameRate: 60
)

let unityEnv = UnityContinousEnvironment(filename: appPath, model: agent, basePort: 5004, sideChannels: [channel])
let start = DispatchTime.now()

unityEnv.train(onNextState: { model, reward in
    episodeReturn += reward
    if totalStepCount != 0 && totalStepCount % 5000 == 0 {
        let end = DispatchTime.now()
        let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
        let timeInterval = Double(nanoTime) / 1_000_000_000
        let avgEpisodeReturns = episodeReturns.reduce(0, +) / Float32(episodeReturns.count)
        let max = episodeReturns.max()!
        let min = episodeReturns.min()!
        let maxIndex = episodeReturns.firstIndex(of: max)!
        print("Step: \(totalStepCount) Time Elapsed: \(timeInterval) seconds Mean Reward: \(avgEpisodeReturns) max: \(max), min: \(min), max index: \(maxIndex)")
        episodeReturns.removeAll()
    }
    if totalStepCount != 0 && totalStepCount % 500000 == 0 {
        try? channel.setConfigurationParameters(timeScale: 1)
    }
    totalStepCount += 1
}, onEndOfEpisode: { model, reward in
    episodeReturn += reward
    if episode != 0 && episode % updateStep == 0 {
        agent.update(episodeCount: episode)
    }
    episodeReturns.append(episodeReturn)
    episodeReturn = 0
    episode += 1
})

//
//// try env.close()
