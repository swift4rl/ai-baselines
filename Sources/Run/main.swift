////
////  main.swift
////  App
////
////  Created by Sercan Karaoglu on 01/08/2020.
////

import PythonKit
import TensorFlow
import Foundation
import environments
// Initialize Python. This comment is a hook for internal use, do not remove.
let tf = Python.import("tensorflow")

let dirPath = "/Users/sercankaraoglu/log/ai-baselines/"

let appPath = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Desktop")
    .appendingPathComponent("cartpole.app")
    .appendingPathComponent("Contents")
    .appendingPathComponent("MacOS")
    .appendingPathComponent("ML experiment")

print(appPath)

let channel = EngineConfigurationChannel()
try channel.setConfigurationParameters(timeScale: 3.0)
let unityEnv = UnityContinousEnvironment(filename: appPath, basePort: 5004, sideChannels: [])
guard let env = UnityToGymWrapper(env: unityEnv, uint8Visual: false, flattenBranched: false, allowMultipleObs: false) else {
    throw UnityException.UnityGymException(reason: "failed to create unitygymwrapper")
}

let observationSize: Int =  8 //Int(env.observationSpace?.shape[0] ?? 0)
let actionCount: Int = 2

// Hyperparameters
/// The !size of the hidden layer of the 2-layer actor network and critic network. The actor network
/// has the shape observationSize - hiddenSize - actionCount, and the critic network has the same
/// shape but with a single output node.
let hiddenSize: Int = 64
/// The learning rate for both the actor and the critic.
let learningRate: Float = 0.05
/// The discount factor. This measures how much to "discount" the future rewards
/// that the agent will receive. The discount factor must be from 0 to 1
/// (inclusive). Discount factor of 0 means that the agent only considers the
/// immediate reward and disregards all future rewards. Discount factor of 1
/// means that the agent values all rewards equally, no matter how distant
/// in the future they may be. Denoted gamma in the PPO paper.
let discount: Float = 0.99
/// Number of epochs to run minibatch updates once enough trajectory segments are collected. Denoted
/// K in the PPO paper.
let epochs: Int = 40
/// Parameter to clip the probability ratio. The ratio is clipped to [1-clipEpsilon, 1+clipEpsilon].
/// Denoted epsilon in the PPO paper.
let clipEpsilon: Float = 0.1
/// Coefficient for the entropy bonus added to the objective. Denoted c_2 in the PPO paper.
let entropyCoefficient: Float = 0.0001
/// Maximum number of episodes to train the agent. The training is terminated
/// early if maximum score is achieved consecutively 10 times.
let maxEpisodes: Int = 10000
/// Maximum timestep per episode.
let maxTimesteps: Int = 10000
/// The length of the trajectory segment. Denoted T in the PPO paper.
let updateTimestep: Int = 100


var agent: PPOAgent = PPOAgent(
    observationSize: observationSize,
    hiddenSize: hiddenSize,
    actionCount: actionCount,
    learningRate: learningRate,
    discount: discount,
    epochs: epochs,
    clipEpsilon: clipEpsilon,
    entropyCoefficient: entropyCoefficient
)

// Sets up tensorboard form python, since its not int swift tensorflow api yet
let formatter = DateFormatter()
// initially set the format based on your datepicker date / server String
formatter.dateFormat = "yyyy-MM-dd HH:mm:ss"
let dateString = formatter.string(from: Date())
let logDir = dirPath + "logs/scalars/" + dateString
print("Saves tensorboard logs to \(logDir)")
let file_writer = tf.summary.create_file_writer(logDir + "/metrics")
file_writer.set_as_default()


// Training loop
var timestep: Int = 0
var episodeReturn: Float = 0
var episodeReturns: [Float] = []
var maxEpisodeReturn: Float = -1
for episodeIndex in 0..<maxEpisodes {
    if case var .SingleStepResult(state, _, _, _) = try env.reset() {
        var isDone: Bool = false
        var reward: Float
        for timeStep in 0..<maxTimesteps {
            timestep += 1
            state = state.reshaped(to: TensorShape(1, state.shape[0]))
            (state, isDone, reward) = try agent.step(env: env, state: state)

            if timestep % updateTimestep == 0 {
                print("training...")
                agent.update()
                timestep = 0
            }

            episodeReturn += reward
            if isDone {
                episodeReturns.append(episodeReturn)
//                print(String(format: "Episode: %d | Return: %.2f | Timesteps: %d", episodeIndex, episodeReturn, timeStep))
//                tf.summary.scalar("episode_return", data: episodeReturn, step: episodeIndex)
//                tf.summary.scalar("episode_timesteps", data: timeStep, step: episodeIndex)
                episodeReturn = 0
                break
            }
        }
        if episodeReturns.count == 10 {
            let avgEpisodeReturns = episodeReturns.reduce(0, +) / 10.0
            print(String(format: "Average returns of last 10 episodes: %.2f", avgEpisodeReturns))
            episodeReturns.removeAll()
        }
    }
        
}

try env.close()
