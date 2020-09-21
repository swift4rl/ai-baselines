//
//  BaseEnv.swift
//  App
//
//  Created by Sercan Karaoglu on 16/08/2020.
//

import Foundation
import TensorFlow

typealias AgentId = Int32
typealias BehaviorName = String

/**
Contains the data a single Agent collected since the last
simulation step.
 */
struct DecisionStep {
    let obs: [Tensor<Float32>]
    let reward: Float32
    let agentId: AgentId
    let actionMask: Optional<[Tensor<Bool>]>
    /**
    - Parameters:
     - obs is a list of numpy arrays observations collected by the agent.
     - reward is a float. Corresponds to the rewards collected by the agent
     since the last simulation step.
     - agentId is an int and an unique identifier for the corresponding Agent.
     - actionMask is an optional list of one dimensional array of booleans.
     Only available in multi-discrete action space type.
     Each array corresponds to an action branch. Each array contains a mask
     for each action of the branch. If true, the action is not available for
     the agent during this simulation step.
     */
    init(obs: [Tensor<Float32>], reward: Float32, agentId: AgentId, actionMask: Optional<[Tensor<Bool>]>) {
        self.obs = obs
        self.reward = reward
        self.agentId = agentId
        self.actionMask = actionMask
    }
}

/**
 Contains the data a batch of similar Agents collected since the last
 simulation step. Note that all Agents do not necessarily have new
 information to send at each simulation step. Therefore, the ordering of
 agents and the batch size of the DecisionSteps are not fixed across
 simulation steps.
 */
class DecisionSteps{
    let obs: [Tensor<Float32>]
    let reward: Tensor<Float32>
    let agentId: [AgentId]
    let actionMask: Optional<[Tensor<Bool>]>
    var _agentIdToIndex: Optional<[AgentId: Int]>
    /**
     - Returns: A Dict that maps agent_id to the index of those agents in
    this DecisionSteps.
    */
    var agentIdToIndex: [AgentId: Int] {
        if self._agentIdToIndex == Optional.none {
            self._agentIdToIndex = [:]
            for (aIdx, aId) in self.agentId.enumerated(){
                self._agentIdToIndex![aId] = aIdx
            }
        }
        return self._agentIdToIndex!
    }
    /**
     - Parameters:
         - obs is a list of numpy arrays observations collected by the batch of
         agent. Each obs has one extra dimension compared to DecisionStep: the
         first dimension of the array corresponds to the batch size of the batch.
         - reward is a float vector of length batch size. Corresponds to the
         rewards collected by each agent since the last simulation step.
         - agentId is an int vector of length batch size containing unique
         identifier for the corresponding Agent. This is used to track Agents
         across simulation steps.
         - actionMask is an optional list of two dimensional array of booleans.
         Only available in multi-discrete action space type.
         Each array corresponds to an action branch. The first dimension of each
         array is the batch size and the second contains a mask for each action of
         the branch. If true, the action is not available for the agent during
         this simulation step.
     */
    init(obs: [Tensor<Float32>], reward: Tensor<Float32>, agentId: [AgentId], actionMask: Optional<[Tensor<Bool>]>){
        self.obs = obs
        self.reward = reward
        self.agentId = agentId
        self.actionMask = actionMask
        self._agentIdToIndex = Optional.none
    }

    func len() -> Int {
        return self.agentId.count
    }

    /**
    returns the DecisionStep for a specific agent.
    :param agent_id: The id of the agent
    :returns: The DecisionStep
     */
    subscript(agentId: AgentId) -> DecisionStep? {
        if !(_agentIdToIndex?.keys.contains(agentId) ?? false){
            return Optional.none
        }
        let agentIndex = _agentIdToIndex![agentId]!
        var agentObs: [Tensor<Float32>] = []
        for batchedObs in self.obs {
            agentObs.append(batchedObs[agentIndex])
        }
        var _agentMask: [Tensor<Bool>] = []
        if self.actionMask != Optional.none {
            _agentMask = []
            for mask in self.actionMask! {
                _agentMask.append(mask[agentIndex])
            }
        }
        return DecisionStep(
            obs: agentObs,
            reward: self.reward.scalars[agentIndex],
            agentId: agentId,
            actionMask: _agentMask
        )
    }
    
    func iter() -> IndexingIterator<[AgentId]>{
        return self.agentId.makeIterator()
    }

    /**
      - Parameters:
        - spec: The BehaviorSpec for the DecisionSteps
      - Returns: an empty DecisionSteps.
     */
    static func empty<T>(spec: T) -> DecisionSteps where T: BehaviorSpec{
        var obs: [Tensor<Float32>] = []
        for shape in spec._observationShapes {
            var s = [0]
            s += shape
            obs.append(Tensor<Float32>(zeros: TensorShape(s)))
        }
        return DecisionSteps(
            obs: obs,
            reward: Tensor<Float32>(zeros: TensorShape([0])),
            agentId: [],
            actionMask: []
        )
    }

}
/**
 A NamedTuple to containing information about the observations and actions
 spaces for a group of Agents under the same behavior.
 */
protocol BehaviorSpec {
    associatedtype Scalar: TensorFlowNumeric
    
    var _observationShapes: [[Int]] { get set }
    var _actionType: ActionType { get set }
    var _actionShape: [Int] { get set }
    /// true if this Behavior uses discrete actions
    var isActionDiscrete: Bool { get }
    var isActionContinuous: Bool { get }
    /**
       Returns a Tuple of int corresponding to the number of possible actions
       for each branch (only for discrete actions). Will return None in
       for continuous actions.
    */
    var discreteActionBranches: [Int]? { get }
    
    /** the dimension of the action.
        - In the continuous case, will return the number of continuous actions.
        - In the (multi-)discrete case, will return the number of action.
        branches.
     */
    var actionSize: Int { get }
    
    init()
    init(observationShapes: [[Int]], actionShape: [Int])
    
    func createRandomAction(nAgents: Int) -> Tensor<Scalar>
    func createEmptyAction(nAgents: Int) -> Tensor<Scalar>
}

extension BehaviorSpec {
    /**
    - Parameters:
     - observation_shapes: is a List of Tuples of int : Each Tuple corresponds
     to an observation's dimensions. The shape tuples have the same ordering as
     the ordering of the DecisionSteps and TerminalSteps.
     - action_type: is the type of data of the action. it can be discrete or
     continuous. If discrete, the action tensors are expected to be int32. If
     continuous, the actions are expected to be float32.
     - action_shape is:
       - An int in continuous action space corresponding to the number of
     floats that constitute the action.
       - A Tuple of int in discrete action space where each int corresponds to
       the number of discrete actions available to the agent.
    */
    init(observationShapes: [[Int]], actionType: ActionType, actionShape: [Int]) {
        self.init()
        _observationShapes = observationShapes
        _actionType = actionType
        _actionShape = actionShape
    }
    
    /**
    Generates a numpy array corresponding to an empty action (all zeros)
    for a number of agents.
    - Parameters:
     - n_agents: The number of agents that will have actions generated
    */
    func createEmptyAction(nAgents: Int) -> Tensor<Scalar> {
        return Tensor<Scalar>(zeros: [nAgents, actionSize])
    }
    
}

struct BehaviorSpecContinousAction: BehaviorSpec {
    
    typealias Scalar = Float32
    
    init() {}

    init(observationShapes: [[Int]], actionShape: [Int]) {
        self.init(observationShapes: observationShapes, actionType: ActionType.CONTINUOUS, actionShape: actionShape)
    }
    
    var isActionDiscrete: Bool = false
    
    var isActionContinuous: Bool = true
    
    var _observationShapes: [[Int]] = []
    
    var _actionType: ActionType = ActionType.CONTINUOUS
    
    var _actionShape: [Int] = []
    
    var actionSize: Int {
        return self._actionShape[0]
    }
    
    var discreteActionBranches: [Int]? {
        return Optional.none
    }
    
    func createRandomAction(nAgents: Int) -> Tensor<Float32> {
        let action = Tensor<Float32>.init(
            randomUniform: [nAgents, self.actionSize],
            lowerBound: Tensor(-1.0),
            upperBound: Tensor(1.0)
        )
        return action
    }
    
}

struct BehaviorSpecDiscreteAction: BehaviorSpec {

    typealias Scalar = Int32
    
    init() {}

    init(observationShapes: [[Int]], actionShape: [Int]) {
        self.init(observationShapes: observationShapes, actionType: ActionType.DISCRETE, actionShape: actionShape)
    }
    
    var isActionDiscrete: Bool = true
    
    var isActionContinuous: Bool = false
    
    var _observationShapes: [[Int]] = []
    
    var _actionType: ActionType = ActionType.DISCRETE
    
    var _actionShape: [Int] = []
    
    var actionSize: Int {
        return self._actionShape.count
    }
    
    var discreteActionBranches: [Int]? {
        return Optional.some( self._actionShape )
    }
    
    /**
    Generates a numpy array corresponding to a random action (either discrete
    or continuous) for a number of agents.
     - Parameters:
        - n_agents: The number of agents that will have actions generated
    */
    func createRandomAction(nAgents: Int) -> Tensor<Int32> {
        let action = Tensor<Int32>(
            stacking: (0...self.actionSize-1).map{ i in
                let branchSize = self.discreteActionBranches!
                return Tensor<Int32>.init(
                        randomUniform: TensorShape(nAgents),
                        lowerBound: Tensor(Int32(0)),
                        upperBound: Tensor(Int32(branchSize[i]))
                    )
                },
            alongAxis: 1
        )
        return action
    }

}


protocol BaseEnv {
    
    associatedtype BehaviorSpecType: BehaviorSpec
    
    /// Signals the environment that it must move the simulation forward
    /// by one step.
    func step() -> Void
    
    /// Signals the environment that it must reset the simulation
    func reset() -> Void
    
    /// Signals the environment that it must close.
    func close() -> Void
    
    /**
     Returns a Mapping from behavior names to behavior specs.
     Agents grouped under the same behavior name have the same action and
     observation specs, and are expected to behave similarly in the
     environment.
     Note that new keys can be added to this mapping as new policies are instantiated.
     */
    func behaviorSpecs() -> [String: BehaviorSpecType]
    
    /**
     Sets the action for all of the agents in the simulation for the next
     step. The Actions must be in the same order as the order received in
     the DecisionSteps.
     - Parameters:
        - behaviorName: The name of the behavior the agents are part of
        - action: A two dimensional Tensor corresponding to the action
     (either int or float)
     */
    func setActions(behaviorName: BehaviorName, action: Tensor<Int32>) -> Void
    
    /**
    Sets the action for one of the agents in the simulation for the next
    step.
     - Parameters:
        - behaviorName: The name of the behavior the agent is part of
        - agentId: The id of the agent the action is set for
        - action: A one dimensional Tensor corresponding to the action
    (either int or float)
    */
    func setActionForAgent(
        behaviorName: BehaviorName,
        agentId: AgentId,
        action: Tensor<Int32>
    ) -> Void
        

    /**
     Retrieves the steps of the agents that requested a step in the
     simulation.
      - Parameters:
        - behaviorName: The name of the behavior the agents are part of
      - Returns: A tuple containing :
        - A DecisionSteps NamedTuple containing the observations,
      the rewards, the agent ids and the action masks for the Agents
      of the specified behavior. These Agents need an action this step.
        - A TerminalSteps NamedTuple containing the observations,
      rewards, agent ids and interrupted flags of the agents that had their
      episode terminated last step.
     */
    func getSteps(behaviorName: BehaviorName) -> (DecisionSteps, TerminalSteps)

}

enum BaseEnvError: Error {
    case InvalidKeyWithinTerminalSteps(message: String)
}
/// Contains the data a single Agent collected when its episode ended
struct TerminalStep {
    
    /// obs is a array of Tensors observations collected by the agent.
    var obs: [Tensor<Float>]
    
    /// reward is a float. Corresponds to the rewards collected by the agent
    /// since the last simulation step.
    var reward: Float
    
    /// interrupted is a bool. Is true if the Agent was interrupted since the last
    /// decision step. For example, if the Agent reached the maximum number of steps for
    /// the episode.
    var interrupted: Bool
    
    /// agentId is an int and an unique identifier for the corresponding Agent.
    var agentId: AgentId
    
    init(obs: [Tensor<Float>], reward: Float, interrupted: Bool, agentId: AgentId) {
        self.obs = obs
        self.reward = reward
        self.interrupted = interrupted
        self.agentId = agentId
    }
}

/**
 Contains the data a batch of Agents collected when their episode
 terminated. All Agents present in the TerminalSteps have ended their
 episode.
 */
class TerminalSteps{
    
    var obs: [Tensor<Float32>]
    var reward: Tensor<Float32>
    var interrupted: Tensor<Bool>
    var agentId: [AgentId]
    var _agentIdToIndex: [AgentId: Int]? = Optional<[AgentId: Int]>.none
    /**
     - Returns: A Dict that maps agent_id to the index of those agents in
    this TerminalSteps.
    */
    var agentIdToIndex: [AgentId: Int] {
        get {
            if _agentIdToIndex == Optional<[AgentId: Int]>.none {
                _agentIdToIndex = Optional<[AgentId: Int]>.some([:])
            }
            for (aIdx, aId) in self.agentId.enumerated() {
                self._agentIdToIndex![aId] = aIdx
            }
            return self._agentIdToIndex!
        }
    }
    /**
     -   Parameters:
         - obs: is a list of Tensor arrays observations collected by the batch of
         agent. Each obs has one extra dimension compared to DecisionStep: the
         first dimension of the array corresponds to the batch size of the batch.
         - reward: is a float vector of length batch size. Corresponds to the
         rewards collected by each agent since the last simulation step.
         - interrupted: is an array of booleans of length batch size. Is true if the
         associated Agent was interrupted since the last decision step. For example, if the
         Agent reached the maximum number of steps for the episode.
         - agentId:  is an int vector of length batch size containing unique
         identifier for the corresponding Agent. This is used to track Agents
         across simulation steps.
     */
    init(obs: [Tensor<Float32>], reward: Tensor<Float32>, interrupted: Tensor<Bool>, agentId: [AgentId]) {
        self.obs = obs
        self.reward = reward
        self.interrupted = interrupted
        self.agentId = agentId
    }

    func len() -> Int {
        return self.agentId.count
    }

    /**
    returns the TerminalStep for a specific agent.
    :param agent_id: The id of the agent
    :returns: obs, reward, done, agent_id and optional action mask for a
    specific agent
    */
    subscript(agentId: AgentId) -> TerminalStep? {
        if !(_agentIdToIndex?.keys.contains(agentId) ?? false) {
            return Optional.none
        }
        let agentIndex = _agentIdToIndex![agentId]!
        var agentObs: [Tensor<Float32>] = []
        for batchedObs in self.obs {
            agentObs.append(batchedObs[agentIndex])
        }
        
        return Optional.some(
            TerminalStep(
                obs: agentObs,
                reward: self.reward.scalars[agentIndex],
                interrupted: self.interrupted.scalars[agentIndex],
                agentId: agentId
            )
        )
    }

    func iter() -> IndexingIterator<[AgentId]>{
        return self.agentId.makeIterator()
    }

    /**
     - Parameters:
        - spec: The BehaviorSpec for the TerminalSteps
     - Returns: an empty TerminalSteps.
     */
    static func empty<T>(spec: T) -> TerminalSteps where T: BehaviorSpec{
        var obs: [Tensor<Float32>] = []
        for shape in spec._observationShapes {
            var s = [0]
            s += shape
            obs.append(
                Tensor<Float32>(zeros: TensorShape(s))
            )
        }
        
        return TerminalSteps(
            obs: obs,
            reward: Tensor<Float32>(zeros: TensorShape([0])),
            interrupted: Tensor<Bool>(repeating: false, shape: TensorShape([0])),
            agentId: []
        )
    }
}

enum ActionType {
    case DISCRETE, CONTINUOUS
}
