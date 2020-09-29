//
//  BaseEnv.swift
//  App
//
//  Created by Sercan Karaoglu on 16/08/2020.
//

import Foundation
import TensorFlow
import Logging
import Version

public typealias AgentId = Int32
public typealias BehaviorName = String

protocol Steps {
    var obs: [Tensor<Float32>] { get set }
    var reward: Tensor<Float32> { get set }
    var agentId: [AgentId] { get set }
}
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
public class DecisionSteps: Steps {
    var obs: [Tensor<Float32>]
    var reward: Tensor<Float32>
    var agentId: [AgentId]
    var actionMask: Optional<[Tensor<Bool>]>
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
        for shape in spec.observationShapes {
            var s = [0]
            s += shape.map({Int($0)})
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
public protocol BehaviorSpec {
    associatedtype Scalar: TensorFlowNumeric & Hashable & Strideable
    
    var observationShapes: [[Int32]] { get set }
    var actionType: ActionType { get set }
    var actionShape: [Int32] { get set }
    /// true if this Behavior uses discrete actions
    var isActionDiscrete: Bool { get }
    var isActionContinuous: Bool { get }
    /**
       Returns a Tuple of int corresponding to the number of possible actions
       for each branch (only for discrete actions). Will return None in
       for continuous actions.
    */
    var discreteActionBranches: [Scalar]? { get }
    
    /** the dimension of the action.
        - In the continuous case, will return the number of continuous actions.
        - In the (multi-)discrete case, will return the number of action.
        branches.
     */
    var actionSize: Int { get }
    
    init()
    init(observationShapes: [[Int32]], actionShape: [Int32])
    
    func createRandomAction(nAgents: Int) -> Tensor<Scalar>
    func createEmptyAction(nAgents: Int) -> Tensor<Scalar>
    
    static func create(observationShapes: [[Int32]], actionShape: [Int32]) -> Self
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
    init(observationShapes: [[Int32]], actionType: ActionType, actionShape: [Int32]) {
        self.init()
        self.observationShapes = observationShapes
        self.actionType = actionType
        self.actionShape = actionShape
    }
    
    /**
    Generates a numpy array corresponding to an empty action (all zeros)
    for a number of agents.
    - Parameters:
     - n_agents: The number of agents that will have actions generated
    */
    public func createEmptyAction(nAgents: Int) -> Tensor<Scalar> {
        return Tensor<Scalar>(zeros: [nAgents, actionSize])
    }
    
}

public struct BehaviorSpecContinousAction: BehaviorSpec {
    
    public typealias Scalar = Float32
    
    public static func create(observationShapes: [[Int32]], actionShape: [Int32]) -> BehaviorSpecContinousAction {
        return BehaviorSpecContinousAction(observationShapes: observationShapes, actionShape: actionShape)
    }
    
    public init() {}

    public init(observationShapes: [[Int32]], actionShape: [Int32]) {
        self.init(observationShapes: observationShapes, actionType: ActionType.CONTINUOUS, actionShape: actionShape)
    }
    
    public var isActionDiscrete: Bool = false
    
    public var isActionContinuous: Bool = true
    
    public var observationShapes: [[Int32]] = []
    
    public var actionType: ActionType = ActionType.CONTINUOUS
    
    public var actionShape: [Int32] = []
    
    public var actionSize: Int {
        return Int(self.actionShape[0])
    }
    
    public var discreteActionBranches: [Scalar]? {
        return Optional.none
    }
    
    public func createRandomAction(nAgents: Int) -> Tensor<Float32> {
        let action = Tensor<Float32>.init(
            randomUniform: [nAgents, self.actionSize],
            lowerBound: Tensor(-1.0),
            upperBound: Tensor(1.0)
        )
        return action
    }
    
}

public struct BehaviorSpecDiscreteAction: BehaviorSpec {
    
    public typealias Scalar = Int32
    
    public static func create(observationShapes: [[Int32]], actionShape: [Int32]) -> BehaviorSpecDiscreteAction {
        return BehaviorSpecDiscreteAction(observationShapes: observationShapes, actionShape: actionShape)
    }
    
    public init() {}

    public init(observationShapes: [[Int32]], actionShape: [Int32]) {
        self.init(observationShapes: observationShapes, actionType: ActionType.DISCRETE, actionShape: actionShape)
    }
    
    public var isActionDiscrete: Bool = true
    
    public var isActionContinuous: Bool = false
    
    public var observationShapes: [[Int32]] = []
    
    public var actionType: ActionType = ActionType.DISCRETE
    
    public var actionShape: [Int32] = []
    
    public var actionSize: Int {
        return self.actionShape.count
    }
    
    public var discreteActionBranches: [Scalar]? {
        return Optional.some( self.actionShape )
    }
    
    /**
    Generates a numpy array corresponding to a random action (either discrete
    or continuous) for a number of agents.
     - Parameters:
        - n_agents: The number of agents that will have actions generated
    */
    public func createRandomAction(nAgents: Int) -> Tensor<Int32> {
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

struct Defaults {
    static let logger = Logger(label: "environments.UnityEnvironment")
    /**
    Communication protocol version.
    When connecting to C#, this must be compatible with Academy.k_ApiVersion.
    We follow semantic versioning on the communication version, so existing
    functionality will work as long the major versions match.
    This should be changed whenever a change is made to the communication protocol.
     */
    static let API_VERSION = "1.0.0"
    
    /**
     Default port that the editor listens on. If an environment executable
     isn't specified, this port will be used.
     */
    static let DEFAULT_EDITOR_PORT = 5004
    /* Default base port for environments. Each environment will be offset from this
     by it's worker_id.
     */
    static let BASE_ENVIRONMENT_PORT = 5005
    /// Command line argument used to pass the port to the executable environment.
    static let _PORT_COMMAND_LINE_ARG = "--mlagents-port"
}

public protocol BaseEnv {
    
    associatedtype BehaviorSpecImpl: BehaviorSpec
    
    typealias BehaviorMapping = [BehaviorName: BehaviorSpecImpl]
    
    var props: Props<BehaviorSpecImpl> { get set }
    
    /// Signals the environment that it must move the simulation forward
    /// by one step.
    mutating func step() throws -> Void
    
    /// Signals the environment that it must reset the simulation
    mutating func reset() throws -> Void
    
    /// Signals the environment that it must close.
    mutating func close() throws -> Void
    
    /**
     Returns a Mapping from behavior names to behavior specs.
     Agents grouped under the same behavior name have the same action and
     observation specs, and are expected to behave similarly in the
     environment.
     Note that new keys can be added to this mapping as new policies are instantiated.
     */
    var behaviorSpecs: BehaviorMapping { get }
    
    /**
     Sets the action for all of the agents in the simulation for the next
     step. The Actions must be in the same order as the order received in
     the DecisionSteps.
     - Parameters:
        - behaviorName: The name of the behavior the agents are part of
        - action: A two dimensional Tensor corresponding to the action
     (either int or float)
     */
    mutating func setActions(behaviorName: BehaviorName, action: Tensor<BehaviorSpecImpl.Scalar>) throws -> Void
    
    /**
    Sets the action for one of the agents in the simulation for the next
    step.
     - Parameters:
        - behaviorName: The name of the behavior the agent is part of
        - agentId: The id of the agent the action is set for
        - action: A one dimensional Tensor corresponding to the action
    (either int or float)
    */
    mutating func setActionForAgent(behaviorName: String, agentId: AgentId, action: Tensor<BehaviorSpecImpl.Scalar>) throws -> Void
        

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
    func getSteps(behaviorName: BehaviorName) throws -> (DecisionSteps, TerminalSteps)?

}

extension BaseEnv {
    
    public var port: Int {
        get { return props.port }
        set { props.port = newValue }
    }
    
    public var behaviorSpecs: BehaviorMapping {
        get { return props.envSpecs }
    }
    
    var isFirstMessage: Bool {
        get { return props.isFirstMessage }
        set { props.isFirstMessage = newValue }
    }
    
    var loaded: Bool {
        get { return props.loaded }
        set { props.loaded = newValue }
    }
    
    var noGraphics: Bool {
        get { return props.noGraphics }
        set { props.noGraphics = newValue }
    }
    
    var envSpecs: [String: BehaviorSpecImpl] {
        get { return props.envSpecs }
        set { props.envSpecs = newValue }
    }
    
    var envState: [String: (DecisionSteps, TerminalSteps)] {
        get { return props.envState }
        set { props.envState = newValue }
    }
    
    var envActions: [String: Tensor<BehaviorSpecImpl.Scalar>] {
        get { return props.envActions }
        set { props.envActions = newValue }
    }
    
    var sideChannelManager: SideChannelManager {
        get { return props.sideChannelManager }
        set { props.sideChannelManager = newValue }
    }
    
    var communicator: RpcCommunicator {
        get { return props.communicator }
        set { props.communicator = newValue }
    }
    
    static var logger: Logger {
        get { return Defaults.logger}
    }
    
    static func raiseVersionException(unityComVer: String) throws -> Void {
        throw UnityException.UnityEnvironmentException(reason: """
            The communication API version is not compatible between Unity and Swift.
            Swift API: \(Defaults.API_VERSION), Unity API: \(unityComVer).\n
        """)
    }
    
    static func checkCommunicationCompatibility(unityComVer: String, swiftApiVersion: String, unityPackageVersion: String) -> Bool {
        let unityCommunicatorVersion: Version = try! Version(unityComVer)
        let apiVersion: Version = try! Version(swiftApiVersion)
        if unityCommunicatorVersion.major != apiVersion.major{
            /// Major versions mismatch.
            return false
        } else if unityCommunicatorVersion.minor != apiVersion.minor {
            /// Non-beta minor versions mismatch.  Log a warning but allow execution to continue.
            logger.warning("""
                WARNING: The communication API versions between Unity and python differ at the minor version level.
                Swift API: \(swiftApiVersion), Unity API: \(unityCommunicatorVersion).\n
                This means that some features may not work unless you upgrade the package with the lower version.
                """)
        } else {
            logger.info("""
                Connected to Unity environment with package version \(unityPackageVersion)
                and communication version \(unityComVer)
            """)
        }
        return true
    }
    
    static func getCapabilitiesProto() -> CommunicatorObjects_UnityRLCapabilitiesProto{
        var capabilities = CommunicatorObjects_UnityRLCapabilitiesProto()
        capabilities.baseRlcapabilities = true
        return capabilities
    }
    
    static func warnCsharpBaseCapabilities(
        caps: CommunicatorObjects_UnityRLCapabilitiesProto, unityPackageVer: String, swiftPackageVer: String
    ) -> Void {
        if !caps.baseRlcapabilities {
            logger.warning("""
                WARNING: The Unity process is not running with the expected base Reinforcement Learning
                capabilities. Please be sure upgrade the Unity Package to a version that is compatible with this
                swift package.\n
                Python package version: \(swiftPackageVer), C# package version: \(unityPackageVer)
            """)
        }
    }
    
    public init?(
        filename: String?,
        workerId: Int = 0,
        basePort: Int?,
        seed: Int32 = 0,
        noGraphics: Bool = false,
        timeoutWait: Int = 60,
        additionalArgs: [String]? = Optional.none,
        sideChannels: [SideChannel]? = Optional.none,
        logFolder: String? = Optional.none
        ) throws {
        try self.init(filename: filename, basePort: basePort)
        if let filename = filename {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: filename)
            var args: [String] = []
            if self.noGraphics {
                args += ["-nographics", "-batchmode"]
            }
            args += [Defaults._PORT_COMMAND_LINE_ARG, String(self.port)]
            if let logFolder = logFolder {
                args += ["-logFile", "\(logFolder)/Player-\(workerId).log"]
            }
            if let aArgs = additionalArgs {
                args += aArgs
            }
            task.arguments = args
            let outputPipe = Pipe()
            let errorPipe = Pipe()

            task.standardOutput = outputPipe
            task.standardError = errorPipe
            try task.run()
        }
        self.sideChannelManager = try SideChannelManager(sideChannels: sideChannels)
        let port: Int = Defaults.DEFAULT_EDITOR_PORT
        self.communicator = RpcCommunicator(workerId: workerId, port: port)
        var rlInitParametersIn = CommunicatorObjects_UnityRLInitializationInputProto()
        rlInitParametersIn.seed = seed
        rlInitParametersIn.communicationVersion = Defaults.API_VERSION
        rlInitParametersIn.packageVersion = "0.20.0"
        rlInitParametersIn.capabilities = Self.getCapabilitiesProto()
        let acaOutput = self.sendAcademyParameters(initParameters: rlInitParametersIn)
        let acaParams = acaOutput?.rlInitializationOutput
        
        if let unityComVer = acaParams?.communicationVersion,
           let unityPackageVersion = acaParams?.packageVersion,
           !Self.checkCommunicationCompatibility(unityComVer: unityComVer, swiftApiVersion: Defaults.API_VERSION, unityPackageVersion: unityPackageVersion){
            try Self.raiseVersionException(unityComVer: unityComVer)
        }
        self.isFirstMessage = true
        if let output = acaOutput {
            self.updateBehaviorSpecs(output: output)
        }
    }
    
    public mutating func step() throws -> Void {
        if self.isFirstMessage {
            return try self.reset()
        }
        if !self.loaded {
            throw UnityException.UnityEnvironmentException(reason: "No Unity environment is loaded.")
        }
        for groupName in self.envSpecs.keys {
            if !(self.envActions.keys.contains(groupName)) {
                var nAgents = 0
                if self.envState.keys.contains(groupName){
                    nAgents = self.envState[groupName]?.0.len() ?? 0
                }
                self.envActions[groupName] = self.envSpecs[groupName]?.createEmptyAction(nAgents: nAgents)
            }
        }
        let stepInput = self.generateStepInput(vectorAction: self.envActions)
        // todo: measure time here
        if let outputs = self.communicator.exchange(inputs: stepInput) {
            self.updateBehaviorSpecs(output: outputs)
            let rlOutput = outputs.rlOutput
            try self.updateState(output: rlOutput)
            self.envActions.removeAll()
        } else {
            throw UnityException.UnityCommunicatorStoppedException(reason: "Communicator has exited.")
        }
    }
    
    public mutating func reset() throws -> Void {
        if self.loaded {
            if let outputs = self.communicator.exchange(inputs: self.generateResetInput()){
                self.updateBehaviorSpecs(output: outputs)
                let rlOutput = outputs.rlOutput
                try self.updateState(output: rlOutput)
                self.isFirstMessage = false
                self.envActions.removeAll()
            } else {
                throw UnityException.UnityCommunicatorStoppedException(reason: "Communicator has exited.")
            }
        } else{
            throw UnityException.UnityEnvironmentException(reason: "No Unity environment is loaded.")
        }
    }
    
    /// Sends a shutdown signal to the unity environment, and closes the socket connection.
    public mutating func close() throws -> Void {
        if self.loaded {
            self.loaded = false
            self.communicator.close()
        } else{
            throw UnityException.UnityEnvironmentException(reason: "No Unity environment is loaded.")
        }
    }
    
    public mutating func setActions(behaviorName: BehaviorName, action: Tensor<BehaviorSpecImpl.Scalar>) throws -> Void {
        try self.assertBehaviorExists(behaviorName: behaviorName)
        if !self.envState.keys.contains(behaviorName) {
            return
        }
        if let actionSize = self.envSpecs[behaviorName]?.actionSize, let decisionStepLen = self.envState[behaviorName]?.0.len(){
            let expectedShape = TensorShape(decisionStepLen, actionSize)
            if action.shape != expectedShape{
                throw UnityException.UnityActionException(reason: """
                    The behavior \(behaviorName) needs an input of dimension \(expectedShape) for
                    (<number of agents>, <action size>) but received input of
                    dimension \(action.shape)
                    """)
            }
            self.envActions[behaviorName] = action
        }
    }
    
    func assertBehaviorExists(behaviorName: String) throws -> Void {
        if !self.envSpecs.keys.contains(behaviorName) {
            throw UnityException.UnityActionException(reason: """
                The group \(behaviorName) does not correspond to an existing agent group
                in the environment
            """)
        }
    }
    
    public mutating func setActionForAgent(behaviorName: String, agentId: AgentId, action: Tensor<BehaviorSpecImpl.Scalar>) throws -> Void {
        try self.assertBehaviorExists(behaviorName: behaviorName)
        if !self.envState.keys.contains(behaviorName){
            return
        }
        if let spec = self.envSpecs[behaviorName]{
            let expectedShape = TensorShape([ spec.actionSize ])
            if action.shape != expectedShape {
                throw UnityException.UnityActionException(reason: """
                    The Agent \(agentId) with BehaviorName \(behaviorName) needs an input of dimension
                    \(expectedShape) but received input of dimension \(action.shape)
                    """
                )
            }

            if  !self.envActions.keys.contains(behaviorName), let nAgents = self.envState[behaviorName]?.0.len() {
                self.envActions[behaviorName] = spec.createEmptyAction(nAgents: nAgents)
            }
            
            guard let index = self.envState[behaviorName]?.0.agentId.firstIndex(where: {$0 == agentId}) else {
                throw UnityException.UnityEnvironmentException(reason: "agent_id \(agentId) is did not request a decision at the previous step")
            }
            self.envActions[behaviorName]?[index] = action
        }
    }
    
    public func getSteps(behaviorName: BehaviorName) throws -> (DecisionSteps, TerminalSteps)? {
        try self.assertBehaviorExists(behaviorName: behaviorName)
        return self.envState[behaviorName]
    }
    
    func generateStepInput(vectorAction: [String: Tensor<BehaviorSpecImpl.Scalar>]) -> CommunicatorObjects_UnityInputProto {
        var rlIn = CommunicatorObjects_UnityRLInputProto()
        for b in vectorAction.keys {
            let nAgents = self.envState[b]?.0.len() ?? 0
            if nAgents == 0 {
                continue
            }
            for i in 0 ..< nAgents{
                var action = CommunicatorObjects_AgentActionProto()
                action.vectorActions = vectorAction[b]?[i].scalar as! [Float32]
                rlIn.agentActions[b]?.value += [action]
                rlIn.command = CommunicatorObjects_CommandProto.step
            }
        }
        rlIn.sideChannel = self.sideChannelManager.generateSideChannelMessages()
        
        return self.wrapUnityInput(rlInput: rlIn)
    }
    
    func generateResetInput() -> CommunicatorObjects_UnityInputProto {
        var rlIn = CommunicatorObjects_UnityRLInputProto()
        rlIn.command = CommunicatorObjects_CommandProto.reset
        rlIn.sideChannel = self.sideChannelManager.generateSideChannelMessages()
        return self.wrapUnityInput(rlInput: rlIn)
    }
    
    mutating func updateBehaviorSpecs(output: CommunicatorObjects_UnityOutputProto) -> Void {
        let initOutput = output.rlInitializationOutput
        for brainParam in initOutput.brainParameters {
            let agentInfos = output.rlOutput.agentInfos[brainParam.brainName]
            if let value = agentInfos?.value{
                let agent = value[0]
                let newSpec: BehaviorSpecImpl = behaviorSpecFromProto(brainParamProto: brainParam, agentInfo: agent)
                self.envSpecs[brainParam.brainName] = newSpec
                Self.logger.info("Connected new brain:\n \(brainParam.brainName)")
            }
        }
    }
    
    /// Collects experience information from all external brains in environment at current step.
    mutating func updateState(output: CommunicatorObjects_UnityRLOutputProto) throws -> Void {
        for brainName in self.envSpecs.keys {
            if output.agentInfos.keys.contains(brainName) {
                if let agentInfo = output.agentInfos[brainName], let envSpec = self.envSpecs[brainName] {
                    self.envState[brainName] = try stepsFromProto(agentInfoList: agentInfo.value, behaviorSpec: envSpec)
                }
            } else {
                if let envSpec =  self.envSpecs[brainName] {
                    self.envState[brainName] = (DecisionSteps.empty(spec: envSpec), TerminalSteps.empty(spec: envSpec))
                }
            }
        }
        try self.sideChannelManager.processSideChannelMessage(message: output.sideChannel)
    }
    func sendAcademyParameters(initParameters: CommunicatorObjects_UnityRLInitializationInputProto) -> CommunicatorObjects_UnityOutputProto? {
        var inputs = CommunicatorObjects_UnityInputProto()
        inputs.rlInitializationInput = initParameters
        return self.communicator.initialize(inputs: inputs)
    }
    func wrapUnityInput(rlInput: CommunicatorObjects_UnityRLInputProto) -> CommunicatorObjects_UnityInputProto {
        var result = CommunicatorObjects_UnityInputProto()
        result.rlInput = rlInput
        return result
    }

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
public class TerminalSteps: Steps {
    
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
        for shape in spec.observationShapes {
            var s = [0]
            s += shape.map({Int($0)})
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

public enum ActionType {
    case DISCRETE, CONTINUOUS
}
