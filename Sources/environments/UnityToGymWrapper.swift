//
//  UnityToGymWrapper.swift
//  App
//
//  Created by Sercan Karaoglu on 16/08/2020.
//

import Foundation
import Logging
import TensorFlow

enum StepResult<T: TensorFlowNumeric>{
    case SingleStepResult(observation: Tensor<T>, reward: Float32, done: Bool, info:[String:AnyObject])
    case MultiStepResult(observation: [Tensor<T>], reward: Float32, done: Bool, info:[String:AnyObject])
}

protocol Space {
}

struct Discrete: Space {
    init(_ branch: Int32) {
        
    }
}

struct MultiDiscrete: Space {
    init(_ branches: [Int32]) {
        
    }
}

struct Box<Scalar: TensorFlowScalar>: Space {
    init(min: Tensor<Scalar>, max: Tensor<Scalar>) {
    }
    init(min: Scalar, max: Scalar, shape: [Int32]?) {
    }
}

struct Tuple: Space {
    init(_ spaces: [Space]) {
    }
}


/**
 Flattens branched discrete action spaces into single-branch discrete action spaces.
 */
class ActionFlattener<T : Numeric & Hashable & Comparable & Strideable> {
    var actionSpace: Space
    var actionShape: [T]
    var actionLookup: [T: [T]]
    
    /**
    Initialize the flattener.
     - Parameters:
        - branchedActionSpace: A List containing the sizes of each branch of the action
    space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
    */
    init(_ branchedActionSpace: [T]) {
        self.actionShape = branchedActionSpace
        self.actionLookup = Self.createLookup(self.actionShape)
        self.actionSpace = Discrete(Int32(self.actionLookup.count))
    }

    /**
    Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
    Each key in the Dict maps to one unique set of branched actions, and each value
    contains the List of branched actions.
    */
    static func createLookup(_ branchedActionSpace: [T]) -> [T: [T]] {
        let possibleVals: [[T]] = branchedActionSpace.map{Array(stride(from: T.init(exactly: 0)!, to: $0, by: 1))}
        let allActions = Array(Product(possibleVals))
        
        return allActions.enumerated().reduce(into: [:]){map, el in
            map[T(exactly: el.0)!] = el.1
        }
        
    }

    /**
    Convert a scalar discrete action into a unique set of branched actions.
     - Parameters:
        - action: A scalar value representing one of the discrete actions.
    - Returns:
        - The List containing the branched actions.
    */
    func lookupAction(_ action: T)-> [T]?{
        return self.actionLookup[action]
    }
    
    func lookupAction(_ action: Tensor<T>) -> Tensor<T> where T: TensorFlowScalar{
        var act = action
        if let a = action.scalar,
           let lA = lookupAction(a){
            act = Tensor<T>(shape: TensorShape(lA.count), scalars: lA)
        }
        return act;
    }
    
}

class UnityToGymWrapper<Env: BaseEnv> {
    typealias Scalar = Env.BehaviorSpecImpl.Scalar
    
    var env: Env
    var uint8Visual: Bool = false
    var flattenBranched: Bool = false
    var allowMultipleObs: Bool = false
    var gameOver: Bool = false
    var previousDecisionStep: DecisionSteps? = Optional.none
    var flattener: ActionFlattener<Scalar>? = Optional.none
    var name: BehaviorName? = Optional.none
    var groupSpec: Env.BehaviorSpecImpl? = Optional.none
    var actionSpace: Space? = Optional.none
    var observationSpace: Space? = Optional.none
    var visualObs: Tensor<Float32> = Tensor()

    static var logger: Logger {
        get { return Defaults.logger}
    }

    init(unityEnv: Env, uint8Visual: Bool, flattenBranched: Bool, allowMultipleObs: Bool) throws {
        self.env = unityEnv
        if !env.behaviorSpecs.isEmpty {
            try env.step()
        }
        self.previousDecisionStep = Optional.none
        self.flattener = Optional.none
        self.gameOver = false
        self.allowMultipleObs = allowMultipleObs

        if env.behaviorSpecs.count != 1 {
            throw UnityException.UnityGymException(reason: """
            There can only be one behavior in a UnitEnvironment
            if it is wrapped in a gym
            """)
        }
        self.name = self.env.behaviorSpecs.keys.first
        self.groupSpec = self.name.flatMap{self.env.behaviorSpecs[$0]}

        if self.getNVisObs() == 0 && self.getVecObsSize() == 0 {
            throw UnityException.UnityGymException(reason: """
                There are no observations provided by the environment
            """)
        }
        
        if let nVisObs = self.getNVisObs(), !(nVisObs >= 1 && uint8Visual) {
            Self.logger.warning("""
                uint8_visual was set to true, but visual observations are not in use.
                This setting will not have any effect.
                """
            )
        }
        else {
            self.uint8Visual = uint8Visual
        }
        if let nVisObs = self.getNVisObs(), let vecObsSize = self.getVecObsSize(),
           ((nVisObs + vecObsSize) >= 2 && !self.allowMultipleObs) {
            Self.logger.warning("""
                The environment contains multiple observations.
                You must define allowMultipleObs=True to receive them all.
                Otherwise, only the first visual observation (or vector observation if
                there are no visual observations) will be provided in the observation.
                """
            )
        }
        // Check for number of agents in scene.
        try self.env.reset()
        guard let (decisionSteps, _) = try self.name.flatMap({try self.env.getSteps(behaviorName: $0)}) else {
            throw UnityException.UnityGymException(reason: """
                There are no steps provided by the environment
            """)
        }

        try Self.checkAgents(decisionSteps.len())
        self.previousDecisionStep = decisionSteps

        guard let groupSpec = groupSpec else {
            throw UnityException.UnityGymException(reason: "groupSpec is empty")
        }
        // Set action spaces
        if let groupSpec = groupSpec as? BehaviorSpecDiscreteAction,
           let branches = groupSpec.discreteActionBranches {
            if self.groupSpec?.actionSize == 1 {
               self.actionSpace = Discrete(branches[0])
            } else {
                if flattenBranched, let a = ActionFlattener<Int32>(branches) as? ActionFlattener<Scalar>{
                    self.flattener = a
                    self.actionSpace = a.actionSpace
                } else {
                    self.actionSpace = MultiDiscrete(branches)
                }
            }

        } else {
            if flattenBranched {
                Self.logger.warning("""
                    The environment has a non-discrete action space. It will
                    not be flattened.
                """)
            }
            let high = Tensor<Float32>(repeating: 1, shape: TensorShape(groupSpec.actionShape.map{Int($0)}))
            self.actionSpace = Box(min: -high, max: high)
        }
        // Set observations space
        var listSpaces: [Space] = []
        
        if let shapes = self.getVisObsShape(){
            for s in shapes{
                if uint8Visual{
                    listSpaces.append(Box<UInt8>(min: 0, max: 255, shape: s))
                } else {
                    listSpaces.append(Box<Float32>(min: 0, max: 1, shape: s))
                }
            }
        }
        if let vecObsSize = self.getVecObsSize(), vecObsSize > 0 {
            let high = Tensor<Float32>(repeating: Float32.infinity, shape: [Int(vecObsSize)])
            listSpaces.append(Box(min: -high, max: high))
        }
        if self.allowMultipleObs {
            self.observationSpace = Tuple(listSpaces)
        } else {
            self.observationSpace = listSpaces[0]
        }
    }
    /**
     Resets the state of the environment and returns an initial observation.
     - Returns: observation (object/list): the initial observation of the
     space.
     */
    func reset() throws-> StepResult<Float32> {
        try self.env.reset()
        guard let (decisionSteps, _) = try self.name.flatMap({try self.env.getSteps(behaviorName: $0)}) else {
            throw UnityException.UnityGymException(reason: """
                There are no steps provided by the environment
            """)
        }
        let nAgents = decisionSteps.len()
        try Self.checkAgents(nAgents)
        self.gameOver = false

        return self.singleStep(decisionSteps)
    }

    /**
     Run one timestep of the environment's dynamics. When end of
     episode is reached, you are responsible for calling `reset()`
     to reset this environment's state.
     Accepts an action and returns a tuple (observation, reward, done, info).
     - Parameters:
         - action (object/list): an action provided by the environment
     - Returns:
        - observation (object/list): agent's observation of the current environment
        - reward (float/list) : amount of reward returned after previous action
        - done (boolean/list): whether the episode has ended.
        - info (dict): contains auxiliary diagnostic information.
     """
     */
    func step(action: Tensor<Scalar>) throws -> StepResult<Float32>{
        var act = action
        if let flattener = self.flattener {
            act = flattener.lookupAction(action)
        }

        if let spec = self.groupSpec {
            act = action.reshaped(to: TensorShape(1, spec.actionSize))
        }

        if let n = self.name {
            try self.env.setActions(behaviorName: n, action: act)
        }

        try self.env.step()
        guard let (decisionStep, terminalStep) = try self.name.flatMap({try self.env.getSteps(behaviorName: $0)}) else {
            throw UnityException.UnityGymException(reason: """
                There are no steps provided by the environment
            """)
        }
        try Self.checkAgents(max(decisionStep.len(), terminalStep.len()))
        if terminalStep.len() != 0{
            self.gameOver = true
            return self.singleStep(terminalStep)
        } else {
            return self.singleStep(decisionStep)
        }
    }

    func singleStep<StepsImpl: Steps>(_ info: StepsImpl) -> StepResult<Float32> {
        var defaultObservationSingle: Tensor<Float32>? = nil
        var defaultObservationMulti: [Tensor<Float32>] = []
    
        if self.allowMultipleObs {
            let visualObs = self.getVisObsList(info)
            var visualObsList: [Tensor<Float32>] = []
            for obs in visualObs {
                visualObsList.append(self.preprocessSingle(obs[0]))
            }
            defaultObservationMulti = visualObsList
            if let vecObsSize = self.getVecObsSize(), vecObsSize >= 1 {
                defaultObservationMulti.append(self.getVectorObs(info).gathering(atIndices: Tensor<Int32>(Int32(0)), alongAxis: 0))
            }
        } else {
            if let nVisObs = self.getNVisObs(), nVisObs >= 1 {
                let visualObs = self.getVisObsList(info)
                defaultObservationSingle = self.preprocessSingle(visualObs[0][0])
            } else {
                defaultObservationSingle = self.getVectorObs(info).gathering(atIndices: Tensor<Int32>(Int32(0)), alongAxis: 0)
            }
        }
        if let nVisObs = self.getNVisObs(), nVisObs >= 1 {
            let visualObs = self.getVisObsList(info)
            self.visualObs = self.preprocessSingle(visualObs[0][0])
        }
        let done = info is TerminalSteps ? true : false
        if let obs = defaultObservationSingle {
            return .SingleStepResult(observation: obs, reward: info.reward.scalars[0], done: done, info: [:])
        } else {
            return .MultiStepResult(observation: defaultObservationMulti, reward: info.reward.scalars[0], done: done, info: [:])
        }
        
    }

    func preprocessSingle(_ singleVisualObs: Tensor<Float32>) -> Tensor<Float32> {
        if self.uint8Visual {
            return (255.0 * singleVisualObs)
        } else {
            return singleVisualObs
        }
    }
    
    func getVectorObs(_ stepResult: Steps) -> Tensor<Float32>{
        var result: [Tensor<Float32>] = []
        for obs in stepResult.obs{
            if obs.shape.count == 2{
                result.append(obs)
            }
        }
        return Tensor(concatenating: result, alongAxis: 1)
    }
    
    func getVisObsList(_ stepResult: Steps) -> [Tensor<Float32>]{
        var result: [Tensor<Float32>] = []
        for obs in stepResult.obs {
            if obs.shape.count == 4{
                result.append(obs)
            }
        }
        return result
    }
    
    func getNVisObs() -> Int32? {
        guard let observationShapes = self.groupSpec?.observationShapes else {
            return Optional.none
        }
        var result: Int32 = 0
        for shape in observationShapes {
            if shape.count == 3 {
                result += 1
            }
        }
        return result
    }

    func getVecObsSize() -> Int32? {
        guard let observationShapes = self.groupSpec?.observationShapes else {
            return Optional.none
        }
        var result: Int32 = 0
        for shape in observationShapes {
            if shape.count == 1 {
                result += shape[0]
            }
        }
        return result
    }

    func getVisObsShape() -> [[Int32]]? {
        guard let observationShapes = self.groupSpec?.observationShapes else {
            return Optional.none
        }
        var result: [[Int32]] = [[]]
        for shape in observationShapes {
            if shape.count == 3 {
                result.append(shape)
            }
        }
        return result
    }

    static func checkAgents(_ nAgents: Int) throws -> Void {
        if nAgents > 1 {
            throw UnityException.UnityGymException(reason:"There can only be one Agent in the environment but \(nAgents) were detected."
            )
        }
    }
}
