//
//  UnityGymWrapper.swift
//  environments
//
//  Created by Yeshwanth Kumar on 22/09/2020.
//


import Foundation

class UnityGymException{}

class UnityToGymWrapper {

    //var unityEnv: BaseEnv?
    var uint8Visual: Bool = false
    var flattenBranched: Bool = false
    var allowMultipleObs: Bool = false
    var previousDecisionStep: DecisionSteps
    var terminalStep: TerminalSteps
    
    //Hidden flag used by Atari environments to determine if the game is over
    var gameOver: Bool
    
    init<T: BaseEnv>(unityEnv: T, uint8Visual: Bool, flattenBranched: Bool, allowMultipleObs: Bool) throws {
        self.gameOver = false
        self.allowMultipleObs = allowMultipleObs
        
        let behaviourSpec = unityEnv.behaviorSpecs()
        // Check brain configuration
        if behaviourSpec.count != 1 {
            throw UnityGymWrapperError.unityGymException(reason: """
            There can only be one behavior in a UnitEnvironment
            if it is wrapped in a gym
            """)
        }
        
        let name = Array(behaviourSpec.keys)[0]
        
        //self.groupSpec = self.env.behaviorSpecs[self.name]
        
//        if self.getNVisObs() == 0 && self.getVecObsSize() == 0 {
//            throw UnityGymException("""
//                There are no observations provided by the environment
//            """)
//        }
//
//        if !(self.getNVisObs() >= 1 && uint8Visual) {
//            logger.warning(
//                "uint8_visual was set to true, but visual observations are not in use. "
//                "This setting will not have any effect."
//            )
//        }
//        else {
//            self.uint8_visual = uint8Visual
//        }
//        if (
//            self.getNVisObs() + self.getVecObsSize() >= 2
//            && !self.allowMultipleObs
//        ) {
//            logger.warning(
//                "The environment contains multiple observations. "
//                "You must define allow_multiple_obs=True to receive them all. "
//                "Otherwise, only the first visual observation (or vector observation if"
//                "there are no visual observations) will be provided in the observation."
//            )
//        }
        
        // Check for number of agents in scene.
        unityEnv.reset()
        let steps = unityEnv.getSteps(behaviorName: name)
            

        do { try self._check_agents(nAgents:steps.0.agentIdToIndex.count)} //TODO: yesh: ugly unwrapping. Fix it
        self.previousDecisionStep = steps.0

        // Set action spaces
        if self.groupSpec.isActionDiscrete() {
            branches = self.groupSpec.discreteActionBranches
            if self.groupSpec.actionShape == 1:
                self._action_space = spaces.Discrete(branches[0])
            else:
                if flatten_branched:
                    self._flattener = ActionFlattener(branches)
                    self._action_space = self._flattener.action_space
                else:
                    self._action_space = spaces.MultiDiscrete(branches)

        } else {
            if flattenBranched {
                logger.warning(
                    "The environment has a non-discrete action space. It will "
                    "not be flattened."
                )
            }
            high = np.array([1] * self.group_spec.action_shape)
            self.actionSpace = spaces.Box(-high, high, dtype=np.float32)
        }
        // Set observations space
        list_spaces: List[gym.Space] = []
        shapes = self.getVisObsShape()
        for shape in shapes{
            if uint8Visual:
                list_spaces.append(spaces.Box(0, 255, dtype=np.uint8, shape=shape))
            else:
                list_spaces.append(spaces.Box(0, 1, dtype=np.float32, shape=shape))
        }
        if self.getVecObsSize() > 0 {
            // vector observation is last
            high = np.array([np.inf] * self._get_vec_obs_size())
            list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
        }
        if self.allowMultipleObs {
            self.observationSpace = spaces.Tuple(list_spaces)
        } else {
            self.observationSpace = listSpaces[0]
        }
    }
    
    
    func _check_agents(nAgents: Int) throws {
        if nAgents > 1 {
        throw UnityGymWrapperError.unityGymException(reason: """
        There can only be one agent in a UnitEnvironment
        """)
        }
    }
}




