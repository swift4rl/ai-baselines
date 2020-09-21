//
//  RpcUtils.swift
//  environments
//
//  Created by Sercan Karaoglu on 20/09/2020.
//

import Foundation

/**
Converts brain parameter and agent info proto to BehaviorSpec object.
 - Parameters:
  - brainParamProto: protobuf object.
  - agentInfo: protobuf object.
 - Returns: BehaviorSpec object.
 */
func behaviorSpecFromProto<T: BehaviorSpec>(
    brainParamProto: CommunicatorObjects_BrainParametersProto,
    agentInfo: CommunicatorObjects_AgentInfoProto
) -> T {
    let observationShape = agentInfo.observations.map({[$0.shape]})
    let actionType = brainParamProto.vectorActionSpaceType == 0 ? ActionType.DISCRETE : ActionType.CONTINUOUS

    if actionType == ActionType.CONTINUOUS {
        actionShape = brainParamProto.vectorActionSize[0]
    } else {
        actionShape = [brainParamProto.vectorActionSize]
    }
    return T(observationShape, actionType, actionShape)
}

func stepsFromProto<BehaviorSpecImpl: BehaviorSpec>
(agentInfoList: [CommunicatorObjects_AgentInfoProto], behaviorSpec: BehaviorSpecImpl)
-> (DecisionSteps, TerminalSteps) {
    let decisionAgentInfoList: [CommunicatorObjects_AgentInfoProto] = agentInfoList.filter({!agentInfo.done})
    let terminalAgentInfoList: [CommunicatorObjects_AgentInfoProto] = agentInfoList.filter({agentInfo.done})
    
    let decisionObsList: [Tensor<Float32>] = []
    let terminalObsList: [Tensor<Float32>] = []
    for (obs_index, obsShape) in behaviorSpec.observationShapes.enumerated() {
        let isVisual = obsShape.count == 3
        if isVisual{
            obsShape = cast(Tuple[int, int, int], obs_shape)
            decisionObsList.append(
                processVisualObservation(
                    obsIndex, obsShape, decisionAgentInfoList
                )
            )
            terminalObsList.append(
                processVisualObservation(
                    obsIndex, obsShape, terminalAgentInfoList
                )
            )
        } else {
            decisionObsList.append(
                processVectorObservation(
                    obsIndex, obsShape, decisionAgentInfoList
                )
            )
            terminalObsList.append(
                processVectorObservation(
                    obsIndex, obsShape, terminalAgentInfoList
                )
            )
        }
    }
    let decisionReward: Tensor<Float32> = decisionAgentInfoList.map({$0.reward})
    
    let terminalRewards: Tensor<Float32> = terminalAgentInfoList.map({$0.reward})

    raiseOnNanAndInf(decisionRewards, "rewards")
    raiseOnNanAndInf(terminalRewards, "rewards")

    let maxStep = terminalAgentInfoList.map({$0.maxStepReached})
    
    let decisionAgentId = decisionAgentInfoList.map({$0.id})
    let terminalAgentId = terminalAgentInfoList.map({$0.id})
    
    if behaviorSpec.isActionDiscrete {
        let nAgents = decisionAgentInfoList.count
        let aSize = behaviorSpec.discreteActionBranches?.reduce(into: 0, {$0 + $1})
        
        let maskMatrix = Tensor<Int>(zeros: [nAgents, aSize])
        for (agentIndex, agentInfo) in decisionAgentInfoList{
            if agentInfo.actionMask != nil && agentInfo.actionMask.count == aSize {
                mask_matrix[agentIndex, :] = [
                    false if agent_info.action_mask[k] else true
                            for k in range(a_size)
                        ]
            }
        }
        action_mask = (1 - mask_matrix).astype(np.bool)
        indices = _generate_split_indices(behavior_spec.discrete_action_branches)
        action_mask = np.split(action_mask, indices, axis=1)
    }
    return (
        DecisionSteps(
            decisionObsList, decisionRewards, decisionAgentId, actionMask
        ),
        TerminalSteps(terminalObsList, terminalRewards, maxStep, terminalAgentId),
    )
}

/**
 Check for NaNs or Infinite values in the observation or reward data.
 If there's a NaN in the observations, the np.mean() result will be NaN
 If there's an Infinite value (either sign) then the result will be Inf
 Raise a Runtime error in the case that NaNs or Infinite values make it into the data.
 */
func raiseOnNanAndInf(data: Tensor<Float32>, source: String) -> Tensor<Float32> {
    if data.size == 0{
        return data
    }
    
    let d: Tensor<Float32> = data.mean()
    has_nan = np.isnan(d)
    has_inf = not np.isfinite(d)

    if has_nan:
        raise RuntimeError(f"The {source} provided had NaN values.")
    if has_inf:
        raise RuntimeError(f"The {source} provided had Infinite values.")
}
