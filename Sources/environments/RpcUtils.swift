//
//  RpcUtils.swift
//  environments
//
//  Created by Sercan Karaoglu on 20/09/2020.
//

import Foundation
import TensorFlow

typealias SpaceTypeProto = CommunicatorObjects_SpaceTypeProto
/**
Converts brain parameter and agent info proto to BehaviorSpec object.
 - Parameters:
  - brainParamProto: protobuf object.
  - agentInfo: protobuf object.
 - Returns: BehaviorSpec object.
 */
func behaviorSpecFromProto<T>(
    brainParamProto: CommunicatorObjects_BrainParametersProto,
    agentInfo: CommunicatorObjects_AgentInfoProto
) -> T where T: BehaviorSpec{
    let observationShape = agentInfo.observations.map({$0.shape})
    
    if SpaceTypeProto.discrete == brainParamProto.vectorActionSpaceType {
        return T.create(observationShapes: observationShape, actionShape: brainParamProto.vectorActionSize)
    } else {
        return T.create(observationShapes: observationShape, actionShape: [brainParamProto.vectorActionSize[0]])
    }
    
}

func stepsFromProto<BehaviorSpecImpl: BehaviorSpec>
(agentInfoList: [CommunicatorObjects_AgentInfoProto], behaviorSpec: BehaviorSpecImpl)
throws -> (DecisionSteps, TerminalSteps)  {
    let decisionAgentInfoList: [CommunicatorObjects_AgentInfoProto] = agentInfoList.filter({!$0.done})
    let terminalAgentInfoList: [CommunicatorObjects_AgentInfoProto] = agentInfoList.filter({$0.done})
    
    var decisionObsList: [Tensor<Float32>] = []
    var terminalObsList: [Tensor<Float32>] = []
    for (obsIndex, obsShape) in behaviorSpec.observationShapes.enumerated() {
        let isVisual = obsShape.count == 3
        if isVisual{
            decisionObsList.append(
                processVisualObservation(
                    obsIndex: obsIndex, shape: obsShape, agentInfoList: decisionAgentInfoList
                )
            )
            terminalObsList.append(
                processVisualObservation(
                    obsIndex: obsIndex, shape: obsShape, agentInfoList: terminalAgentInfoList
                )
            )
        } else {
            decisionObsList.append(
                processVectorObservation(
                    obsIndex: obsIndex, shape: obsShape, agentInfoList: decisionAgentInfoList
                )
            )
            terminalObsList.append(
                processVectorObservation(
                    obsIndex: obsIndex, shape: obsShape, agentInfoList: terminalAgentInfoList
                )
            )
        }
    }
    let decisionRewards: [Float32] = decisionAgentInfoList.map({$0.reward})
    
    let terminalRewards: [Float32] = terminalAgentInfoList.map({$0.reward})

    try raiseOnNanAndInf(data: decisionRewards, source: "rewards")
    try raiseOnNanAndInf(data: terminalRewards, source: "rewards")

    let maxStep = terminalAgentInfoList.map({$0.maxStepReached})
    
    let decisionAgentId = decisionAgentInfoList.map({$0.id})
    let terminalAgentId = terminalAgentInfoList.map({$0.id})
    
    if behaviorSpec.isActionDiscrete {
        let nAgents = decisionAgentInfoList.count
        if let aSize = behaviorSpec.discreteActionBranches?.reduce(0, +) {
            var maskMatrix = Tensor<Bool>(repeating: true, shape: [nAgents, Int(aSize)])
            
            for (agentIndex, agentInfo) in decisionAgentInfoList.enumerated() {
                if agentInfo.actionMask.count == aSize {
                    maskMatrix[agentIndex] = Tensor<Bool>(agentInfo.actionMask).elementsLogicalNot()
                }
            }
            var actionMask = maskMatrix.elementsLogicalNot()
            if let dims = behaviorSpec.discreteActionBranches {
                let indices = generateSplitIndices(dims: dims)
                // TODO actionMask = np.split(action_mask, indices, axis=1)
            }
        }
    }
    
    return (
        DecisionSteps(
            decisionObsList, decisionRewards, decisionAgentId, actionMask
        ),
        TerminalSteps(terminalObsList, terminalRewards, maxStep, terminalAgentId),
    )
}
func processVisualObservation(
    obsIndex: Int,
    shape: [Int32]
    agentInfoList: [CommunicatorObjects_AgentInfoProto]
) -> Tensor<Float32> {
    if agentInfoList.count == 0 {
        return Tensor<Float32>(repeating: 0, shape: TensorShape(shape.map({Int($0)})))
    }

    batched_visual = [
        observation_to_np_array(agent_obs.observations[obs_index], shape)
        for agent_obs in agent_info_list
    ]
    return np.array(batched_visual, dtype=np.float32)
}

func processVectorObservation(
    obsIndex: Int,
    shape: [Int32],
    agentInfoList: [ CommunicatorObjects_AgentInfoProto ]
) throws -> Tensor<Float> {
    if agentInfoList.count == 0 {
        return Tensor<Float32>(repeating:0, shape: TensorShape(Int(shape[0])))
    }
    let obs = Tensor<Float>(stacking: agentInfoList.map({Tensor<Float>($0.observations[obsIndex].floatData.data)}) )
    try raiseOnNanAndInf(data: obs.scalars, source: "observations")
    return obs
}

func generateSplitIndices(dims: [Int32]) -> [Int32]{
    if dims.count <= 1 {
        return []
    }
    var result = [dims[0]]
    for i in 0...(dims.count - 2) {
        result += [dims[i + 1] + result[i]]
    }
    return result
}
/**
 Check for NaNs or Infinite values in the observation or reward data.
 If there's a NaN in the observations, the np.mean() result will be NaN
 If there's an Infinite value (either sign) then the result will be Inf
 Raise a Runtime error in the case that NaNs or Infinite values make it into the data.
 */
func raiseOnNanAndInf(data: [Float32], source: String) throws -> Void {
    if data.count == 0{
        return ;
    }
    guard let _ = data.first(where: {$0.isNaN}) else {
        throw UnityException.UnityEnvironmentException(reason: "The \(source) provided had NaN values.")
    }
    guard let _ = data.first(where: {$0.isInfinite}) else {
        throw UnityException.UnityEnvironmentException(reason: "The \(source) provided had Infinite values.")
    }
}
