//
//  RpcUtils.swift
//  environments
//
//  Created by Sercan Karaoglu on 20/09/2020.
//

import Foundation
import TensorFlow

typealias SpaceTypeProto = CommunicatorObjects_SpaceTypeProto
typealias CompressionTypeProto = CommunicatorObjects_CompressionTypeProto
/*
Converts brain parameter and agent info proto to BehaviorSpec object.
 - Parameters:
  - brainParamProto: protobuf object.
  - agentInfo: protobuf object.
 - Returns: BehaviorSpec object.
 */
func behaviorSpecFromProto<T: BehaviorSpec>(
    brainParamProto: CommunicatorObjects_BrainParametersProto,
    agentInfo: CommunicatorObjects_AgentInfoProto
) -> T{
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
            throw UnityException.UnityObservationException(reason: "Visual representation has not yet been supported!")
//            decisionObsList.append(
//                try processVisualObservation(
//                    obsIndex: obsIndex, shape: obsShape, agentInfoList: decisionAgentInfoList
//                )
//            )
//            terminalObsList.append(
//                try processVisualObservation(
//                    obsIndex: obsIndex, shape: obsShape, agentInfoList: terminalAgentInfoList
//                )
//            )
        } else {
            decisionObsList.append(
                try processVectorObservation(
                    obsIndex: obsIndex, shape: obsShape, agentInfoList: decisionAgentInfoList
                )
            )
            terminalObsList.append(
                try processVectorObservation(
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
    
    var actionMask: Optional<[Tensor<Bool>]> = Optional.none
    if let behaviorSpec = behaviorSpec as? BehaviorSpecDiscreteAction{
        let nAgents = decisionAgentInfoList.count
        if let aSize = behaviorSpec.discreteActionBranches?.reduce(0, +) {
            var maskMatrix = Tensor<Bool>(repeating: true, shape: [nAgents, Int(aSize)])
            
            for (agentIndex, agentInfo) in decisionAgentInfoList.enumerated() {
                if agentInfo.actionMask.count == aSize {
                    maskMatrix[agentIndex] = Tensor<Bool>(agentInfo.actionMask).elementsLogicalNot()
                }
            }
            if let dims = behaviorSpec.discreteActionBranches {
                actionMask = Optional.some(maskMatrix.elementsLogicalNot().split(sizes: dims.map({Int($0)}), alongAxis: 1))
            }
        }
    }
    return (
        DecisionSteps(
            obs: decisionObsList, reward: Tensor(decisionRewards), agentId: decisionAgentId, actionMask: actionMask
        ),
        TerminalSteps(obs: terminalObsList, reward: Tensor(terminalRewards), interrupted: Tensor(maxStep), agentId: terminalAgentId)
    )
}

//func processVisualObservation(
//    obsIndex: Int,
//    shape: [Int32],
//    agentInfoList: [CommunicatorObjects_AgentInfoProto]
//) throws -> Tensor<Float32> {
//    if agentInfoList.count == 0 {
//        return Tensor<Float32>(repeating: 0, shape: TensorShape(shape.map({Int($0)})))
//    }
//    return try Tensor(agentInfoList.map({try observationToNpArray(obs: $0.observations[obsIndex], expectedShape: shape)}))
//}

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

///**
//Converts observation proto into numpy array of the appropriate size.
//  - Parameters:
//    - obs: observation proto to be converted
//    - expectedShape: optional shape information, used for sanity checks.
//
//  - Returns:
//    processed numpy array of observation from environment
// */
//func observationToNpArray(
//    obs:  CommunicatorObjects_ObservationProto, expectedShape: [Int32]? = Optional.none
//) throws -> Tensor<Float32> {
//    let obsShape = TensorShape(obs.shape.map({Int($0)}))
//    if let expectedShape = expectedShape {
//        if obsShape != TensorShape(expectedShape.map({Int($0)})) {
//            throw UnityException.UnityObservationException(
//                reason:"Observation did not have the expected shape - got \(obs.shape) but expected \(expectedShape)"
//            )
//        }
//    }
//    let grayScale = obs.shape[2] == 1
//    if obs.compressionType == CompressionTypeProto.none {
//        var img = Tensor<Float32>(obs.floatData.data)
//        img = img.reshaped(to: TensorShape(obs.shape.map({Int($0)})))
//        return img
//    } else {
//        var img = processPixels(obs.compressedData, grayScale)
//        if obsShape != TensorShape(img.shape.map({Int($0)})) {
//            throw UnityException.UnityObservationException(reason: """
//                Decompressed observation did not have the expected shape
//                decompressed had \(img.shape) but expected \(obs.shape)
//                """
//            )
//        }
//        return img
//    }
//}

///**
//Converts byte array observation image into numpy array, re-sizes it,
//and optionally converts it to grey scale
//- Parameters:
// - gray_scale: Whether to convert the image to grayscale.
// - image_bytes: input byte array corresponding to image
//- Returns:
// - processed numpy array of observation from environment
//*/
//func processPixels(imageBytes: Data, grayScale: Bool) -> Tensor<Float32>{
//    imageBytes.withUnsafeBytes { (floatPtr: UnsafePointer<Float>) in
//        floatPtr[index] / 255.0
//    }
//    CGImage
//    imageBytes.withUnsafeMutableBytes({ mtb in })
//    let ctx = CGContext.from(pixels: imageBytes, width: 128)
//    ctx.
//    s = np.array(image, dtype=np.float32) / 255.0
//    if gray_scale:
//        s = np.mean(s, axis=2)
//        s = np.reshape(s, [s.shape[0], s.shape[1], 1])
//    return s
//}

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
