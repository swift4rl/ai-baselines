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
//TODO
    return nil;
}

func stepsFromProto<BehaviorSpecImpl: BehaviorSpec>(
    agentInfoList: [CommunicatorObjects_AgentInfoProto],
    behaviorSpec: BehaviorSpecImpl) -> (DecisionSteps, TerminalSteps){
//TODO
    return nil;
}
