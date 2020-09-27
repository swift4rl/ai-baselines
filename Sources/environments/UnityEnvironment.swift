//
//  UnityEnv.swift
//  App
//
//  Created by Sercan Karaoglu on 19/04/2020.
//

import Foundation
import TensorFlow
import GRPC
import NIO

struct Props<T: BehaviorSpec> {
    
    var isFirstMessage: Bool = true
    var communicator: RpcCommunicator
    var client: CommunicatorObjects_UnityToExternalProtoClient
    var sideChannelManager: SideChannelManager
    var loaded: Bool = false
    var envState: [String: (DecisionSteps, TerminalSteps)] = [:]
    var envSpecs: [String: T] = [:]
    var envActions: [String: Tensor<T.Scalar>] = [:]
}

class UnityContinousEnvironment: BaseEnv {
    typealias BehaviorSpecImpl = BehaviorSpecContinousAction
    var props: Props<BehaviorSpecContinousAction>
    
    init(communicator co: RpcCommunicator, client cl: CommunicatorObjects_UnityToExternalProtoClient, sideChannelManager scm: SideChannelManager) {
        self.props = Props<BehaviorSpecContinousAction>(communicator: co, client: cl, sideChannelManager: scm)
    }
    
}

class UnityDiscreteEnvironment: BaseEnv {
    typealias BehaviorSpecImpl = BehaviorSpecDiscreteAction
    var props: Props<BehaviorSpecDiscreteAction>
    
    init(communicator co: RpcCommunicator, client cl: CommunicatorObjects_UnityToExternalProtoClient, sideChannelManager scm: SideChannelManager) {
        self.props = Props<BehaviorSpecDiscreteAction>(communicator: co, client: cl, sideChannelManager: scm)
    }
}
