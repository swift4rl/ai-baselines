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

public class Props {
    
    var isFirstMessage: Bool = true
    var communicator: RpcCommunicator? = Optional.none
    var sideChannelManager: SideChannelManager? = Optional.none
    var loaded: Bool = false
    var noGraphics: Bool = false
    var envState: [String: (DecisionSteps, TerminalSteps)] = [:]
    var envSpecs: [String: BehaviorSpecContinousAction] = [:]
    var envActions: [String: Tensor<BehaviorSpecContinousAction.Scalar>] = [:]
    var port: Int = 5004
}

open class UnityContinousEnvironment: BaseEnv {
    //public typealias BehaviorSpecImpl = BehaviorSpecContinousAction
}
//
//open class UnityDiscreteEnvironment: BaseEnv {
//    public typealias BehaviorSpecImpl = BehaviorSpecDiscreteAction
//    public var props: Props<BehaviorSpecDiscreteAction>
//
//    required public init() {
//        self.props = Props<BehaviorSpecDiscreteAction>()
//    }
//}
