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

public struct Props<T: BehaviorSpec> {
    
    var isFirstMessage: Bool = true
    var communicator: RpcCommunicator
    var sideChannelManager: SideChannelManager?
    var loaded: Bool = false
    var noGraphics: Bool = false
    var envState: [String: (DecisionSteps, TerminalSteps)] = [:]
    var envSpecs: [String: T] = [:]
    var envActions: [String: Tensor<T.Scalar>] = [:]
    var port: Int = 5004
}

open class UnityContinousEnvironment: BaseEnv {
    public typealias BehaviorSpecImpl = BehaviorSpecContinousAction
    public var props: Props<BehaviorSpecContinousAction>
    
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
        self.props = Props<BehaviorSpecContinousAction>(communicator: RpcCommunicator(workerId: 0, port: 5004))
        
    }
    
}

open class UnityDiscreteEnvironment: BaseEnv {
    public typealias BehaviorSpecImpl = BehaviorSpecDiscreteAction
    public var props: Props<BehaviorSpecDiscreteAction>
    
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
        self.props = Props<BehaviorSpecDiscreteAction>(communicator: RpcCommunicator(workerId: 0, port: 5004))
    }
}
