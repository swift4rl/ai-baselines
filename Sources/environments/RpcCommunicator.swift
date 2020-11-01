//
//  RpcCommunicator.swift
//  App
//
//  Created by Sercan Karaoglu on 25/04/2020.
//

import Foundation
import GRPC
import NIO
import Logging

protocol UnityEnvironmentListener {
    func onRLInitOutput(output: CommunicatorObjects_UnityOutputProto) -> CommunicatorObjects_UnityInputProto
    func generateResetInput() -> CommunicatorObjects_UnityInputProto
    func updateBehaviorSpecs(output: CommunicatorObjects_UnityOutputProto)
    func updateState(output: CommunicatorObjects_UnityRLOutputProto) -> CommunicatorObjects_UnityInputProto
}

class UnityToExternalServicerImplementation : CommunicatorObjects_UnityToExternalProtoProvider {
    
    typealias Element = (CommunicatorObjects_UnityMessageProto, EventLoopPromise<CommunicatorObjects_UnityMessageProto>)
    var unityEnvironmentListener: UnityEnvironmentListener
    
    init(listener: UnityEnvironmentListener) {
        self.unityEnvironmentListener = listener
    }
    
    func exchange(request: CommunicatorObjects_UnityMessageProto, context: StatusOnlyCallContext) -> EventLoopFuture<CommunicatorObjects_UnityMessageProto> {
        //print("Request => \(request)")
        var message = CommunicatorObjects_UnityMessageProto()
        message.header.status=200
        if request.unityOutput.hasRlInitializationOutput && request.unityOutput.rlInitializationOutput.brainParameters.isEmpty {
            message.unityInput = unityEnvironmentListener.onRLInitOutput(output: request.unityOutput)
            //print("Response => \(message)")
            return context.eventLoop.makeSucceededFuture(message)
        }
        if !request.hasUnityOutput {
            message.unityInput = self.unityEnvironmentListener.generateResetInput()
            print("Response => \(message)")
            return context.eventLoop.makeSucceededFuture(message)
        }
        
        if request.unityOutput.hasRlInitializationOutput && !request.unityOutput.rlInitializationOutput.brainParameters.isEmpty {
            unityEnvironmentListener.updateBehaviorSpecs(output: request.unityOutput)
        }
        
        message.unityInput = unityEnvironmentListener.updateState(output: request.unityOutput.rlOutput)
        //print("Response => \(message)")
        return context.eventLoop.makeSucceededFuture(message)
        
    }
}

public class RpcCommunicator {
    
    var workerId: Int = 0
    var host: String = "0.0.0.0"
    var port: Int = 5004
    var timeoutWait: Int = 30
    var isOpen: Bool = false
    // Create a provider using the features we read.
    let provider: UnityToExternalServicerImplementation
    var server: Server?
    var group: MultiThreadedEventLoopGroup
    let sleepInterval = 0.05
    
    required init(workerId: Int=0, port: Int=5005, group: MultiThreadedEventLoopGroup = MultiThreadedEventLoopGroup(numberOfThreads: 1), listener: UnityEnvironmentListener) {
        self.workerId = workerId
        self.port = port + workerId
        self.provider = UnityToExternalServicerImplementation(listener: listener)
        self.group = group
    }
    
    func startServer() {
        // Start the server and print its address once it has started.
        let s: EventLoopFuture<Server> = Server.insecure(group: group)
            .withServiceProviders([provider])
            .bind(host: host, port: port)
        
        s.map { s -> SocketAddress? in
            self.server = s
            return s.channel.localAddress
        }.whenSuccess { address in
            self.isOpen = true
            print("server started on port \(address!.port!)")
        }
        // TODO handle this properly
        _ = try? s.flatMap{
            $0.onClose
        }.wait()
    }
    
//    func close() {
//        if self.isOpen{
//            var message = CommunicatorObjects_UnityMessageProto()
//            message.header.status = 400
//            try! self.group.syncShutdownGracefully()
//        }
//    }
}
