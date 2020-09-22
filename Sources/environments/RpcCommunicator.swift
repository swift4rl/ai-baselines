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

//TODO: consider GCD https://www.raywenderlich.com/5370-grand-central-dispatch-tutorial-for-swift-4-part-1-2
//or https://gist.github.com/lattner/31ed37682ef1576b16bca1432ea9f782
class UnityToExternalServicerImplementation : CommunicatorObjects_UnityToExternalProtoProvider, IteratorProtocol {
    
    typealias Element = (CommunicatorObjects_UnityMessageProto, EventLoopPromise<CommunicatorObjects_UnityMessageProto>)
    
    fileprivate lazy var q = [(CommunicatorObjects_UnityMessageProto, EventLoopPromise<CommunicatorObjects_UnityMessageProto>)]()
    
    func next() -> Element? {
        if q.isEmpty {
            return nil
        } else {
            let first = q.first
            q.remove(at: 0)
            return first
        }
    }
    
    
    fileprivate func handle(_ context: StatusOnlyCallContext, _ request: CommunicatorObjects_UnityMessageProto) -> EventLoopFuture<CommunicatorObjects_UnityMessageProto> {
        let p = context.eventLoop.makePromise(of: CommunicatorObjects_UnityMessageProto.self)
        q.append((request,p))
        return p.futureResult
    }
    
    func initialize(request: CommunicatorObjects_UnityMessageProto, context: StatusOnlyCallContext) ->
    EventLoopFuture<CommunicatorObjects_UnityMessageProto> {
        return handle(context, request)
    }
    
    func exchange(request: CommunicatorObjects_UnityMessageProto, context: StatusOnlyCallContext) -> EventLoopFuture<CommunicatorObjects_UnityMessageProto> {
        print("========?>>>>>")
     var rlIn = CommunicatorObjects_UnityRLInputProto()
         rlIn.command = CommunicatorObjects_CommandProto.quit
         rlIn.sideChannel = request.unityOutput.rlOutput.sideChannel
         var value = CommunicatorObjects_UnityInputProto()
         value.rlInput = rlIn
         var ret = CommunicatorObjects_UnityMessageProto()
         ret.unityInput = value
        print(value)
         return context.eventLoop.makeSucceededFuture(ret)
    /*EventLoopFuture<CommunicatorObjects_UnityMessageProto> {
        print("exchange -->>>")
        print(request.unityInput)
        let p = context.eventLoop.makePromise(of: CommunicatorObjects_UnityMessageProto.self)
        q.append((request,p))
        return p.futureResult
         }
         */
    }
    
    
}

public class RpcCommunicator: Communicator {
   
    var workerId: Int = 0
    var host: String = "0.0.0.0"
    var port: Int = 5004
    var timeoutWait: Int = 30
    var isOpen: Bool = false
    // Create a provider using the features we read.
    let provider = UnityToExternalServicerImplementation()
    var server: Server?
    
    public required init(workerId: Int=0, port: Int=5005, group: MultiThreadedEventLoopGroup =  MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)) {
        defer {
          try! group.syncShutdownGracefully()
        }
        self.workerId = workerId
        self.port = port + workerId
        
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
        // Wait on the server's `onClose` future to stop the program from exiting.
        do {
            _ = try s.flatMap {
                $0.onClose
            }.map{
                self.isOpen = false
            }.wait()
        } catch let e{
            print(e.localizedDescription)
        }

    }
    
    func initialize(inputs: CommunicatorObjects_UnityInputProto) -> Optional<CommunicatorObjects_UnityOutputProto> {
        print("rpc --->> initalize --->>")
        var message = CommunicatorObjects_UnityMessageProto()
        message.header.status=200
        message.unityInput = inputs
        let n = provider.next()
        return n?.0.unityOutput
    }
    
    func exchange(inputs: CommunicatorObjects_UnityInputProto) -> Optional<CommunicatorObjects_UnityOutputProto> {
        var message = CommunicatorObjects_UnityMessageProto()
        print("rpc --->> exchange -->>")
        message.header.status = 200
        message.unityInput = inputs
        let m = self.provider.next()
        let request = m?.0
        let promise = m?.1
        promise?.succeed(message)
        if request?.header.status != 200{
            return Optional.none
        }
        return request?.unityOutput
    }

    func close() {
        if self.isOpen{
            var message = CommunicatorObjects_UnityMessageProto()
            message.header.status = 400
            let m = self.provider.next()
            m?.1.succeed(message)
            server?.close().whenSuccess{ s in
                self.isOpen = false
            }
        }
    }
}
