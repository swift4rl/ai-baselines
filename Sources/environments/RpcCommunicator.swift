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
    
    init(firstMsg: EventLoopPromise<Bool>) {
        self.firstMsg = firstMsg
    }
    
    fileprivate lazy var q = [(CommunicatorObjects_UnityMessageProto, EventLoopPromise<CommunicatorObjects_UnityMessageProto>)]()
    var firstMsg: EventLoopPromise<Bool>? = Optional.none
    
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
        firstMsg?.succeed(true)
        firstMsg = Optional.none
        return p.futureResult
    }
    
    func initialize(request: CommunicatorObjects_UnityMessageProto, context: StatusOnlyCallContext) ->
    EventLoopFuture<CommunicatorObjects_UnityMessageProto> {
        firstMsg?.succeed(true)
        firstMsg = Optional.none
        return handle(context, request)
    }
    
    func exchange(request: CommunicatorObjects_UnityMessageProto, context: StatusOnlyCallContext) -> EventLoopFuture<CommunicatorObjects_UnityMessageProto> {
        let p = context.eventLoop.makePromise(of: CommunicatorObjects_UnityMessageProto.self)
        q.append((request,p))
        firstMsg?.succeed(true)
        firstMsg = Optional.none
        return p.futureResult
    }
}

public class RpcCommunicator: Communicator {
   
    var workerId: Int = 0
    var host: String = "0.0.0.0"
    var port: Int = 5004
    var timeoutWait: Int = 30
    var isOpen: Bool = false
    // Create a provider using the features we read.
    let provider: UnityToExternalServicerImplementation
    var server: Server?
    var group: MultiThreadedEventLoopGroup
    
    public required init(workerId: Int=0, port: Int=5005, group: MultiThreadedEventLoopGroup =  MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)) {
        self.workerId = workerId
        self.port = port + workerId
        self.provider = UnityToExternalServicerImplementation(firstMsg: group.next().makePromise(of:Bool.self))
        // Start the server and print its address once it has started.
        let s: EventLoopFuture<Server> = Server.insecure(group: group)
            .withServiceProviders([provider])
            .bind(host: host, port: port)
        self.group = group
        
        s.map { s -> SocketAddress? in
            self.server = s
            return s.channel.localAddress
        }.whenSuccess { address in
            self.isOpen = true
            print("server started on port \(address!.port!)")
        }
        
        _ = s.flatMap {
            $0.onClose
        }.map{
            self.isOpen = false
        }
       
    }
    
    func initialize(inputs: CommunicatorObjects_UnityInputProto) -> Optional<CommunicatorObjects_UnityOutputProto> {
        print("rpc --->> initalize --->>")
        var message = CommunicatorObjects_UnityMessageProto()
        message.header.status=200
        message.unityInput = inputs
        do {
            let res = try self.provider.firstMsg?.futureResult.wait()
            print("init \(res)")
        }catch {
            print("error")
        }
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
            try! self.group.syncShutdownGracefully()
        }
    }
}
