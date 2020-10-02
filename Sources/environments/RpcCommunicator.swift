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
class UnityToExternalServicerImplementation : CommunicatorObjects_UnityToExternalProtoProvider {
    
    typealias Element = (CommunicatorObjects_UnityMessageProto, EventLoopPromise<CommunicatorObjects_UnityMessageProto>)
    var firstMsg: EventLoopPromise<Bool>?
    
    init(_ firstMsg: EventLoopPromise<Bool>?) {
        self.firstMsg = firstMsg
    }
    
    fileprivate lazy var q = [(CommunicatorObjects_UnityMessageProto, EventLoopPromise<CommunicatorObjects_UnityMessageProto>)]()
    
    func next(delete: Bool = true) -> Element? {
        if q.isEmpty {
            return nil
        } else {
            let first = q.first
            if(delete){
                q.remove(at: 0)
            }
            return first
        }
        
    }
    
    func initialize(request: CommunicatorObjects_UnityMessageProto, context: StatusOnlyCallContext) ->
    EventLoopFuture<CommunicatorObjects_UnityMessageProto> {
        print("rpc --->> initalize --->> \(request.debugDescription))")
        let p = context.eventLoop.makePromise(of: CommunicatorObjects_UnityMessageProto.self)
        q.append((request,p))
        self.firstMsg?.succeed(true)
        self.firstMsg = Optional.none
        return p.futureResult
    }
    
    func exchange(request: CommunicatorObjects_UnityMessageProto, context: StatusOnlyCallContext) -> EventLoopFuture<CommunicatorObjects_UnityMessageProto> {
        print("rpc --->> exchange --->> \(request.debugDescription))")
        let p = context.eventLoop.makePromise(of: CommunicatorObjects_UnityMessageProto.self)
        q.append((request,p))
        self.firstMsg?.succeed(true)
        self.firstMsg = Optional.none
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
    
    public required init(workerId: Int=0, port: Int=5005, group: MultiThreadedEventLoopGroup =  MultiThreadedEventLoopGroup(numberOfThreads: 1)) {
        self.workerId = workerId
        self.port = port + workerId
        self.provider = UnityToExternalServicerImplementation(group.next().makePromise())
        
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
        // TODO handle this properly
        _ = try? s.wait()
        
    }
    
    func initialize(inputs: CommunicatorObjects_UnityInputProto) -> Optional<CommunicatorObjects_UnityOutputProto> {
        
        var message = CommunicatorObjects_UnityMessageProto()
        message.header.status=200
        message.unityInput = inputs
        var m = self.provider.next()
        while m == nil {
            sleep(1)
            m = self.provider.next()
        }
        _ = try? self.provider.firstMsg?.futureResult.wait()
        print("model --->> initalize --->> \(message.debugDescription)")
        m?.1.succeed(message)
        var n = self.provider.next(delete: false)
        while n == nil {
            sleep(1)
            n = self.provider.next(delete: false)
        }
        return m?.0.unityOutput
    }
    
    func exchange(inputs: CommunicatorObjects_UnityInputProto) -> Optional<CommunicatorObjects_UnityOutputProto> {
        var message = CommunicatorObjects_UnityMessageProto()
        print("model --->> exchange -->>")
        message.header.status = 200
        message.unityInput = inputs
        var m = self.provider.next()
        while m == nil {
            sleep(1)
            m = self.provider.next()
        }
        print("model --->> initalize --->> \(message.debugDescription))")
        m?.1.succeed(message)
        
        m = self.provider.next(delete: false)
        while m == nil {
            sleep(1)
            m = self.provider.next(delete: false)
        }
        if m?.0.header.status != 200{
            return Optional.none
        }
        return m?.0.unityOutput
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
