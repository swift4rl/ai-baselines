//
//  Communicator.swift
//  App
//
//  Created by Sercan Karaoglu on 25/04/2020.
//

import Foundation
import NIO

protocol Communicator {

    init(workerId: Int, port: Int, group: MultiThreadedEventLoopGroup)

    func initialize(inputs: CommunicatorObjects_UnityInputProto) -> Optional<CommunicatorObjects_UnityOutputProto>
    
    func exchange(inputs: CommunicatorObjects_UnityInputProto) -> Optional<CommunicatorObjects_UnityOutputProto>

    func close()
    
}
