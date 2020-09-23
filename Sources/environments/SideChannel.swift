//
//  SideChannel.swift
//  environments
//
//  Created by Sercan Karaoglu on 05/09/2020.
//

import Foundation
import NIO

class SideChannel {
    let channelId: UUID
    var messageQueue = [ByteBuffer]()
    
    init(channelId: UUID) {
        self.channelId = channelId
    }
    
    func queueMessageToSend(msg: OutgoingMessage) -> Void {
        self.messageQueue.append(msg.buffer)
    }
    
    func onMessageReceived(msg: IncomingMessage) -> Void {}
}
