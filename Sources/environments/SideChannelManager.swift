//
//  SideChannelManager.swift
//  environments
//
//  Created by Sercan Karaoglu on 19/09/2020.
//

import Foundation
import NIO
import Logging

class SideChannelManager {
    let logger = Logger(label: "environments.SideChannelManager")
    var sideChannelsDict: [UUID: SideChannel]
    
    init(sideChannels: [SideChannel]?) throws {
        self.sideChannelsDict = try SideChannelManager.getSideChannelsDict(sideChannels: sideChannels)
    }

    func processSideChannelMessage(message: Data) throws -> Void {
        var data = ByteBuffer(data: message)
        while data.readableBytes > 0 {
            guard let channelId: UUID = data.readString(length: 36).flatMap({UUID(uuidString: $0)}) else {
                throw UnityException.UnityEnvironmentException(reason: "There was a problem reading a message channelId in a SideChannel.")
            }
            
            guard let messageData: ByteBuffer = data.readInteger(as: Int.self).flatMap({data.readSlice(length: $0)}) else {
                throw UnityException.UnityEnvironmentException(reason: """
                    The message received by the side channel \(channelId) was
                    unexpectedly short. Make sure your Unity Environment
                    sending side channel data properly.
                    """
                )
            }
            
            if self.sideChannelsDict.keys.contains(channelId){
                let incomingMessage = IncomingMessage(buffer: messageData)
                try self.sideChannelsDict[channelId]!.onMessageReceived(msg: incomingMessage)
            } else {
                logger.warning("Unknown side channel data received. Channel type: \(channelId).")
            }
        }
    }
    
    func generateSideChannelMessages() -> Data {
        var result = ByteBuffer()
        for (channelId, channel) in self.sideChannelsDict {
            for message in channel.messageQueue {
                var m = message
                result.writeString(channelId.uuidString)
                result.writeInteger(message.readableBytes)
                result.writeBuffer(&m)
            }
            channel.messageQueue = []
        }
        let ret = result.readData(length: result.readableBytes)!
        return ret
    }
    
    static func getSideChannelsDict(sideChannels: [SideChannel]?) throws -> [UUID: SideChannel] {
        var sideChannelsDict: [UUID: SideChannel] = [:]
        if let _sideChannels = sideChannels {
            for sc in _sideChannels {
                if sideChannelsDict.keys.contains(sc.channelId) {
                    throw UnityException.UnityEnvironmentException(reason:"""
                        There cannot be two side channels with
                        the same channel id \(sc.channelId)
                    """)
                }
                sideChannelsDict[sc.channelId] = sc
            }
        }
        return sideChannelsDict
    }
}
