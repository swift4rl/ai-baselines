//
//  SideChannelManager.swift
//  environments
//
//  Created by Sercan Karaoglu on 19/09/2020.
//

import Foundation
import NIO
import Logging

open class SideChannelManager {
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
    
    struct DebugData {
        let x1: [UInt8]
        let x2: [UInt8]
        let x3: [UInt8]
        let x4: [UInt8]
        let x5: [UInt8]
    }
    static func toLittleEndain(uuid: UUID) -> Data {
        let (u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16) = uuid.uuid
        let data = Data([u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16])
    
        return Data(data[0..<4].reversed()) + Data(data[4..<6].reversed()) + Data(data[6..<8].reversed()) +  Data(data[8...])
    }
    
    func generateSideChannelMessages() -> Data {
        var result = ByteBuffer()
        for (channelId, channel) in self.sideChannelsDict {
            for message in channel.messageQueue {
                var m = message
                result.writeData(Self.toLittleEndain(uuid: channelId))
                result.writeInteger(Int32(message.readableBytes), endianness: .little)
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
