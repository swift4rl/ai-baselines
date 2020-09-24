import Foundation
import NIO

class RawBytesChannel: SideChannel {
    var receivedMessages = [[UInt8]]()

    override func onMessageReceived(msg: IncomingMessage) throws -> Void {
        receivedMessages.append(msg.getRawBytes())
    }

    func getAndClearReceivedMessages() -> [[UInt8]] {
        let ret = receivedMessages
        receivedMessages = []
        return ret
    }

    func setRawData(data: ByteBuffer) throws -> Void {
        let msg = OutgoingMessage()
        msg.setRawBytes(data)
        queueMessageToSend(msg: msg)
    }
    
}