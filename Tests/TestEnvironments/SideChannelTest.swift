import XCTest
@testable import environments

class IntChannel: SideChannel {
    var intList = [Int32]()

    init() {
        super.init(channelId: UUID(uuidString: "a85ba5c0-4f87-11ea-a517-784f4387d1f7")!)
    }

    override func onMessageReceived(msg: IncomingMessage) -> Void {
        let intVal = msg.readInt32()
        intList.append(intVal)
    }

    func sendInt(_ intVal: Int32) -> Void {
        let msg = OutgoingMessage()
        msg.writeInt32(intVal)
        super.queueMessageToSend(msg: msg)
    }
}

class SideChannelTest: XCTestCase {
    func testIntChannel() throws {
        let sender = IntChannel()
        let receiver = IntChannel()
        sender.sendInt(5)
        sender.sendInt(6)
        let data = try SideChannelManager(sideChannels: [sender]).generateSideChannelMessages()
        try SideChannelManager(sideChannels: [receiver]).processSideChannelMessage(message: data)
        XCTAssertEqual(receiver.intList[0], 5)
        XCTAssertEqual(receiver.intList[1], 6)
    }
}