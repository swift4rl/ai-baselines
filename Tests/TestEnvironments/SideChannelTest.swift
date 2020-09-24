import XCTest
import NIO
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

    func testRawBytes() throws {
        let guid = UUID()
        let sender = RawBytesChannel(channelId: guid)
        let receiver = RawBytesChannel(channelId: guid)
        try sender.setRawData(data: ByteBuffer(data: "foo".data(using: .utf8)!))
        try sender.setRawData(data: ByteBuffer(data: "bar".data(using: .utf8)!))
        let data = try SideChannelManager(sideChannels: [sender]).generateSideChannelMessages()
        try SideChannelManager(sideChannels: [receiver]).processSideChannelMessage(message: data)
        let messages = receiver.getAndClearReceivedMessages()
        XCTAssertEqual(2, messages.count)
        XCTAssertEqual("foo", String(bytes: messages[0], encoding: .utf8))
        XCTAssertEqual("bar", String(bytes: messages[1], encoding: .utf8))
    }

    func testMessageBool() throws {
        let vals = [true, false]
        let msgOut = OutgoingMessage()
        for v in vals {
            msgOut.writeBool(v)
        }
        let msgIn = IncomingMessage(buffer: msgOut.buffer)
        var readVals = [Bool]()
        for _ in vals {
            readVals.append(msgIn.readBool())
        }
        XCTAssertEqual(vals, readVals)

        //Test reading with defaults
        XCTAssertFalse(msgIn.readBool())
        XCTAssertTrue(msgIn.readBool(defaultValue: true))
    }

    func testMessageInt32() throws {
        let val: Int32 = 1337
        let msgOut = OutgoingMessage()
        msgOut.writeInt32(val)
        let msgIn = IncomingMessage(buffer: msgOut.buffer)
        let readVal = msgIn.readInt32()
        XCTAssertEqual(val, readVal)

        //Test reading with defaults
        XCTAssertEqual(0, msgIn.readInt32())
        XCTAssertEqual(val, msgIn.readInt32(defaultValue: val))
    }

    func testMessageFloat32() throws {
        let val: Float32 = 42.0
        let msgOut = OutgoingMessage()
        msgOut.writeFloat32(val)
        let msgIn = IncomingMessage(buffer: msgOut.buffer)
        let readVal = msgIn.readFloat32()
        XCTAssertEqual(val, readVal)

        //Test reading with defaults
        XCTAssertEqual(0.0, msgIn.readFloat32())
        XCTAssertEqual(val, msgIn.readFloat32(defaultValue: val))
    }

    func testMessageString() throws {
        let val = "mlagents!"
        let msgOut = OutgoingMessage()
        try msgOut.writeString(val)
        let msgIn = IncomingMessage(buffer: msgOut.buffer)
        let readVal = msgIn.readString()
        XCTAssertEqual(val, readVal)

        //Test reading with defaults
        XCTAssertEqual("", msgIn.readString())
        XCTAssertEqual(val, msgIn.readString(defaultValue: val))
    }

    func testMessageFloat32List() throws {
        let val: [Float32] = [1.0, 3.0, 9.0]
        let msgOut = OutgoingMessage()
        msgOut.writeFloat32List(val)
        let msgIn = IncomingMessage(buffer: msgOut.buffer)
        let readVal = msgIn.readFloat32List()
        XCTAssertEqual(val, readVal)

        //Test reading with defaults
        XCTAssertEqual([], msgIn.readFloat32List())
        XCTAssertEqual(val, msgIn.readFloat32List(defaultValue: val))
    }

    func testEngineConfiguration() throws {
        let sender = EngineConfigurationChannel()
        let receiver = RawBytesChannel(channelId: sender.channelId)

        let config = EngineConfig.defaultConfig
        try sender.setConfiguration(config)
        var data = try SideChannelManager(sideChannels: [sender]).generateSideChannelMessages()
        try SideChannelManager(sideChannels: [receiver]).processSideChannelMessage(message: data)
        let receivedData = receiver.getAndClearReceivedMessages()
        XCTAssertEqual(5, receivedData.count)

        let sentTimeScale: Float32 = 4.5
        try sender.setConfigurationParameters(timeScale: sentTimeScale)
        data = try SideChannelManager(sideChannels: [sender]).generateSideChannelMessages()
        try SideChannelManager(sideChannels: [receiver]).processSideChannelMessage(message: data)
        let message = IncomingMessage(buffer: ByteBuffer(bytes: receiver.getAndClearReceivedMessages()[0]))
        let _ = message.readInt32()
        let timeScale = message.readFloat32()
        XCTAssertEqual(sentTimeScale, timeScale)

        XCTAssertThrowsError(try sender.setConfigurationParameters(width: nil, height: 42))

        try sender.setConfigurationParameters(timeScale: sentTimeScale)
        data = try SideChannelManager(sideChannels: [sender]).generateSideChannelMessages()
        XCTAssertThrowsError(try SideChannelManager(sideChannels: [sender]).processSideChannelMessage(message: data))
    }

    func testEnvironmentParameters() throws {
        let sender = EnvironemntParametersChannel()
        let receiver = RawBytesChannel(channelId: sender.channelId)

        try sender.setFloatParameter(key: "param-1", value: 0.1)
        var data = try SideChannelManager(sideChannels: [sender]).generateSideChannelMessages()
        try SideChannelManager(sideChannels: [receiver]).processSideChannelMessage(message: data)
        let message = IncomingMessage(buffer: ByteBuffer(bytes: receiver.getAndClearReceivedMessages()[0]))
        let key = message.readString()
        let dtype = message.readInt32()
        let value = message.readFloat32()
        XCTAssertEqual(key, "param-1")
        XCTAssertEqual(dtype, EnvironemntParametersChannel.FLOAT)
        XCTAssertEqual(value, 0.1, accuracy: 1e-8)

        try sender.setFloatParameter(key: "param-1", value: 0.1)
        try sender.setFloatParameter(key: "param-2", value: 0.1)
        try sender.setFloatParameter(key: "param-3", value: 0.1)
        data = try SideChannelManager(sideChannels: [sender]).generateSideChannelMessages()
        try SideChannelManager(sideChannels: [receiver]).processSideChannelMessage(message: data)
        XCTAssertEqual(receiver.getAndClearReceivedMessages().count, 3)

        try sender.setFloatParameter(key: "param-1", value: 0.1)
        data = try SideChannelManager(sideChannels: [sender]).generateSideChannelMessages()
        XCTAssertThrowsError(try SideChannelManager(sideChannels: [sender]).processSideChannelMessage(message: data))
    }

    func testStatsChannel() throws {
        let receiver = StatsSideChannel()
        let message = OutgoingMessage()
        try message.writeString("stats-1")
        message.writeFloat32(42.0)
        message.writeInt32(1)

        try receiver.onMessageReceived(msg: IncomingMessage(buffer: message.buffer))

        let stats = receiver.getAndResetStats()

        XCTAssertEqual(stats.count, 1)
        let (val, method) = stats["stats-1"]![0]
        XCTAssertEqual(val, 42.0, accuracy: 1e-8)
        XCTAssertEqual(method, StatsSideChannel.MOST_RECENT)
    }
}