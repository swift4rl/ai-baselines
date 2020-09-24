import Foundation

class EnvironemntParametersChannel: SideChannel {
    static let FLOAT: Int32 = 0
    static let SAMPLER: Int32 = 1

    static let UNIFORM: Int32 = 0
    static let GAUSSIAN: Int32 = 1
    static let MULTIRANGEUNIFORM: Int32 = 2

    init() {
        super.init(channelId: UUID(uuidString: "534c891e-810f-11ea-a9d0-822485860400")!)
    }

    override func onMessageReceived(msg: IncomingMessage) throws -> Void {
        throw UnityException.UnityCommunicationException(
            reason: "The EnvironemntParametersChannel received a message from Unity, this should not have happend."
            )
    }

    func setFloatParameter(key: String, value: Float32) throws -> Void {
        let msg = OutgoingMessage()
        try msg.writeString(key)
        msg.writeInt32(EnvironemntParametersChannel.FLOAT)
        msg.writeFloat32(value)
        queueMessageToSend(msg: msg)
    }

    func setUniformSamplerParameters(key: String, minValue: Float32, maxValue: Float32, seed: Int32) throws -> Void {
        let msg = OutgoingMessage()
        try msg.writeString(key)
        msg.writeInt32(EnvironemntParametersChannel.SAMPLER)
        msg.writeInt32(seed)
        msg.writeInt32(EnvironemntParametersChannel.UNIFORM)
        msg.writeFloat32(minValue)
        msg.writeFloat32(maxValue)
        queueMessageToSend(msg: msg)
    }

    func setGaussianSamplerParameters(key: String, mean: Float32, stDev: Float32, seed: Int32) throws -> Void {
        let msg = OutgoingMessage()
        try msg.writeString(key)
        msg.writeInt32(EnvironemntParametersChannel.SAMPLER)
        msg.writeInt32(seed)
        msg.writeInt32(EnvironemntParametersChannel.GAUSSIAN)
        msg.writeFloat32(mean)
        msg.writeFloat32(stDev)
        queueMessageToSend(msg: msg)
    }

    func setMultirangeSamplerParameters(key: String, intervals: [(Float32, Float32)], seed: Int32) throws -> Void {
        let msg = OutgoingMessage()
        try msg.writeString(key)
        msg.writeInt32(EnvironemntParametersChannel.SAMPLER)
        msg.writeInt32(seed)
        msg.writeInt32(EnvironemntParametersChannel.MULTIRANGEUNIFORM)
        for interval in intervals {
            msg.writeFloat32(interval.0)
            msg.writeFloat32(interval.1)
        }
        queueMessageToSend(msg: msg)
    }
}