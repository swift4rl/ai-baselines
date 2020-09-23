import Foundation

struct EngineConfig {
    let width: Int32?
    let height: Int32?
    let qualityLevel: Int32?
    let timeScale: Float32?
    let targetFrameRate: Int32?
    let captureFrameRate: Int32?

    static let defaultConfig = EngineConfig(width: 80, height: 80, qualityLevel: 1, timeScale: 20.0, targetFrameRate: -1, captureFrameRate: 60)
}

class EngineConfigurationChannel: SideChannel {
    static let SCREEN_RESOLUTION: Int32 = 0
    static let QUALITY_LEVEL: Int32 = 1
    static let TIME_SCALE: Int32 = 2
    static let TARGET_FRAME_RATE: Int32 = 3
    static let CAPTURE_FRAME_RATE: Int32 = 4

    init() {
        super.init(channelId: UUID(uuidString: "e951342c-4f7e-11ea-b238-784f4387d1f7")!)
    }

    override func onMessageReceived(msg: IncomingMessage) throws -> Void {
        throw UnityException.UnityCommunicationException(
            reason: "The EngineConfigurationChannel received a message from Unity, this should not have happend."
            )
    }

    func setConfigurationParameters(
        width: Int32? = nil,
        height: Int32? = nil,
        qualityLevel: Int32? = nil,
        timeScale: Float32? = nil,
        targetFrameRate: Int32? = nil,
        captureFrameRate: Int32? = nil
        ) throws {
        if (width == nil && height != nil) || (width != nil && height == nil) {
            throw UnityException.UnitySideChannelException(
                reason: "You cannot set the width/height of the screen resolution without also setting the height/width"
                )
        }

        if width != nil && height != nil {
            let screenMsg = OutgoingMessage()
            screenMsg.writeInt32(EngineConfigurationChannel.SCREEN_RESOLUTION)
            screenMsg.writeInt32(width!)
            screenMsg.writeInt32(height!)
            queueMessageToSend(msg: screenMsg)
        }

        if qualityLevel != nil {
            let qualityLevelMessage = OutgoingMessage()
            qualityLevelMessage.writeInt32(EngineConfigurationChannel.QUALITY_LEVEL)
            qualityLevelMessage.writeInt32(qualityLevel!)
            queueMessageToSend(msg: qualityLevelMessage)
        }

        if timeScale != nil {
            let timeScaleMessage = OutgoingMessage()
            timeScaleMessage.writeInt32(EngineConfigurationChannel.TIME_SCALE)
            timeScaleMessage.writeFloat32(timeScale!)
            queueMessageToSend(msg: timeScaleMessage)
        }

        if targetFrameRate != nil {
            let targetFrameRateMessage = OutgoingMessage()
            targetFrameRateMessage.writeInt32(EngineConfigurationChannel.TARGET_FRAME_RATE)
            targetFrameRateMessage.writeInt32(targetFrameRate!)
            queueMessageToSend(msg: targetFrameRateMessage)
        }

        if captureFrameRate != nil {
            let captureFrameRateMessage = OutgoingMessage()
            captureFrameRateMessage.writeInt32(EngineConfigurationChannel.CAPTURE_FRAME_RATE)
            captureFrameRateMessage.writeInt32(targetFrameRate!)
            queueMessageToSend(msg: captureFrameRateMessage)
        }
    }

    func setConfiguration(_ config: EngineConfig) throws -> Void {
        try setConfigurationParameters(
            width: config.width,
            height: config.height,
            qualityLevel: config.qualityLevel,
            timeScale: config.timeScale,
            targetFrameRate: config.targetFrameRate,
            captureFrameRate: config.captureFrameRate
            )
    }
}