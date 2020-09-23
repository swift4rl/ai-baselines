import Foundation

class StatsSideChannel: SideChannel {
    typealias StatsAggregationMethod = Int32

    static let AVERAGE: StatsAggregationMethod = 0
    static let MOST_RECENT: StatsAggregationMethod = 1

    typealias StatList = [(Float32, StatsAggregationMethod)]
    typealias EnvironmentStats = [String: StatList]

    var stats = EnvironmentStats()

    init() {
        super.init(channelId: UUID(uuidString: "a1d8f7b7-cec8-50f9-b78b-d3e165a78520")!)
    }

    override func onMessageReceived(msg: IncomingMessage) throws -> Void {
        let key = msg.readString()
        let val = msg.readFloat32()
        let aggType = msg.readInt32()

        stats[key, default: []].append((val, aggType))
    }

    func getAndResetStats() -> EnvironmentStats {
        let s = stats
        stats = EnvironmentStats()
        return s
    }
}