import XCTest
import TensorFlow
@testable import environments

final class StepTests: XCTestCase {
    var ts: TerminalSteps!
    var continousSpecs: BehaviorSpecContinousAction!
    var discreteSpecs: BehaviorSpecDiscreteAction!
    var ds: DecisionSteps!
    
    override func setUp() {
        super.setUp()
        ts = TerminalSteps(
            obs: [Tensor<Float32>(shape: TensorShape(3,4), scalars: Array(stride(from: 0.0, to: 12.0, by: 1)))],
            reward: Tensor<Float32>(rangeFrom: 0, to: 3, stride: 1),
            interrupted: Tensor<Bool>([true, false, true]),
            agentId: Array(Int32(10)...Int32(13))
        )
        continousSpecs = BehaviorSpecContinousAction(
            observationShapes: [[3, 2], [5]], actionShape: [3]
        )
        discreteSpecs = BehaviorSpecDiscreteAction(
            observationShapes: [[3, 2], [5]], actionShape: [3]
        )
    }

    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
        ts = nil
        ds = nil
        continousSpecs = nil
        discreteSpecs = nil
    }
    
    func testDecisionSteps(){
        let _actionMask: Optional<[Tensor<Bool>]> = Optional.some([Tensor<Bool>(repeating: false, shape: TensorShape(3, 4))])
        let _agentId: [AgentId] = Array(AgentId(10)...AgentId(13))
        let _reward: Tensor<Float32> = Tensor<Float32>(rangeFrom: 0, to: 3, stride: 1)
        let _obs: [Tensor<Float32>] = [Tensor<Float32>(shape: TensorShape(3,4), scalars: Array(stride(from: 0.0, to: 12.0, by: 1)))]
        
        ds = DecisionSteps(
            obs: _obs,
            reward: _reward,
            agentId: _agentId,
            actionMask: _actionMask
        )
        
        XCTAssertTrue(ds.agentIdToIndex[10] == 0)
        XCTAssertTrue(ds.agentIdToIndex[11] == 1)
        XCTAssertTrue(ds.agentIdToIndex[12] == 2)
        
        XCTAssertTrue(ds.agentIdToIndex[-1] == Optional.none)
        let actionMask = ds[10]!.actionMask!
        XCTAssertTrue(actionMask is Array<Tensor<Bool>>)
        XCTAssertTrue(actionMask.count == 1)
        XCTAssertTrue(actionMask[0].scalars == [false, false, false, false])
        
        let s = Set<Int>(stride(from: 0, to: 4, by: 1))
        ds.iter().forEach { agentId in
            let x: Int = ds.agentIdToIndex[agentId]!
            XCTAssertTrue(s.contains(x), "\(x) is not in \(s)")
        }
    }
    
    func testEmptyDecisionSteps(){
        let ds = DecisionSteps.empty(spec: continousSpecs)
        XCTAssertTrue(ds.obs.count == 2)
        XCTAssertTrue(ds.obs[0].shape == TensorShape(0, 3, 2))
        XCTAssertTrue(ds.obs[1].shape == TensorShape(0, 5))
    }
    
    func testTerminalSteps(){
        XCTAssertTrue(ts.agentIdToIndex[10] == 0)
        XCTAssertTrue(ts.agentIdToIndex[11] == 1)
        XCTAssertTrue(ts.agentIdToIndex[12] == 2)
        
        XCTAssertTrue(ts[10]!.interrupted)
        XCTAssertFalse(ts[11]!.interrupted)
        XCTAssertTrue(ts[12]!.interrupted)
        
        XCTAssertTrue(ts.agentIdToIndex[-1] == Optional.none)
        
        let s = Set<Int>(stride(from: 0, to: 4, by: 1))
        ts.iter().forEach{ agentId in
            let x: Int = ts.agentIdToIndex[agentId]!
            XCTAssertTrue(s.contains(x), "\(x) is not in \(s)")
        }
    }
    
    func testEmptyTerminalSteps(){
        let ts = TerminalSteps.empty(spec: continousSpecs)
        XCTAssertTrue(ts.obs.count == 2)
        XCTAssertTrue(ts.obs[0].shape == TensorShape(0, 3, 2))
        XCTAssertTrue(ts.obs[1].shape == TensorShape(0, 5))
    }
    
    func testSpecs(){
        XCTAssertTrue(continousSpecs.discreteActionBranches == Optional.none)
        XCTAssertTrue(continousSpecs.actionSize == 3)
        XCTAssertTrue(continousSpecs.createEmptyAction(nAgents: 5).shape == TensorShape(5, 3))
        XCTAssertTrue(continousSpecs.createEmptyAction(nAgents: 5).scalars is Array<Float32>)
        
        XCTAssertTrue(discreteSpecs.discreteActionBranches == [3])
        XCTAssertTrue(discreteSpecs.actionSize == 1)
        XCTAssertTrue(discreteSpecs.createEmptyAction(nAgents: 5).shape == TensorShape(5, 1))
        XCTAssertTrue(discreteSpecs.createEmptyAction(nAgents: 5).scalars is Array<Int32>)
    }
    
    func testActionGenerator(){
        let actionLen = 30
        let specs = BehaviorSpecContinousAction(observationShapes: [[5]], actionShape: [actionLen])
        let zeroAction = specs.createEmptyAction(nAgents: 4)
        XCTAssertEqual(zeroAction, Tensor<Float32>(repeating: 0.0, shape: TensorShape(4, actionLen)))
        let randomAction = specs.createRandomAction(nAgents: 4)
        XCTAssertTrue(randomAction.scalars is Array<Float32>)
        XCTAssertTrue(randomAction.shape == TensorShape(4, actionLen))
        XCTAssertTrue(randomAction.min().scalar! >= -1)
        XCTAssertTrue(randomAction.max().scalar! <= 1)
        
        let _actionShape = [10, 20, 30]
        let _specs = BehaviorSpecDiscreteAction(observationShapes: [[5]], actionShape: _actionShape)
        let _zeroAction = _specs.createEmptyAction(nAgents: 4)
        XCTAssertEqual(_zeroAction, Tensor<Int32>(repeating: 0, shape: TensorShape(4, _actionShape.count)))
        let _randomAction = _specs.createRandomAction(nAgents: 4)
        XCTAssertTrue(_randomAction.scalars is Array<Int32>)
        XCTAssertEqual(_randomAction.shape, TensorShape(4, _actionShape.count))
        XCTAssertTrue(_randomAction.min().scalar! >= 0)
        for (index, branchSize) in _actionShape.enumerated() {
            let r: Int32 = _randomAction.gathering(atIndices: Tensor<Int32>(Int32(index)), alongAxis: 1).max().scalar!
            XCTAssertLessThan(r, Int32(branchSize))
        }
    }
}
