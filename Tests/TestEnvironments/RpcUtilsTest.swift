//
//  RpcCommunicatorTest.swift
//  TestEnvironments
//
//  Created by Sercan Karaoglu on 26/08/2020.
//

import Foundation
import XCTest
import TensorFlow
import GRPC
import NIO
import Logging
@testable import environments

extension Array {
    static func *(lhs: Array,rhs : Array) ->
              [[Array.Iterator.Element]]
    {
        let product = rhs.flatMap({ x in lhs.map{y in [x,y]}})
        return product
    }
}

final class RpcUtilsTest: XCTestCase {
    
    func split<T>(_ t: Tensor<T>, _ i: [Int]) -> [Tensor<T>] {
       return t.split(sizes: [3,3,3], alongAxis: 1)
    }
    
    func testSplit(){
        let givenTensor = Tensor<Bool>(repeating: false, shape: TensorShape(16,9))
        let givenIndices = [3,6]
        let r1 = Tensor<Bool>(repeating: false, shape: TensorShape(16,3))
        let expected = [r1, r1, r1]
        XCTAssertEqual(expected, split(givenTensor,givenIndices))
    }
  
    
    func testCartesian(){
        let a: [[Int]] = [2, 3, 3].map{Array(0..<$0)}
        let r = Array(Product(a))
        XCTAssertEqual(18, r.count)
        XCTAssertEqual([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2]], r)
        XCTAssertEqual(r.enumerated().reduce(into: [:]){map, el in
            map[el.0] = el.1
        }, [0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 1, 0], 4: [0, 1, 1], 5: [0, 1, 2], 6: [0, 2, 0], 7: [0, 2, 1], 8: [0, 2, 2], 9: [1, 0, 0], 10: [1, 0, 1], 11: [1, 0, 2], 12: [1, 1, 0], 13: [1, 1, 1], 14: [1, 1, 2], 15: [1, 2, 0], 16: [1, 2, 1], 17: [1, 2, 2]])
    }
    
    func testActionFlattener(){
//        let aF = ActionFlattener<Int32>([2,3,3])
//        let r = aF.lookupAction(Tensor<Int32>(2))
//        XCTAssertEqual(Tensor<Int32>([0,0,2]), r)
//        
    }
    
    func testTensor() {
        let given = Tensor<Int32>(repeating: 1, shape: [2,8])
        let result = given.gathering(atIndices: Tensor<Int32>(Int32(0)), alongAxis: 0)
        XCTAssertEqual(TensorShape(8), result.shape)
    }
    
    func testReShape(){
        var arr = Tensor<Float>([1,2,3,4])
        print(arr)
        print(arr.shape)
        arr = arr.reshaped(to: TensorShape(1, arr.shape[0]))
        print(arr)
        print(arr.shape)
    }
    
    func testProcessVectorObservationn() throws {
        let nAgents: Int = 10
        let shapes: [[Int32]] = [[3], [4]]
        let listProto = generateListAgentProto(nAgents, shapes)
        print(listProto)
        for (obsIndex, shape) in shapes.enumerated(){
            let arr = try processVectorObservation(obsIndex: obsIndex, shape: shape, agentInfoList: listProto)
            let expected = [nAgents] + [shape]
            print(arr)
            XCTAssertTrue(arr.scalars.map({ abs($0 - 0.1) < 0.01 }).reduce(true, {$0 && $1}))
        }
    }
    
    func generateListAgentProto(_ nAgent: Int, _ shape: [[Int32]], infiniteRewards: Bool = false, nanObservations: Bool = false) -> [CommunicatorObjects_AgentInfoProto] {
        var result:[CommunicatorObjects_AgentInfoProto] = []
        for agentIndex in Int32(0) ..< Int32(nAgent) {
            var ap = CommunicatorObjects_AgentInfoProto()
            ap.reward = infiniteRewards ? Float.infinity : Float(agentIndex)
            ap.done = agentIndex % 2 == 0
            ap.maxStepReached = agentIndex % 4 == 0
            ap.id = agentIndex
            ap.actionMask += zip(Array(repeating: true, count: 5),Array(repeating: false, count: 5)).flatMap{[$0,$1]}
            var obsProtoList: [CommunicatorObjects_ObservationProto] = []
            for obsIndex in 0 ..< shape.count {
                var obsProto = CommunicatorObjects_ObservationProto()
                obsProto.shape += shape[obsIndex]
                obsProto.floatData.data += Array<Float>(
                    repeating: ( nanObservations ? Float.nan : 0.1 ),
                    count: Int(shape[obsIndex].reduce(1, *))
                )
                obsProtoList.append(obsProto)
            }
            ap.observations += obsProtoList
            result.append(ap)
        }
        return result
    }
}
