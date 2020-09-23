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
}
