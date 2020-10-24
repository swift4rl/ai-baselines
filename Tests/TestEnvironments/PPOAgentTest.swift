//
//  PPOAgentTest.swift
//  TestEnvironments
//
//  Created by Sercan Karaoglu on 23/10/2020.
//

import Foundation
import XCTest
import TensorFlow
@testable import environments

final class PPOAgentTest: XCTestCase {
    func testNegLog(){
        let proba = DiagGaussianProbabilityDistribution(flat: Tensor<Float32>([0.5774765, -0.72072923]))
        let res = proba.neglogp(of: Tensor<Float32>([1.0542214]))
        XCTAssertEqual(res, Tensor<Float32>([0.6785612]))
    }
}
