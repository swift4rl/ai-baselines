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
    
    func testReturns() {
//        let trajectory = Trajectory(states: [],
//                   actions: [ 0.6068515 , 0.5681956 , 1.1042405 ,-1.6569525 ,-1.0698782 ,-0.23497434, -0.42413482, 0.7799508 , 0.8121773 , 1.3566769 ].map({Tensor($0)}),
//                   values: [0.16713698,0.2267337 ,0.23888506,0.25573793,0.27021408,0.2639263, 0.2693527 ,0.27208522,0.3037117 ,0.34298047].map({Tensor($0)}),
//                   rewards: [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
//                   logProbs: [1.1039613 ,1.0813456 ,1.5310276 ,2.2869434 ,1.4888189 ,0.94611955, 1.0080734 ,1.224591  ,1.2508421 ,1.8432592 ].map({Tensor($0)}),
//                   isDones: Array(repeating: false, count: 10)
//        )
//        let expectedReturns = Tensor<Float32>([1.06030595,1.0091256, 0.95406783,0.89463987,0.83069035,0.763026, 0.69079542,0.6138513,0.53037493,0.43955067])
//        let returns = trajectory.returns(discount: 0.99, lam: 0.95)
//        XCTAssertEqual(expectedReturns, returns)
    }
}
