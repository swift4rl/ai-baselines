//
//  DiagGaussianProbabilityDistribution.swift
//  environments
//
//  Created by Sercan Karaoglu on 15/10/2020.
//

import Foundation
import TensorFlow
import Darwin

// Below code comes from eaplatanios/swift-rl:
// https://github.com/eaplatanios/swift-rl/blob/master/Sources/ReinforcementLearning/Distributions/Distribution.swift
public protocol Distribution {
  associatedtype Value

  func entropy() -> Tensor<Float>

  /// Returns a random sample drawn from this distribution.
  func sample() -> Value
}

public protocol DifferentiableDistribution: Distribution, Differentiable {
  @differentiable(wrt: self)
  func entropy() -> Tensor<Float>
}

public struct DiagGaussianProbabilityDistribution: DifferentiableDistribution, KeyPathIterable {
        
    public var mean: Tensor<Float32>
    public var logstd: Tensor<Float32>
    @noDerivative public var std: Tensor<Float32>

    /**
    Probability distributions from multivariate Gaussian input

    :param flat: ([float]) the multivariate Gaussian input data
     */
    
    @inlinable
    @differentiable
    init(mean: Tensor<Float32>, logstd: Tensor<Float32>){
        self.mean = mean
        self.logstd = logstd.clipped(min: -20, max: 2)
        self.std = exp(self.logstd)
    }
    

    @inlinable
    @noDerivative
    public func mode() -> Tensor<Float32> {
       // let mode = tanh(self.mean)
       // print("mean: \(self.mean) mode: \(mode)")
        return self.mean
    }
    
    @inlinable
    @noDerivative
    public func sample() -> Tensor<Float32> {
       // print("mean: \(self.mean) mode: \(tanh(self.mean))")
//        let random = Tensor<Float32>(randomNormal: self.mean.shape, mean: Tensor(0), standardDeviation: Tensor(1))
//        return (self.mean + self.std * random).clipped(min: -3, max: 3) / 3
        return self.mean
    }

    @inlinable
    @differentiable
    public func neglogp(of x: Tensor<Float32>) -> Tensor<Float32> {
        let mse =  ((x - self.mean) / (self.std + 1e-5) ).squared()
        
        let sigmaTrace = 2 * logstd

        let log2pi = log(2.0 * Float32.pi)

        let logLikelihood = -0.5 * (mse + sigmaTrace + log2pi)

        return logLikelihood.sum(alongAxes: -1)
    }
    
    @differentiable
    public func logProbability(of value: Tensor<Float32>) -> Tensor<Float> {
        return neglogp(of: value)
    }
    
    @inlinable
    @differentiable
    public func klDivergence(other: DiagGaussianProbabilityDistribution) -> Tensor<Float32> {
        let d1: Tensor<Float32> = other.logstd - self.logstd + (self.std.squared() + (self.mean - other.mean).squared())
        let d2: Tensor<Float32> = (2.0 * other.std.squared()) - 0.5
        return (d1 / d2).sum(alongAxes: -1)
                
    }

    @inlinable
    @differentiable
    public func entropy() -> Tensor<Float32> {
        (self.logstd + 0.5 * log(2.0 * Float32.pi * 2.718281828459045)).sum(alongAxes: -1)
    }

}
