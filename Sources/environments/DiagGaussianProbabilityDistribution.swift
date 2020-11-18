//
//  DiagGaussianProbabilityDistribution.swift
//  environments
//
//  Created by Sercan Karaoglu on 15/10/2020.
//

import Foundation
import ReinforcementLearning
import TensorFlow
import Darwin

public struct DiagGaussianProbabilityDistribution: DifferentiableDistribution, KeyPathIterable {
        
    public var flat: Tensor<Float32>
    public var mean: Tensor<Float32>
    public var logstd: Tensor<Float32>
    public var std: Tensor<Float32>

    /**
    Probability distributions from multivariate Gaussian input

    :param flat: ([float]) the multivariate Gaussian input data
     */
    @inlinable
    @differentiable(wrt: flat)
    init(flat: Tensor<Float32>){
        self.flat = flat
        let x = flat.split(count: 2, alongAxis: flat.shape.count - 1)
        self.mean = x[0]
        self.logstd = x[1]
        self.std = exp(self.logstd)
}
    
    func flatparam() -> Tensor<Float32> { self.flat }

    @inlinable
    public func mode() -> Tensor<Float32> { self.mean }

    @inlinable
    @differentiable
    public func neglogp(of x: Tensor<Float32>) -> Tensor<Float32> {
        let mse =  ( ((x - self.mean) / (self.std + 1e-5) ).squared()).sum(alongAxes: -1)
        
        let sigmaTrace = logstd.sum(alongAxes: -1)

        let log2pi = log(2.0 * Float32.pi)

        let logLikelihood = 0.5 * mse + sigmaTrace + 0.5 * log2pi

        return logLikelihood
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

    @inlinable
    @differentiable
    public func sample() -> Tensor<Float32> {
        return tanh(self.mean + self.std * Tensor<Float32>(randomNormal: self.mean.shape))
    }
}
