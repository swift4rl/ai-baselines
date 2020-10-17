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

struct DiagGaussianProbabilityDistribution: DifferentiableDistribution, KeyPathIterable {
        
    var flat: Tensor<Float32>
    var mean: Tensor<Float32>
    var logstd: Tensor<Float32>
    var std: Tensor<Float32>
    var actions: Tensor<Float32>
    var sigmas: Tensor<Float32>

    /**
    Probability distributions from multivariate Gaussian input

    :param flat: ([float]) the multivariate Gaussian input data
     */
    @inlinable
    @differentiable(wrt: flat)
    init(flat: Tensor<Float32>){
        //print(flat)
        self.flat = flat
        let x = flat.split(count: 2, alongAxis: flat.shape.count - 1)
        self.actions = x[0]
        //print(actions)
        self.sigmas = x[1]
        //print(sigmas)
        self.mean = actions.mean().reshaped(to: TensorShape(1))
        //print(mean)
        self.std = sigmas.mean().reshaped(to: TensorShape(1))
        //print(std)
        self.logstd = log(std)
}
    
    func flatparam() -> Tensor<Float32> { self.flat }

    @inlinable
    func mode() -> Tensor<Float32> { self.mean }

    @inlinable
    @noDerivative
    func neglogp(of x: Tensor<Float32>) -> Tensor<Float32> {
//        let x1: Tensor<Float32> = exp(-0.5 * pow(((x - self.mean) / self.std),2))
//        let x2: Tensor<Float32> = self.std * sqrt(2 * Float32.pi)
//        return log(x1/x2 + 1e-5)
        let x1 =  0.5 * ( ((x - self.mean) / self.std ).squared()).sum(alongAxes: -1)
        let x2 =  0.5 * log(2.0 * Float32.pi)
        let x3 = self.logstd.sum(alongAxes: -1)
        return x1 + x2 + x3
    }
    
    func logProbability(of value: Tensor<Float32>) -> Tensor<Float> {
        return -neglogp(of: value)
    }
    
    @inlinable
    @differentiable
    func klDivergence(other: DiagGaussianProbabilityDistribution) -> Tensor<Float32> {
        let d1: Tensor<Float32> = other.logstd - self.logstd + (self.std.squared() + (self.mean - other.mean).squared())
        let d2: Tensor<Float32> = (2.0 * other.std.squared()) - 0.5
        return (d1 / d2).sum(alongAxes: -1)
                
    }

    @inlinable
    @differentiable(wrt: self)
    func entropy() -> Tensor<Float32> {
        (self.logstd + 0.5 * log(2.0 * Float32.pi * 2.718281828459045)).sum(alongAxes: -1)
    }

    @inlinable
    func sample() -> Tensor<Float32> { self.mean + self.std * Tensor<Float32>(randomNormal: self.mean.shape) }

}
