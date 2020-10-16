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

class DiagGaussianProbabilityDistribution: DifferentiableDistribution, KeyPathIterable {
    var probabilities: Tensor<Float32>
    var flat: Tensor<Float32>
    var mean: Tensor<Float32>
    var logstd: Tensor<Float32>
    var std: Tensor<Float32>
    
    /**
    Probability distributions from multivariate Gaussian input

    :param flat: ([float]) the multivariate Gaussian input data
     */
    init(flat: Tensor<Float32>){
        super.init()
        self.flat = flat
        if case let [mean, logstd] = flat.split(sizes: Tensor(2), alongAxis: flat.shape.count){
            self.mean = mean
            self.logstd = logstd
        }
        self.std = logstd.exp()
    }
    
    func flatparam() -> Tensor<Float32> { self.flat }
    
    func mode() -> Tensor<Float32> { self.mean }
    
    func logProbability(of x: Tensor<Float32>) -> Tensor<Float32> {
        0.5 * ((x-self.mean) / self.std).sum(alongAxes: -1)
                      + 0.5 * log2(2.0 * Float32.pi) * x.shape[-1]
                      + self.logstd.sum(alongAxes: -1)
    }
    
    func kl(other: DiagGaussianProbabilityDistribution) -> Tensor<Float32> {
        (other.logstd - self.logstd + (self.std.squared() + (self.mean - other.mean).squared()) /
                (2.0 * other.std.squared()) - 0.5).sum(alongAxes: -1)
    }
    
    func entropy() -> Tensor<Float32> {
        (self.logstd + 0.5 * (2.0 * Float32.pi * M_E).log()).sum(alongAxes: -1)
    }
    
    func sample() -> Tensor<Float32> {
        return self.mean + self.std * tf.compat.v1.random_normal(tf.shape(self.mean),
                                                               dtype=self.mean.dtype)
    }

}
