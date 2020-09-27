//
//  Space.swift
//  environments
//
//  Created by Sercan Karaoglu on 26/09/2020.
//

import Foundation
import TensorFlow

protocol Space {
}

struct Discrete: Space {
    init(_ branch: Int32) {
        
    }
}

struct MultiDiscrete: Space {
    init(_ branches: [Int32]) {
        
    }
}

struct Box<Scalar: TensorFlowScalar>: Space {
    init(min: Tensor<Scalar>, max: Tensor<Scalar>) {
    }
    init(min: Scalar, max: Scalar, shape: [Int32]?) {
    }
}

struct Tuple: Space {
    init(_ spaces: [Space]) {
    }
}
