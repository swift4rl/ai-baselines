//
//  ActionFlattener.swift
//  environments
//
//  Created by Sercan Karaoglu on 11/10/2020.
//

import Foundation
/**
 Flattens branched discrete action spaces into single-branch discrete action spaces.
 */
class ActionFlattener<T : Numeric & Hashable & Comparable & Strideable> {
//    var actionSpace: Space
//    var actionShape: [T]
//    var actionLookup: [T: [T]]
//    
//    /**
//    Initialize the flattener.
//     - Parameters:
//        - branchedActionSpace: A List containing the sizes of each branch of the action
//    space, e.g. [2,3,3] for three branches with size 2, 3, and 3 respectively.
//    */
//    init(_ branchedActionSpace: [T]) {
//        self.actionShape = branchedActionSpace
//        self.actionLookup = Self.createLookup(self.actionShape)
//        self.actionSpace = Discrete(Int32(self.actionLookup.count))
//    }
//
//    /**
//    Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
//    Each key in the Dict maps to one unique set of branched actions, and each value
//    contains the List of branched actions.
//    */
//    static func createLookup(_ branchedActionSpace: [T]) -> [T: [T]] {
//        let possibleVals: [[T]] = branchedActionSpace.map{Array(stride(from: T.init(exactly: 0)!, to: $0, by: 1))}
//        let allActions = Array(Product(possibleVals))
//        
//        return allActions.enumerated().reduce(into: [:]){map, el in
//            map[T(exactly: el.0)!] = el.1
//        }
//        
//    }
//
//    /**
//    Convert a scalar discrete action into a unique set of branched actions.
//     - Parameters:
//        - action: A scalar value representing one of the discrete actions.
//    - Returns:
//        - The List containing the branched actions.
//    */
//    func lookupAction(_ action: T)-> [T]?{
//        return self.actionLookup[action]
//    }
//    
//    func lookupAction(_ action: Tensor<T>) -> Tensor<T> where T: TensorFlowScalar{
//        var act = action
//        if let a = action.scalar,
//           let lA = lookupAction(a){
//            act = Tensor<T>(shape: TensorShape(lA.count), scalars: lA)
//        }
//        return act;
//    }
//    
}
