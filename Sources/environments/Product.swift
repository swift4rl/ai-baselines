
//
//  https://github.com/Oyvindkg/SwiftProductGenerator/blob/master/Sources/SwiftProductGenerator/SwiftProductGenerator.swift
//
//
//  Created by Øyvind Grimnes on 09/09/15.
//
//

import Foundation

public struct Product<T, C: Collection>: IteratorProtocol, Sequence
    where C.Iterator.Element == T
{

    private var indices : [C.Index]
    private var pools   : [C]
    private var done    : Bool = false


    public init(_ collections: [C]) {
        self.pools   = collections
        self.indices = collections.map{ $0.startIndex }
        self.done    = pools.reduce(true, { $0 && $1.count == 0 })
    }

    public init(repeating collection: C, count: Int) {
        precondition(count >= 0, "count must be >= 0")
        self.init([C](repeating: collection, count: count))
    }

    public mutating func next() -> [T]? {
        if done {
            return nil
        }

        let element = self.pools.enumerated().map {
            $1[ self.indices[$0] ]
        }

        self.incrementLocationInPool(poolIndex: self.pools.count - 1)
        return element
    }

    mutating private func incrementLocationInPool(poolIndex: Int) {
        guard self.pools.indices.contains(poolIndex) else {
            done = true
            return
        }

        self.indices[poolIndex] = self.pools[poolIndex].index(after: self.indices[poolIndex])

        if self.indices[poolIndex] == self.pools[poolIndex].endIndex {
            self.indices[poolIndex] = self.pools[poolIndex].startIndex
            self.incrementLocationInPool(poolIndex: poolIndex - 1)
        }
    }

    public func generate() -> Product {
        return self
    }
}
