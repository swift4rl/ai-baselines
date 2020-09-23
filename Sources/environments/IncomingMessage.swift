//
//  IncomingMessage.swift
//  environments
//
//  Created by Sercan Karaoglu on 06/09/2020.
//
import Foundation
import NIO

class IncomingMessage {
    var buffer: ByteBuffer
    
    init(buffer: ByteBuffer){
         self.buffer = buffer
    }

    func readBool(defaultValue: Bool = false) -> Bool {
        if buffer.readableBytes < MemoryLayout<Bool>.size {
             return defaultValue
        }
        let data = buffer.readData(length: MemoryLayout<Bool>.size)
        var value: UInt8 = 0x01
        data?.copyBytes(to: &value, count: MemoryLayout<Bool>.size)
        return value == 0x01
    }

    func readInt32(defaultValue: Int32 = 0) -> Int32 {
        if buffer.readableBytes < MemoryLayout<Int32>.size {
             return defaultValue
        }
        guard let ret = buffer.readInteger(as: Int32.self) else {
            return defaultValue
        }
        return ret
    }

    func readFloat32(defaultValue: Float32 = 0.0) -> Float32 {
        if buffer.readableBytes < MemoryLayout<Float32>.size {
             return defaultValue
        }
        let fO = buffer.readInteger(as: UInt32.self).map { Float32(bitPattern: $0) }
        guard let ret = fO else { return defaultValue }
        return ret
    }

    func readFloat32List(defaultValue: [Float32] = []) -> [Float32] {
        if buffer.readableBytes < defaultValue.count * MemoryLayout<Float32>.size {
             return defaultValue
        }
        let listLen = self.readInt32()
        return (0..<listLen).map{ _ -> Float32 in readFloat32() }
    }

    func readString(defaultValue: String = "") -> String {
        if buffer.readableBytes == 0 {
             return defaultValue
        }
        let encodedStrLen = self.readInt32()
        guard let ret = self.buffer.readString(length: Int(encodedStrLen), encoding: String.Encoding.ascii) else { return defaultValue }
        return ret
    }
    
    func getRawBytes() -> [UInt8] {
        return self.buffer.getBytes(at: 0, length: self.buffer.capacity) ?? []
    }
    
    func atEndOfBuffer() -> Bool{
        return self.buffer.readerIndex == self.buffer.capacity
    }
}
