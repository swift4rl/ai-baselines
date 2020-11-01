//
//  OutgoingMessage.swift
//  environments
//
//  Created by Sercan Karaoglu on 05/09/2020.
//

import Foundation
import NIO

class OutgoingMessage {
    var buffer: ByteBuffer = ByteBuffer()
    
    func writeBool(_ b: Bool) -> Void {
        var c = b
        self.buffer.writeBytes(Data(bytes: &c, count: 1))
    }
    
    func writeInt32(_ i: Int32) -> Void {
        self.buffer.writeInteger(i, endianness: .little)
    }

    func writeFloat32(_ f: Float32) -> Void{
        self.buffer.writeInteger(f.bitPattern, endianness: .little)
    }

    func writeFloat32List(_ floatList: [Float]) -> Void{
        self.writeInt32(Int32(floatList.count))
        for f in floatList{
            self.writeFloat32(f)
        }
    }

    func writeString(_ s: String) throws -> Void  {
        let bytes = s.data(using: String.Encoding.utf16)!
        let _s = String(data: bytes, encoding: String.Encoding.utf16)!
        self.writeInt32(Int32(_s.count))
        try buffer.writeString(_s, encoding: String.Encoding.utf16)
    }

    func setRawBytes(_ buffer: ByteBuffer) -> Void {
        self.buffer = ByteBuffer(buffer: buffer)
    }
}
