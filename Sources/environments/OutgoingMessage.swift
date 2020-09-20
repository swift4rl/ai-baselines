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
    
    func writeBool(b: Bool) -> Void {
        var c = b
        self.buffer.writeBytes(Data(bytes: &c, count: 1))
    }
    
    func writeInt32(i: Int32) -> Void {
        self.buffer.writeInteger(i)
    }

    func writeFloat32(f: Float32) -> Void{
        self.buffer.writeInteger(f.bitPattern)
    }

    func writeFloat32List(floatList: [Float]) -> Void{
        self.writeInt32(i: Int32(floatList.count))
        for f in floatList{
            self.writeFloat32(f: f)
        }
    }

    func writeString(s: String) throws -> Void  {
        let bytes = s.data(using: String.Encoding.ascii)!
        let _s = String(data: bytes, encoding: String.Encoding.ascii)!
        self.writeInt32(i: Int32(_s.count))
        try buffer.writeString(_s, encoding: String.Encoding.ascii)
    }

    func setRawBytes(buffer: ByteBuffer) -> Void {
        self.buffer = ByteBuffer(buffer: buffer)
    }
}
