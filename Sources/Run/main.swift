//
//  main.swift
//  App
//
//  Created by Sercan Karaoglu on 01/08/2020.
//
import Foundation
import environments

let rpc = RpcCommunicator(host: "localhost", port: 5004)
rpc.test()
