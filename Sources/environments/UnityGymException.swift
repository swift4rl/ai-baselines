//
//  UnityGymException.swift
//  environments
//
//  Created by Yeshwanth Kumar on 22/09/2020.
//

import Foundation


enum UnityGymWrapperError: Error {
    case unityGymException(reason: String)
}
