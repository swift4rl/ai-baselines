//
//  Exception.swift
//  environments
//
//  Created by Sercan Karaoglu on 31/08/2020.
//

import Foundation

public enum UnityException: Error {
    case UnityEnvironmentException(reason: String)
    case UnityCommunicatorStoppedException(reason: String)
    case UnityActionException(reason: String)
    case UnityGymException(reason: String)
    case UnityCommunicationException(reason: String)
    case UnitySideChannelException(reason: String)
	case UnityObservationException(reason: String)
}
