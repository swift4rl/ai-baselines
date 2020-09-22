using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class PlatformAgent : Agent
{
    public bool lockDirection = true;

    public float forceMultiplier = 10;
    public float areaSize = 10;

    public GameObject pole;

    private Rigidbody platformRigidbody;
    private Rigidbody poleRidgidbody;
    void Start()
    {
        platformRigidbody = GetComponent<Rigidbody>();
        poleRidgidbody = pole.GetComponent<Rigidbody>();

        if (lockDirection)
        {
            poleRidgidbody.constraints = RigidbodyConstraints.FreezePositionX | RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationY;
            platformRigidbody.constraints = RigidbodyConstraints.FreezePositionZ | RigidbodyConstraints.FreezePositionY | RigidbodyConstraints.FreezeRotationZ | RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationY;
        } else
        {
            poleRidgidbody.constraints = RigidbodyConstraints.None;
            platformRigidbody.constraints = RigidbodyConstraints.FreezePositionY | RigidbodyConstraints.FreezeRotationZ | RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationY;
        }
    }

    public override void OnEpisodeBegin()
    {
        platformRigidbody.angularVelocity = Vector3.zero;
        platformRigidbody.velocity = Vector3.zero;
        gameObject.transform.localPosition = new Vector3(0, 0, 0);

        pole.transform.localPosition = new Vector3(0, 1.11f, 0);
        pole.transform.rotation = Quaternion.Euler(lockDirection ? 0 : Random.Range(-5, 5), lockDirection ? 0 : Random.Range(-5, 5), lockDirection ? Random.Range(-10, 10) : Random.Range(-5, 5));
        poleRidgidbody.angularVelocity = Vector3.zero;
        poleRidgidbody.velocity = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if(lockDirection) {
            sensor.AddObservation(pole.transform.localPosition.x);
            sensor.AddObservation(pole.transform.localPosition.y);
            sensor.AddObservation(pole.transform.rotation.x);
            sensor.AddObservation(pole.transform.rotation.y);
            sensor.AddObservation(poleRidgidbody.velocity.x);
            sensor.AddObservation(poleRidgidbody.velocity.y);

            sensor.AddObservation(gameObject.transform.localPosition.x);
            sensor.AddObservation(platformRigidbody.velocity.x);
        } else {
            sensor.AddObservation(pole.transform.localPosition);
            sensor.AddObservation(pole.transform.rotation);
            sensor.AddObservation(poleRidgidbody.velocity);

            sensor.AddObservation(gameObject.transform.localPosition);
            sensor.AddObservation(platformRigidbody.velocity.x);
            sensor.AddObservation(platformRigidbody.velocity.z);
        }  
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = vectorAction[0];

        if (!lockDirection)
        {
            controlSignal.z = vectorAction[1];
        }
        
        platformRigidbody.AddForce(controlSignal * forceMultiplier);

        // Pole fell off platform
        if (pole.transform.localPosition.y < 0.2f || Mathf.Abs(gameObject.transform.localPosition.z) > areaSize || Mathf.Abs(gameObject.transform.localPosition.x) > areaSize)
        {
            SetReward(-1f);
            EndEpisode();
        } else
        {
            SetReward(0.1f);
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        actionsOut[0] = Input.GetAxis("Horizontal");
        actionsOut[1] = Input.GetAxis("Vertical");
    }
}
