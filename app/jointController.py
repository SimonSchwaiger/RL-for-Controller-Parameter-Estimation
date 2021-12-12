#!/usr/bin/env python3

# Pybullet components
import pybullet as p
import pybullet_data

# ROS Components
import rospy
from sensor_msgs.msg import JointState
from jointcontrol.msg import jointMetric

import numpy as np
import time

def clampValue(val, valMax):
    """ Makes sure that a value is within [-valMax, valMax] """
    valMin = -1*valMax
    if val > valMax: return valMax
    elif val < valMin: return valMin
    else: return val

class PIDController:
    """ Discrete PID controller approximated using the Tustin (trapezoid) approximation """
    def __init__(self, kp=0, ki=0, kd=0, ts=0, bufferLength=3) -> None:
        self.bufferLength = bufferLength
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.e = [0 for i in range(self.bufferLength)]
        self.y = [0 for i in range(self.bufferLength)]
        #self.setConstants(kp, ki, kd, ts)
    #
    def setConstants(self, kp, ki, kd, ts):
        self.k1 = kp+((ki*ts)/2)+(2*kd)/ts
        self.k2 = ki*ts-(4*kd)/ts
        self.k3 = -1*kp+(ki*ts)/2+(2*kd)/ts        
    #
    def update(self, e):
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = e[0]*self.k1+e[1]*self.k2+e[2]*self.k3+y[2]
        return y[0]

class strategy4Controller:
    """ Models HEBI control strategy 4 using 3 PID controllers discretised using the tustin approximation """
    def __init__(self, ts=0) -> None:
        """ Class Constructor """
        self.ts = ts
        # Instantiate PIDControllers: Param order is [PositionPID, VelocityPID, EffortPID]
        # https://docs.hebi.us/core_concepts.html#control-strategies
        # Default controller params:
        # https://docs.hebi.us/resources/gains/X5-4_STRATEGY4.xml
        self.PositionPID = PIDController()
        self.VelocityPID = PIDController()
        self.EffortPID = PIDController()
    #
    def updateConstants(self, constants):
        """ Calculates constants in each PID controller """
        self.PositionPID.setConstants(constants[0], constants[1], constants[2], self.ts)
        self.VelocityPID.setConstants(constants[3], constants[4], constants[5], self.ts)
        self.EffortPID.setConstants(constants[6], constants[7], constants[8], self.ts)
    #
    def update(self, vecIn, feedback):
        """ 
        Takes feedback and control signal and processes output signal 
        
        Format vecIn & feedback: [pos, vel, effort]
        """
        # All signal vectors are of form [Position, Velocity, Effort]
        # The values are clamped at the output of each PID controller, like in the Hebi implementation
        # Update Position PID (input = positionIn - positionFeedback)
        effort = clampValue(self.PositionPID.update(vecIn[0] - feedback[0]), 10)
        # Update Effort PID (input = yPositionPID + effortIn - effortFeedback)
        PWM1 = clampValue(self.EffortPID.update(effort + vecIn[2] - feedback[2]), 1)
        # Update Velocity PID (input = velocityIn - velocityFeedback)
        PWM2 = clampValue(self.VelocityPID.update(vecIn[1] - feedback[1]), 1)
        # Return sum of PWM signals
        return PWM1 + PWM2

class bulletSim:
    """ Class that instantiates a simulated robot using Pybullet """
    def __init__(self, ts=1/60):
        """ Class Constructor """
        #
        ## Pybullet Setup
        # Init pybullet client and GUI
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        # Spawn gravity and groundplane
        p.setGravity(0,0,-9.8)
        planeId = p.loadURDF("plane.urdf")
        startPos = [0,0,1]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        # Load robot model
        #TODO fix Endeffector not loading in
        self.robotID = p.loadURDF("SAImon.urdf")
        # Print joint info
        print('Printing Joint Summary')
        for joint in range(p.getNumJoints(self.robotID)):
            print('Joint {}: {}\n'.format(joint, p.getJointInfo(self.robotID, joint)))
        #
        # Set timestep length to match discrete controllers
        p.setTimeStep(ts)
        #
    def __del__(self):
        """ Class Destructor """
        # Close Pybullet simulation
        p.disconnect()
        #
    def positionControlUpdate(self, cmdForce=[0,0,0], cmdPos=[0,0,0]):
        """ Updates Pybullets internal position controller """
        mode = p.POSITION_CONTROL
        p.setJointMotorControl2(self.robotID, 1, controlMode=mode, force=cmdForce[0], targetPosition=cmdPos[0])
        p.setJointMotorControl2(self.robotID, 3, controlMode=mode, force=cmdForce[1], targetPosition=cmdPos[1])
        p.setJointMotorControl2(self.robotID, 5, controlMode=mode, force=cmdForce[2], targetPosition=cmdPos[2])
        p.stepSimulation()
        jointState = np.array([
            p.getJointState(self.robotID, 1),
            p.getJointState(self.robotID, 3),
            p.getJointState(self.robotID, 5)
        ],  dtype=object)
        # Jointstate is formatted as a numpy array so we can easily drop the not needed force information
        return np.delete(jointState, 2, 1).astype(float)
    #
    def applyFriction(self, torque, friction):
        """ Applies friction (in Nm) to a torque (in Nm) """
        # Make sure firction is positive
        friction = abs(friction)
        # If torque is smaller than friction, the joint does not move
        if abs(torque) <= friction:
            return 0
        # If torque is smaller than 0, add friction (since friction takes effect in the opposite direction)
        # Otherwise subtract friction
        if torque < 0: return torque + friction
        else: return torque - friction
    #
    def torqueControlUpdate(self, torques, friction=10):
        """ Applies direct force to the robot joints """
        mode = p.TORQUE_CONTROL
        # Simulate friction directly as part of the joint and apply torque to the robot
        #torques = [ self.applyFriction(torque, friction) for torque in torques ]
        p.setJointMotorControl2(self.robotID, 1, controlMode=mode, force=torques[0])
        p.setJointMotorControl2(self.robotID, 3, controlMode=mode, force=torques[1])
        p.setJointMotorControl2(self.robotID, 5, controlMode=mode, force=torques[2])
        p.stepSimulation()
        jointState = np.array([
            p.getJointState(self.robotID, 1),
            p.getJointState(self.robotID, 3),
            p.getJointState(self.robotID, 5)
        ], dtype=object)
        # Jointstate is formatted as a numpy array so we can easily drop the not needed force information
        # Unfortunately, TORQUE_CONTROL mode does not report back the motor torque
        # Therefore, we need to separately return the input torque as the torque that is present at the joint
        return np.array(
            [
                jointState[:,0].astype(float),
                jointState[:,1].astype(float),
                np.array(torques, dtype=float)
            ]
        ).T
        #return np.delete(jointState, 2, 1).astype(float)

def PWM2Torque(PWM, maxMotorTorque=7.5):
    """ Converts PWM output signals of the strategy 4 controller to direct torque outputs (Nm) for Pybullet """
    # PWM range -> [-1, 1], Since we have X5-4 motors, the max torque is 7.5 Nm
    # Source: https://docs.hebi.us/core_concepts.html#control-strategies
    return (PWM+1)*(maxMotorTorque/2)

def deserialiseJointstate(js):
    """ Converts Jointstate message into the format used for the strategy 4 controller """
    return [ [pos, vel, eff] for pos, vel, eff in zip(js.position, js.velocity, js.effort) ]

class simulatedRobot:
    """ ... """
    def __init__(self, ts=1/60) -> None:
        # Instantiate controllers and simulated robot
        self.controllers = [
            strategy4Controller(ts=ts),
            strategy4Controller(ts=ts),
            strategy4Controller(ts=ts)
        ]
        # Instantiate robot with correct ts
        self.robot = bulletSim(ts=ts)
        # Deactivate the internal positional controller
        feedback = self.robot.positionControlUpdate(cmdForce=[0,0,0])
        print(feedback)
        # Init Node and get controller params
        rospy.init_node("ControllerInterface")
        j1Params = rospy.get_param("jointcontrol/J1/Defaults")
        j2Params = rospy.get_param("jointcontrol/J2/Defaults")
        j3Params = rospy.get_param("jointcontrol/J3/Defaults")
        # Set params for each controller
        self.controllers[0].updateConstants(j1Params)
        self.controllers[1].updateConstants(j2Params)
        self.controllers[2].updateConstants(j3Params)
        # Placeholder for current target jointstate
        self.targetJS = None
        # Subscribe to jointstate target topic
        self.jointstateSub = rospy.Subscriber(
            "jointcontroller/jointstateTarget",
            JointState, 
            self.jointStateTargetCallback
        )
        # Set up rospy loop frequency
        self.looprate = rospy.Rate(1/ts)
        # Wait for everything to register
        time.sleep(4)
        # Wait for target jointstate to exist
        while(self.targetJS == None): time.sleep(1)
        # Start control loop
        while not rospy.is_shutdown():
            # Get control signal in a per joint format from the target jointstate
            controlSignal = deserialiseJointstate(self.targetJS) # 3x3
            print(controlSignal)
            print(feedback)
            # Get torque vector from strategy 4 controller
            torqueVec = [
                PWM2Torque(self.controllers[0].update(controlSignal[0], feedback[0])),
                PWM2Torque(self.controllers[1].update(controlSignal[1], feedback[1])),
                PWM2Torque(self.controllers[2].update(controlSignal[2], feedback[2]))
            ]
            print(torqueVec)
            # Apply torque to simulated robot
            feedback = self.robot.torqueControlUpdate(torqueVec)
            # Sleep for ts for a discrete real-time simulation
            self.looprate.sleep()
    #
    def jointStateTargetCallback(self, js):
        self.targetJS = js
    #
    def updateControlParams(self, j1Params, j2Params, j3Params):
        """ Sets params for each controller """
        self.controllers[0].updateConstants(j1Params)
        self.controllers[1].updateConstants(j2Params)
        self.controllers[2].updateConstants(j3Params)

if __name__ == "__main__":
    sim = simulatedRobot()


"""
DEBUG



docker exec -it ros_ml_container bash

source ~/myenv/bin/activate
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

python







from sensor_msgs.msg import JointState
import rospy

rospy.init_node("TestPub")

jspub = rospy.Publisher("jointcontroller/jointstateTarget", JointState, queue_size=1)

js = JointState()
js.name = ['J1', 'J2', 'J3']
js.position = [0,0,0]
js.velocity = [0,0,0]
js.effort = [0,0,0]

jspub.publish(js)



"""