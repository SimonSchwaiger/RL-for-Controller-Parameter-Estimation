#!/usr/bin/env python3

# Pybullet components
#from posixpath import join
import pybullet as p
import pybullet_data

# ROS Components
import rospy
#from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
#from jointcontrol.msg import jointMetric

import numpy as np
import time

def clampValue(val, valMax):
    """ Makes sure that a value is within [-valMax, valMax] """
    if valMax == None: return val
    if val > valMax: return valMax
    elif val < -1*valMax: return -1*valMax
    else: return val

class PIDController:
    """ Discrete PID controller approximated using the Tustin (trapezoid) approximation """
    def __init__(self, kp=0, ki=0, kd=0, ts=0, feedforward=0, bufferLength=3) -> None:
        #self.bufferLength = bufferLength
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        self.feedforward = feedforward
        if ts != 0:  self.setConstants(kp, ki, kd, ts)
    #
    def setConstants(self, kp, ki, kd, ts):
        self.k1 = kp+((ki*ts)/2)+((2*kd)/ts)
        self.k2 = ki*ts-((4*kd)/ts)
        self.k3 = (-1*kp)+((ki*ts)/2)+((2*kd)/ts)   
    #
    def update(self, e):
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = e[0]*self.k1 + e[1]*self.k2 + e[2]*self.k3 + y[2]
        return y[0] + e[0]*self.feedforward

class PT1Block:
    """ Discrete PT1 block approximated using the Tustin (trapezoid) approximation """
    def __init__(self, kp=1, T1=0, ts=0, bufferLength=2) -> None:
        self.k1 = 0
        self.k2 = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0: self.setConstants(kp, T1, ts)
    #
    def setConstants(self, kp, T1, ts):
        t = 2*(T1/ts)
        self.k1 = kp/(1+t)
        self.k2 = (1-t)/(1+t)
    #
    def update(self, e):
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = (e[0] + e[1])*self.k1 - y[1]*self.k2
        return y[0]

class DBlock:
    def __init__(self, kd=0, ts=0, bufferLength=2) -> None:
        self.k = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0: self.setConstants(kd, ts)
    #
    def setConstants(self, kd, ts):
        self.k = (2*kd)/ts
    #
    def update(self, e):
        if self.k == 0: return 0
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = e[0]*self.k - e[1]*self.k - y[1]
        return y[0]

class PT2Block:
    """ Discrete PT2 Block approximated using the Tustin approximation """
    def __init__(self, T=0, D=0, kp=1, ts=0, bufferLength=3) -> None:
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.k4 = 0
        self.k5 = 0
        self.k6 = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0:  self.setConstants(T, D, kp, ts)
    #
    def setConstants(self, T, D, kp, ts):
        self.k1 = 4*T**2 + 4*D*T*ts + ts**2
        self.k2 = 2*ts**2 - 8*T**2
        self.k3 = 4*T**2 - 4*D*T*ts + ts**2
        self.k4 = kp*ts**2
        self.k5 = 2*kp*ts**2
        self.k6 = kp*ts**2
    #
    def update(self, e):
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = ( e[0]*self.k4 + e[1]*self.k5 + e[2]*self.k6 - y[1]*self.k2 - y[2]*self.k3 )/self.k1
        return y[0]

class smartPID:
    """ Implementation to mimick HEBI PID controller behaviour. Variable names are consistent with HEBI Gains format """
    def __init__(self, kp=0, ki=0, kd=0, targetLP=0, outputLP=0, ts=0, feedforward=0, d_on_error=True, targetMax=None, outputMax=None) -> None:
        self.d_on_error = d_on_error
        if d_on_error:
            self.PID = PIDController(kp=kp, ki=ki, kd=kd, ts=ts)
            self.D = DBlock(kd=0, ts=ts)
        else:
            self.PID = PIDController(kp=kp, ki=ki, kd=0, ts=ts)
            self.D = DBlock(kd=kd, ts=ts)
        #
        self.targetLP = targetLP
        self.outputLP = outputLP
        self.inputFilter  = PT1Block(kp=1, T1=targetLP, ts=ts)
        self.outputFilter = PT1Block(kp=1, T1=outputLP, ts=ts)
        #
        self.targetMax = targetMax
        self.outputMax = outputMax
        self.feedforward = feedforward
    #
    def setConstants(self, kp, ki, kd, ts):
        if self.d_on_error:
            self.PID.setConstants(kp, ki, kd, ts)
        else:
            self.PID.setConstants(kp, ki, 0, ts)
            self.D.setConstants(kd, ts)
        #
        self.inputFilter  = PT1Block(kp=1, T1=self.targetLP, ts=ts)
        self.outputFilter = PT1Block(kp=1, T1=self.outputLP, ts=ts)
    #
    def update(self, target, feedback):
        # Clamp and low pass input
        filteredInput = self.inputFilter.update(
            clampValue(target, self.targetMax)
        )
        # Update internal control blocks
        output = self.D.update(filteredInput) + self.PID.update(filteredInput - feedback)
        # Clamp and low pass output
        output = self.outputFilter.update(
            clampValue(output + self.feedforward*filteredInput, self.outputMax)
        )
        return output
        
class strategy4Controller:
    """ Models HEBI control strategy 4 using 3 PID controllers discretised using the tustin approximation """
    def __init__(self, ts=0, targetConstraints=[None, 3.43, 20], outputConstraints=[10, 1, 1], feedfowards=[0, 1, 1], d_on_errors=[True, True, False], constants=None) -> None:
        """ Class constructor """
        self.ts = ts
        #
        self.PositionPID = smartPID(
            targetMax=targetConstraints[0],
            outputMax=outputConstraints[0],
            feedforward=feedfowards[0],
            d_on_error=d_on_errors[0]
        )
        #
        self.VelocityPID = smartPID(
            targetMax=targetConstraints[1],
            outputMax=outputConstraints[1],
            feedforward=feedfowards[1],
            d_on_error=d_on_errors[1],
            outputLP=0.01
        )
        #
        self.EffortPID = smartPID(
            targetMax=targetConstraints[2],
            outputMax=outputConstraints[2],
            feedforward=feedfowards[2],
            d_on_error=d_on_errors[2],
            outputLP=0.001
        )
        #
        self.PWMFilter = PT2Block(kp=1, T=0.0, D=10, ts=self.ts)
        if constants != None: self.updateConstants(constants)
    #
    def updateConstants(self, constants):
        self.PositionPID.setConstants(constants[0], constants[1], constants[2], self.ts)
        self.VelocityPID.setConstants(constants[3], constants[4], constants[5], self.ts)
        self.EffortPID.setConstants(  constants[6], constants[7], constants[8], self.ts)
    #
    def update(self, vecIn, feedback):
        """ 
        Takes feedback and control signal and processes output signal 
        
        Format vecIn & feedback: [pos, vel, effort]
        """
        effort = self.PositionPID.update(vecIn[0], feedback[0])
        PWM1 = self.EffortPID.update(vecIn[2] + effort, feedback[2])
        PWM2 = self.VelocityPID.update(vecIn[2], feedback[2])
        return self.PWMFilter.update(PWM1 + PWM2)

def PWM2Torque(PWM, maxMotorTorque=7.5):
    """ Converts PWM output signals of the strategy 4 controller to direct torque outputs (Nm) for Pybullet """
    # PWM range -> [-1, 1], Since we have X5-4 motors, the max torque is 7.5 Nm
    # We just assume the conversion to be linear, might be fancier if I have time to measure a more exact conversion
    # Source: https://docs.hebi.us/core_concepts.html#control-strategies
    return PWM*maxMotorTorque

def deserialiseJointstate(js):
    """ Converts Jointstate message into the format used for the strategy 4 controller """
    return [ [pos, vel, eff] for pos, vel, eff in zip(js.position, js.velocity, js.effort) ]

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
        #p.setGravity(0,0,0)
        planeId = p.loadURDF("plane.urdf")
        #startPos = [0,0,1]
        #startOrientation = p.getQuaternionFromEuler([0,0,0])
        # Load robot model
        #TODO fix Endeffector not loading in
        self.robotID = p.loadURDF("SAImon.urdf")
        # Add damping to mimick motor inertia and friction #TODO: what units are used here in pybullet? Nm and Nm/rad?!
        # https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12685
        p.changeDynamics(self.robotID,1,linearDamping=0.04, angularDamping=70)
        p.changeDynamics(self.robotID,3,linearDamping=0.04, angularDamping=70)
        p.changeDynamics(self.robotID,5,linearDamping=0.04, angularDamping=70)
        #p.setJointMotorControl2(self.robotID,1,p.VELOCITY_CONTROL,targetVelocity=0,force=70)
        #p.setJointMotorControl2(self.robotID,3,p.VELOCITY_CONTROL,targetVelocity=0,force=70)
        #p.setJointMotorControl2(self.robotID,5,p.VELOCITY_CONTROL,targetVelocity=0,force=70)
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
    def torqueControlUpdate(self, torques):
        """ Applies direct force to the robot joints """
        mode = p.TORQUE_CONTROL
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
                #np.array([0,0,0], dtype=float)
                np.array(torques, dtype=float)
            ]
        ).T

#class controlmetricTracker:
#    """ Tracks simulated controller performance and publishes it to the jointcontroller/jointMetric topic """
#    def __init__(self) -> None:
#        # Register publisher
#        self.metricPub = rospy.Publisher(
#            "jointcontroller/jointMetric",
#            jointMetric,
#            queue_size=1
#        )
#    #
#    def updateMetrics(self, targets, feedbacks):
#        """ Publishes a vector of target and feedback positions to be analysed by the gym environment """
#        self.metricPub.publish(
#            jointMetric(
#                [ Float32(t) for t in targets ],
#                [ Float32(f) for f in feedbacks ]
#            )
#        )

class simulatedRobot:
    """ ... """
    def __init__(self, ts=1/100) -> None:
        # Instantiate controllers and simulated robot
        self.controllers = [
            strategy4Controller(ts=ts),
            strategy4Controller(ts=ts),
            strategy4Controller(ts=ts)
        ]
        # Instantiate robot with correct ts
        self.robot = bulletSim(ts=ts)
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
        # Deactivate the internal positional controller
        self.feedback = self.robot.positionControlUpdate(cmdForce=[0.5,0.5,0.5], cmdPos=[0,-1.57,0])
        # Wait for everything to register
        time.sleep(4)
        # Wait for target jointstate to exist
        while(self.targetJS == None): time.sleep(1)
    #
    def jointStateTargetCallback(self, js):
        self.targetJS = js
    #
    def updateControlParams(self, j1Params=None, j2Params=None, j3Params=None):
        """ Sets params for each controller. If they are not provided as args, the params are loaded from ros """
        if j1Params != None: j1Params = rospy.get_param("jointcontrol/J1/Defaults")
        if j2Params != None: j2Params = rospy.get_param("jointcontrol/J2/Defaults")
        if j3Params != None: j3Params = rospy.get_param("jointcontrol/J3/Defaults")
        self.controllers[0].updateConstants(j1Params)
        self.controllers[1].updateConstants(j2Params)
        self.controllers[2].updateConstants(j3Params)
    #
    def updateControlLoop(self, realtime=False):
        # Get control signal in a per joint format from the target jointstate
        controlSignal = deserialiseJointstate(self.targetJS) # 3x3
        # Get torque vector from strategy 4 controller
        torqueVec = [
            PWM2Torque(self.controllers[0].update(controlSignal[0], self.feedback[0])),
            PWM2Torque(self.controllers[1].update(controlSignal[1], self.feedback[1])),
            PWM2Torque(self.controllers[2].update(controlSignal[2], self.feedback[2]))
        ]
        print("Applied Torque: {}".format(torqueVec))
        # Apply torque to simulated robot
        self.feedback = self.robot.torqueControlUpdate(torqueVec)
        print("Joint Positions: {}".format(self.feedback.T[0]))
        # Publish control metrics
        #self.tracker.updateMetrics(controlSignal, self.feedback)
        # Sleep for ts for a discrete real-time simulation
        if realtime: self.looprate.sleep()
        # Return jointmetrics for calculation of reward
        return controlSignal, self.feedback

class jointController:
    """ ... """
    def __init__(self) -> None:
        self.sim = simulatedRobot()

        



if __name__ == "__main__":
    sim = simulatedRobot()
    # Start control loop
    while not rospy.is_shutdown():
        _, _ = sim.updateControlLoop(realtime=True)

    # Testee:
        # Recieves:
            # Initial Pose
            # Controller Params (list of floats with length od numParams)
            # Control Signal (list of floats with length ts*simTime)
        # Returns:
            # MSE of control signal - feedback



class testee:
    """ ... """
    def __init__(self) -> None:
        pass
    #
    def update(initialPose=None, controllerParams=None, controlSignal=None):
        pass 


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
js.position = [1.57,-1.57,1.57]
js.velocity = [0,0,0]
js.effort = [0,0,0]

jspub.publish(js)





js = JointState()
js.name = ['J1', 'J2', 'J3']
js.position = [-1.57,0,1.57]
js.velocity = [0,0,0]
js.effort = [0,0,0]

jspub.publish(js)


"""