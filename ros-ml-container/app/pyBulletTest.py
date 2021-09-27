import pybullet as p
import pybullet_data

import rospy
from sensor_msgs.msg import JointState

import time

# init pybullet client and GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

# gravity an groundplane
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])


# load robot model
#TODO fix Endeffector not loading in
robotID = p.loadURDF("SAImon.urdf")

# print joint info
print('Printing Joint Summary')
for joint in range(p.getNumJoints(robotID)):
    print('Joint {}: {}\n'.format(joint, p.getJointInfo(robotID, joint)))


# sim is run in ros callback in order to always use the most up to date jointstates

def jointCallback(data):
    # perform sim update step
    cmdForce =  [60, 60, 60]
    cmdPos = data.position
    #mode = p.VELOCITY_CONTROL
    mode = p.POSITION_CONTROL
    p.setJointMotorControl2(robotID, 1, controlMode=mode, force=cmdForce[0], targetPosition=cmdPos[0])
    p.setJointMotorControl2(robotID, 3, controlMode=mode, force=cmdForce[1], targetPosition=cmdPos[1])
    p.setJointMotorControl2(robotID, 5, controlMode=mode, force=cmdForce[2], targetPosition=cmdPos[2])
    p.stepSimulation()


rospy.init_node('simulatedSAImon', anonymous=True)
rospy.Subscriber("joint_states", JointState, jointCallback)











# close
p.disconnect()








# run sim
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
#for i in range (5000):
#    p.stepSimulation()
#    time.sleep(1./240.)


# get the number of joints as part of the robot
numJoints = p.getNumJoints(boxID)


# get info about each joint
for idx in range(numJoints):
    print('n.{}:      {}\n'.format(idx, p.getJointInfo(boxID, idx)))


# set control strategy to direct torque
# joint friction is simulated by applying a small standard force by the built in controller
# the arms are joints 1, 3 and 5
cmdForce =  [0, 0, 0]
mode = p.VELOCITY_CONTROL
p.setJointMotorControl2(robotID, 1, controlMode=mode, force=maxForce)
p.setJointMotorControl2(robotID, 3, controlMode=mode, force=maxForce)
p.setJointMotorControl2(robotID, 5, controlMode=mode, force=maxForce)






# cleanup
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotID)
print(cubePos,cubeOrn)
p.disconnect()



#TODO migrate everything into class for easy of use
class simRobot:
    """ .. """

    def __init__(self):
        pass

    def __del__(self):
        pass






# useful:

# https://stackoverflow.com/questions/61993517/applying-torque-control-in-pybyllet-makes-object-fly-away-from-the-secene


# code example:

# https://github.com/bulletphysics/pybullet_robots/blob/master/laikago.py