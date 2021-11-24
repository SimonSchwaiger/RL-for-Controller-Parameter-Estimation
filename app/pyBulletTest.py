import pybullet as p
import pybullet_data

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, PointStamped
import tf

from inaccuracy_compensation import getJointstate

import time

# Init pybullet client and GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

# Spawn gravity and groundplane
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])

# Load robot model
#TODO fix Endeffector not loading in
robotID = p.loadURDF("SAImon.urdf")

# Print joint info
print('Printing Joint Summary')
for joint in range(p.getNumJoints(robotID)):
    print('Joint {}: {}\n'.format(joint, p.getJointInfo(robotID, joint)))


## Pybullet PID Control - TODO: change to FORCE_CONTROL and implement virtual controller

# Sim is run in ros callback in order to always use the most up to date jointstates
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

# Setup rospy node
rospy.init_node('simulatedSAImon', anonymous=True)

# Link subscriber to callback to start control loop
rospy.Subscriber("joint_states", JointState, jointCallback)


## Inverse Kinematics Control - TEMPORARY

# Set up tflistener to listen for position of interactive marker
tflistener = tf.TransformListener()

# Set up tf publisher for publishing to gym/jointstates
jointpub = rospy.Publisher("/gym/jointstates", JointState, queue_size = 1)

# Setup method to retrieve interactive marker position
def get_object_position():
    """ Gets the current object position from the tf-color-transformation transformpublisher and converts it to a valid robot state. """
    pt = PointStamped()
    pt.header.frame_id = "object_tf"
    pt.header.stamp = rospy.Time()
    pt.point.x = 0
    pt.point.y = 0
    pt.point.z = 0
    while True:
        try:
            pt = tflistener.transformPoint("world", pt)
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("No Transform available")
            pass
    print(pt)
    return pt

# Setup jointstate for kinematics updates
js = JointState()
js.name = ['Arm/J1', 'Arm/J2', 'Arm/J3']

# Calculate inverse kinematics and publish jointstates accordingly
while not rospy.is_shutdown():
    time.sleep(0.1)
    pt = get_object_position()
    j3, j2, j1 = getJointstate(pt.point.x, pt.point.y, pt.point.z)
    js.header.stamp = rospy.Time.now()
    js.position = [j1, j2, j3]
    jointpub.publish(js)
    


# Close Pybullet simulation
p.disconnect()



# useful:

# https://stackoverflow.com/questions/61993517/applying-torque-control-in-pybyllet-makes-object-fly-away-from-the-secene


# code example:

# https://github.com/bulletphysics/pybullet_robots/blob/master/laikago.py