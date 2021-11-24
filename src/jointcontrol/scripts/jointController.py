#!/usr/bin/env python3

import rospy
from jointcontrol.msg import jointMetric



rospy.init_note("ControllerInterface")


# Set up rospy loop frequency
looprate = rospy.Rate(10)


# controller update
    # for each controller
        # get soll
        # get ist
        # publish as message

    

