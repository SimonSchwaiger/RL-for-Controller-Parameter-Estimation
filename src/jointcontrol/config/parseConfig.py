#!/env/bin/python

import json
import sys
import rospy

if __name__ == "__main__":
    def setParameter(param_name, param_value):
        rospy.set_param(param_name, param_value)

    # Make sure path to config file was given
    assert len(sys.argv)>=1, "No path to Gains Config File was given!"

    # Get path to config file
    path = str(sys.argv[1])
    print("Jointcontrol config path is {}".format(path))

    # Read configuration file containing jointcontroller data
    with open(path) as json_file:
        jsonData = json.load(json_file)

    # Check if the specified numbers of joints are actually included
    assert len(jsonData["Joints"])==jsonData["NumJoints"], "Wrong number of joints specified"

    # Add number of joints as parameter
    setParameter("/jointcontrol/NumJoints", jsonData["NumJoints"])

    # Parse joint dict and create ros parameters
    for j in range(jsonData["NumJoints"]):
        entry = jsonData["Joints"][j]
        # Check if entry is formatted correctly
        assert len(entry["Defaults"])==entry["NumParams"], "Wrong number of given Defaults for ".format(entry["Name"])
        assert len(entry["Minimums"])==entry["NumParams"], "Wrong number of given Minimums for ".format(entry["Name"])
        assert len(entry["Maximums"])==entry["NumParams"], "Wrong number of given Maximums for ".format(entry["Name"])

        # Set up path of current parameter
        name = "/jointcontrol/{}".format(entry["Name"])
        # Add everey key as ROS parameter
        for key in entry:
            setParameter("{}/{}".format(name, key), entry[key])






