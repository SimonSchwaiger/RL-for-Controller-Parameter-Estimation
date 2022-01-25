#!/usr/bin/env python

import json
import sys
import rospy

if __name__ == "__main__":
    """ Parses json config file and uploads content to ros param server """
    def setParameter(param_name, param_value, namespace):
        """ Sets a ros parameter in a given namespace """
        rospy.set_param("/{}/{}".format(namespace, param_name), param_value)

    def parseDict(dictData, namespace):
        """ Iterates over a dict and replicates it on the ROS parameter server """
        # Iterate over the keys and store params
        # If an entry is a dict itself, iterate over it as well
        for key in dictData:
            if key == "ParamNamespace":
                continue
            entry = dictData[key]
            if isinstance(entry, dict):
                parseDict(entry, "{}/{}".format(namespace, key))
            else:
                setParameter(key, entry, namespace)

    # Make sure path to config file was given
    assert len(sys.argv)>=1, "No path to Config File was given!"

    # Get path to config file
    path = str(sys.argv[1])
    print("Config path is {}".format(path))

    # Read configuration file containing data
    with open(path) as json_file:
        jsonData = json.load(json_file)

    # Get namespace
    namespace = jsonData["ParamNamespace"]

    # Parse dict and add entries to ros parameter server
    parseDict(jsonData, namespace)

