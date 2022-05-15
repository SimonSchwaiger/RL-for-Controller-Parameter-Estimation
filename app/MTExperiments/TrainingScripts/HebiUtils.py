
import hebi
import numpy as np




def createTrajectory(config, ts=0.01):
    """
    Creates a trajectory using Hebi's trajectory generator based on a config file

    Based on Hebi python api example:
    https://docs.hebi.us/tools.html#python-api
    """
    # Sample random trajectory from provided trajectories
    if config["sampleRandom"] == True:
        idx = np.random.randint(0, len(config["timePosSeries"])-1)
    else:
        idx = 0
    # Create time, pos, vel and acc arrays
    times = np.array(config["timePosSeries"][idx]["times"], dtype=float)
    #
    numJoints = 1
    numWaypoints = len(config["timePosSeries"][idx]["positions"])
    #
    pos = np.empty((numJoints, numWaypoints))
    vel = np.empty((numJoints, numWaypoints))
    acc = np.empty((numJoints, numWaypoints))
    # Set first and last waypoint values to 0.0
    vel[:,0] = acc[:,0] = 0.0
    vel[:,-1] = acc[:,-1] = 0.0
    # Set all other values to NaN
    vel[:,1:-1] = acc[:,1:-1] = np.nan
    # Set positions
    for t, inpos in enumerate(config["timePosSeries"][idx]["positions"]):
        pos[:,t] = inpos
    # Creaete trajectory generator
    trajectory = hebi.trajectory.create_trajectory(times, pos, vel, acc)
    # Return trajectory of correct length and time discretisation
    return np.array([ trajectory.get_state(t) for t in np.arange(0, config["samplesPerStep"]) * ts ])


if __name__=="__main__":
    # Plot sample trajectory
    config = { 
        "sampleRandom": True,
        "timePosSeries": [
            {"times": [0, 1, 3], "positions": [0, 1, 2]},
            {"times": [0, 1, 3], "positions": [0, 1, 2]},
            {"times": [0, 1, 3], "positions": [0, 1, 2]},
        ],          
        "samplesPerStep": 150, 
        "maxSteps": 40 
    }

    time = np.arange(0, config["samplesPerStep"])*0.01
    res = createTrajectory(config, ts=0.01)

    import matplotlib.pyplot as plt
    plt.plot(time, np.array(res)[:,0])
    plt.show()


