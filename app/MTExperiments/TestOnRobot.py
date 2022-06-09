
import numpy as np
from time import sleep, time
from math import pi
import hebi

#############################################################################
## GET TRAJECTORY LOOKUP AND EVAL LOGS FOR PARAMETER DETERMINATION

logdir = "Data/11PickAndPlaceOptimisation"

## Load Trajectory - parameter pairs
featureLookup = np.load("{}/lookup.npy".format(logdir), allow_pickle=True)

## Load Eval Episode Logs
testRuns = 3
joints = 3
#model = "DDPG"
model = "PPOCont"

evalLogs = []

for jid in range(joints):
    tmp = []
    for run in range(testRuns):
        filename = "{}/testepisodes_{}_jid{}_{}_0.npy".format(logdir, model, jid, run)
        tmp.append(np.load(filename, allow_pickle=True).item())
    evalLogs.append(tmp)


## Flatten logs over training runs
Observations = []
Rewards = []

for jid in range(joints):
    tmpobs = []
    tmpreward = []
    for run in range(testRuns):
        tmpobs.extend(evalLogs[jid][run]["evalEpisodeObservations"])
        tmpreward.extend(evalLogs[jid][run]["evalEpisodeRewards"])
    Observations.append(np.array(tmpobs))
    Rewards.append(np.array(tmpreward))


def matchVectors(vec1, vec2, dec=2):
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)
    return np.all(np.round(vec1, decimals=dec) == np.round(vec2, decimals=dec))

def getTrajectoryParams(jid, start, goal, lookup, Rewards, Observations):
    """ Determines Parameters for a known trajectory from start to goal """
    # Get features from lookup
    for entry in lookup[jid]:
        if abs(entry[0][0]-start) < 0.01 and abs(entry[0][-1]-goal) < 0.01:
            features = entry[1]
            break
    #
    # Get idx of all Observations with these features
    trajBools = [ matchVectors( entry, features ) for entry in Observations[jid][:,9:14] ]
    #
    # Get idx of best performing configuration
    bestPerformerIdx = np.argmax(np.array(Rewards[jid])[trajBools])
    #
    # Get idx of best performing configuration over all results
    bestPerformerIdx = [i for i, n in enumerate(trajBools) if n == True][bestPerformerIdx]
    #
    # Get Params corresponding with best performer
    params = Observations[jid][:,:9][bestPerformerIdx]
    # Return params
    return params

#getTrajectoryParams(0, 0.635761, 0.068747, lookup, Rewards, Observations)

#############################################################################
## ROBOT CONTROL FUNCTIONS

def getTrajectoryGen(startpos, endpos, cycleTime=1.5, numJoints=3, numWaypoints=2):
    """ Creates trajectory generator """
    # Create time, pos, vel and acc arrays
    times = np.arange(0, cycleTime*numWaypoints, cycleTime)
    #
    pos = np.array([ [x, y] for x, y in zip(startpos, endpos)], dtype=float)
    vel = np.empty((numJoints, numWaypoints))
    acc = np.empty((numJoints, numWaypoints))
    # Set first and last waypoint values to 0.0
    vel[:,0] = acc[:,0] = 0.0
    vel[:,-1] = acc[:,-1] = 0.0
    # Set all other values to NaN
    vel[:,1:-1] = acc[:,1:-1] = np.nan
    # Creaete trajectory generator
    return hebi.trajectory.create_trajectory(times, pos, vel, acc)

def getPaddedTrajectoryGen(startpos, endpos, cycleTime=1.5, numJoints=3, numWaypoints=2, padding=4):
    """ Creates trajectory generator with padding before start and after goalpose """
    # Create time array
    # time is padded based on specified time
    times = np.arange(0, (2*padding)*cycleTime+cycleTime*numWaypoints, cycleTime)
    #times = np.append(times, np.array([times[-1]+padding]))
    #times = np.append(0, times)
    #
    # Create position array and extend it with padding
    pos = np.array([ [x, y] for x, y in zip(startpos, endpos)], dtype=float)
    posPadded = np.zeros((pos.shape[0], pos.shape[1]+(2*padding)), dtype=float)
    posPadded[:,padding:padding+pos.shape[1]] = pos
    #
    # Create empty velocity and effort arrays
    vel = np.empty((numJoints, numWaypoints+(2*padding)))
    acc = np.empty((numJoints, numWaypoints+(2*padding)))
    # Set first, last and padded waypoint values to 0.0
    vel[:,0] = acc[:,0] = 0.0
    vel[:,-1] = acc[:,-1] = 0.0
    for i in range(padding):
        # Set padding position
        posPadded[:,i] = pos[:,0]
        posPadded[:,-1-i] = pos[:,-1]
        # Set padding velocity
        vel[:,i] = 0.0
        vel[:,-1-i] = 0.0
    #
    # Set all other vel values to NaN
    vel[:,padding:-padding] = acc[:,padding:-padding] = np.nan
    # Creaete trajectory generator
    return hebi.trajectory.create_trajectory(times, posPadded, vel, acc)

def setGains(gainVec0, gainVec1, gainVec2):
    """ Method for setting controller gains """
    ## Set embedded controller gains
    # Command for setting gains
    #
    gain_command = hebi.GroupCommand(group.size)
    # Unpack the three gain vectors into group command
    gain_command.position_kp = [ gainVec0[0], gainVec1[0], gainVec2[0] ]
    gain_command.position_ki = [ gainVec0[1], gainVec1[1], gainVec2[1] ]
    gain_command.position_kd = [ gainVec0[2], gainVec1[2], gainVec2[2] ]
    gain_command.velocity_kp = [ gainVec0[3], gainVec1[3], gainVec2[3] ]
    gain_command.velocity_ki = [ gainVec0[4], gainVec1[4], gainVec2[4] ]
    gain_command.velocity_kd = [ gainVec0[5], gainVec1[5], gainVec2[5] ]
    gain_command.velocity_kp = [ gainVec0[6], gainVec1[6], gainVec2[6] ]
    gain_command.velocity_ki = [ gainVec0[7], gainVec1[7], gainVec2[7] ]
    gain_command.velocity_kd = [ gainVec0[8], gainVec1[8], gainVec2[8] ]
    #
    ## Set other parameters to match config as specified in thesis
    # Feed forward
    gain_command.position_feed_forward = np.array([0., 0., 0.])
    #gain_command.velocity_feed_forward = np.array([1., 0.25, 1.])
    #gain_command.effort_feed_forward = np.array([1., 0.05, 1.])
    gain_command.velocity_feed_forward = np.array([1., 1., 1.])
    gain_command.effort_feed_forward = np.array([1., 1., 1.])
    # D on error
    gain_command.position_d_on_error = [True, True, True]
    gain_command.velocity_d_on_error = [True, True, True]
    gain_command.effort_d_on_error = [False, False, False]
    # Target Low Pass
    #gain_command.position_target_lowpass = np.array([1., 1., 1.])
    #gain_command.velocity_target_lowpass = np.array([1., 1., 1.])
    #gain_command.effort_target_lowpass = np.array([1., 1., 1.])
    # Output Low Pass
    #gain_command.position_output_lowpass = np.array([1., 1., 1.])
    gain_command.velocity_output_lowpass = np.array([.75, .75, .75])
    gain_command.effort_output_lowpass = np.array([.9, .9, .9])
    # I Clamp
    gain_command.position_i_clamp = np.array([1., 1., 1.])
    gain_command.velocity_i_clamp = np.array([.25, .25, .25])
    gain_command.effort_i_clamp = np.array([.25, .25, .25])
    # Target Limits
    gain_command.position_max_target = np.array([50., 50., 50.])
    gain_command.velocity_max_target = np.array([3.435, 1.502675, 3.435])
    gain_command.effort_max_target = np.array([10., 10., 10.])
    gain_command.position_min_target = np.array([-50., -50., -50.])
    gain_command.velocity_min_target = np.array([-3.435, -1.502675, -3.435])
    gain_command.effort_min_target = np.array([-10., -10., -10.])
    # Output Limits
    gain_command.position_max_output = np.array([10., 10., 10.])
    gain_command.velocity_max_output = np.array([1., 1., 1.])
    gain_command.effort_max_output = np.array([1., 1., 1.])
    gain_command.position_min_output = np.array([-10., -10., -10.])
    gain_command.velocity_min_output = np.array([-1., -1., -1.])
    gain_command.effort_min_output = np.array([-1., -1., -1.])
    # Send command with acknowledgment
    if not group.send_command_with_acknowledgement(gain_command):
        print('Failed to receive ack from group for gain command')

## Control group of modules to perform trajectory
# Source: https://github.com/HebiRobotics/hebi-python-examples/blob/master/basic/07a_robot_3_dof_arm.py
def execute_trajectory(group, trajectory, feedback):
    # Set up command object, timing variables, and other necessary variables
    num_joints = group.size
    command = hebi.GroupCommand(num_joints)
    duration = trajectory.duration
    #
    start = time()
    t = time() - start
    #
    while t < duration:
        # Get feedback and update the timer
        group.get_next_feedback(reuse_fbk=feedback)
        t = time() - start
        #
        # Get new commands from the trajectory
        pos_cmd, vel_cmd, eff_cmd = trajectory.get_state(t)
        #
        # Fill in the command and send commands to the arm
        command.position = pos_cmd
        #command.velocity = vel_cmd
        #command.effort = eff_cmd
        group.send_command(command)
        #sleep(0.01)

def getLogfileName(model, trajectory, useOptimised):
    if useOptimised == True:
        return "hebilog_{}_Optimised_traj{}".format(model, trajectory)
    else:
        return "hebilog_Default_traj{}".format(trajectory)


#############################################################################
## Perform Movement

poses = np.array([
    [-1.988591, -1.398532, -1.617373],       # Put down
    [0.635761, -0.459321, -1.147022],       # Pick up
    [0.068747, -1.772611, -0.514472]       # Home
])

## Extracted Params for Documentation
DDPGparameters = np.array([
    [
        [30, 0., 0., 0.005, 0., 0., 0.25, 0., 0.001],
        [40., 0.46, 0., 0., 0.742, 1., 0., 0., 0.],
        [20.001, 0.1, 0., 2.55, 0.1, 0.1, 0., 0.1, 0.]
    ], # Put Down -> Pick Up
    [
        [40., 1., 0., 10., 0.969, 1., 0. ,1 ,0.],
        [30., 0., 0., 0.05, 0., 0., 0.25, 0., 0.001],
        [37.21, 1., 0., 0.437, 0., 0.327, 5., 1., 0.]
    ]# Pick up -> Home
])

PPOParameters = np.array([
    [
        [30, 0., 0., 0.05, 0., 0., 0.25, 0., 0.001],
        [39.01, 0., 0., 3.701, 0.2, 0., 0.02, 0.041, 0.],
        [15.59, 0., 0., 0., 0.728, 0., 5., 0.486, 0.]
    ],
    [
        [40, 0., 0., 8.528, 0.05, 1., 5., 0.001, 0.],
        [30, 0., 0., 0.05, 0., 0., 0.25, 0., 0.001],
        [24.012, 0., 0., 0., 0.2, 0., 2.25, 0.149, 0.]
    ]
])


## Discover hebi family
family_names = ["SAImon", "SAImon", "SAImon"]
module_names = ["Arm/J1", "Arm/J2", "Arm/J3"]

lookup = hebi.Lookup()
group = lookup.get_group_from_names(family_names, module_names)

if group==None: print("Group could not be initialised!")


useOptimisedParams = False
trajectoryidx = 1

start = poses[trajectoryidx]
goal = poses[trajectoryidx+1]

## Generate Trajectory
trajectory = getPaddedTrajectoryGen(start, goal)

## Set Controller parameters
if useOptimisedParams == True:
    ## Set optimised parameters
    gainVec0 = getTrajectoryParams(0, start[0], goal[0], featureLookup, Rewards, Observations)
    gainVec1 = getTrajectoryParams(1, start[1], goal[1], featureLookup, Rewards, Observations)
    gainVec2 = getTrajectoryParams(2, start[2], goal[2], featureLookup, Rewards, Observations)
    print("Optimised Params: \n {} \n {} \n {}".format(gainVec0, gainVec1, gainVec2))
    setGains(gainVec0, gainVec1, gainVec2)
else:
    # Set default params
    gainVec0 = [30, 0, 0, 0.25, 0, 0.001, 0.25, 0, 0.001] # J1 PIDPIDPID
    gainVec1 = [30, 0, 0, 0.25, 0, 0.001, 0.25, 0, 0.001] # J2 PIDPIDPID
    gainVec2 = [30, 0, 0, 0.25, 0, 0.001, 0.25, 0, 0.001] # J3 PIDPIDPID
    setGains(gainVec0, gainVec1, gainVec2)

## Set up logging
feedback = hebi.GroupFeedback(group.size)
logfileName = getLogfileName(model, trajectoryidx, useOptimisedParams)
group.start_log(logfileName, mkdirs=True)

## Perform trajectory
execute_trajectory(group, trajectory, feedback)

## Stop Logging and visualise
log_file = group.stop_log()

if log_file is not None:
    hebi.util.plot_logs(log_file, 'position', figure_spec=101)
    #hebi.util.plot_logs(log_file, 'velocity', figure_spec=102)
    #hebi.util.plot_logs(log_file, 'effort', figure_spec=103)


