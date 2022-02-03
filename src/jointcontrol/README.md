This package contains the source code for the *jointcontrol* ROS package. The goal of this package, is to learn the dynamics of a complex closed loop controller using a Reinforcement Learning (RL) model. This model is trained in simulation on an approximated reproduction of the real controller with the intent of deploying the fully trained model to the real controller. While physics-based training is done using [the Pybullet simulator](https://pybullet.org/wordpress/), the standardised interface between environment and RL agent defined by [OpenAI Gym](https://gym.openai.com/) was used by implementing a custom Gym environment that connects the Gym and Bullet components to each other using the [Robot Operating System (ROS)](https://www.ros.org/).

This package allow for instantiation of an arbitrary amount of Gym environments, that all connect to a singe instance of the Bullet simulator. This allows for multiple agents to be trained in parallel. Simulation steps are performed synchronised between all instances in order to ensure accurate simulation.

# Important Files

These links can be used to quickly navigate to important files. The files are explained in the following sections.

- parseConfig.py
- physicsServer.py
- jointcontrol_env.py

## Config

The shared simulation environment in Bullet is created based on onfiguration files in the *./config* directory of the ROS package. Upon application start, the json config file is parsed by the parseConfig.py script, which loads all parameters into the ROS param server. The params are loaded as needed by the Gym environments as well as the simulation instance.

The configuration files are divided into two sections, one configuring controller parameters for each joint and one configuring the shared simulation itself. The parameters are explained below, for a full example of a complete config file, see testbenchConfig.json .

### Simulation Params

Simulation parameters are in the namespace /jointcontrol/SimParams/.

| Param                 |                       Example Value | Explanation |
| --------------------- | -----------------------------------:| ----------- |
| JointLinearDamping    |                              [0.04] | Linear damping for simulated joints (vector of len(JointsPerRobot))
| JointAngularDamping   |                                [70] | Angular damping for simulated joints (vector of len(JointsPerRobot))
| JointTorques          |                                [17] | Max torque a joint's actuator is able to provide (vector of len(JointsPerRobot))
| Ts                    |                                0.01 | Time discretisation in seconds (Set to match motors)
| NumRobots             |                                   9 | Number of simulated robots
| JointsPerRobot        |                                   1 | Number of joints per robot
| IdsWithinRobot        |                                 [2] | IDs of movable joints within each robot (vector of len(JointsPerRobot))
| Realtime              |                             "False" | Activates/Deactivates sleep statements to perform simulation in real time
| StationSize           |                              [1, 1] | Spacing between multiple instances of robots in bullet ([x,y], both in metres)
| StationsPerCol        |                                   3 | Determines the amount of robots per column in simulation (purely visual)
| URDFName              |            "TrainingTestbench.urdf" | Robot URDF name
| URDFPath              |  "/catkin_ws/src/jointcontrol/urdf" | Global path to URDF file

### Joint Controller Params

Controller params are set in the namespace /jointcontrol/J{}/. These params are used by the Gym environment with the same index to simulate physical properties of controller as well as plant and set controller bounds.

| Param                 |                              Example Value | Explanation |
| --------------------- | ------------------------------------------:| ----------- |
| NumParams             |                                          9 | Number of configurable parameters for this joint
| Defaults              |            [10, 0, 0, 0.05, 0, 0, 0, 0, 0] | Default parameters for this joint (vector of len(NumParams))
| Minimums              |                [0, 0, 0, 0, 0, 0, 0, 0, 0] | Minimum for each parameter that can be set (vector of len(NumParams))
| Maximums              |              [40, 1, 1, 10, 1, 1, 5, 1, 1] | Maximum for each parameter that can be set (vector of len(NumParams))
| MaxChange             | [10, 0.1, 0.1, 2.5, 0.1, 0.1, 1, 0.1, 0.1] | Maximum allowed change for each param per step (vector of len(NumParams))
| ParamDescription      |                              "PID PID PID" | Description what each parameter represents (for convenience, not used)


## Physics Server

The shared simulation is instantiated by the physicsServer.py script in *./scripts*. This script loads the configuration from the ROS param server, loads in robot models, manages synchronisation between Gym environments and tracks which RobotID, JointID pair in bullet corresponds to which Gym environment.

## Gym Environment

The Gym environment is implemented by jointcontrol_env.py in *./gym-jointcontrol/gym_jointcontrol/envs/*. The controller whose dynamics should be learned by the agend, is modelled after the [Strategy 4 controller](https://docs.hebi.us/core_concepts.html#control-strategies) found in the [Hebi X-series actuators](https://docs.hebi.us/core_concepts.html#core-modules). The environment implements a normal [GymEnv](https://github.com/openai/gym/blob/master/gym/core.py) and provides the *compute_reward* method, in a similar manner to [GoalEnv](https://github.com/openai/gym/blob/3394e245727c1ae6851b504a50ba77c73cd4c65b/gym/core.py#L160). Additionally, the last step can be visualised in matplotlib using the *visualiseTrajectory* method. For detailled info about implemented methods, refer to the generated class reference of jointcontrol_env.py .

