This repository contains the source code and documentation of my master thesis with the title **Reinforcement Learning for Controller Parameter Estimation**. The goal of the thesis was to evaluate use of Reinforcement Learning models, to perform adaptive closed loop control by determining controller parameters of an interlinked controller design during operation.

The application consists of a [Docker](https://www.docker.com/) image that configures the application's environment as well as GPU acceleration, a physics simulation based on the [Bullet](https://pybullet.org/wordpress/) simulator and a custom environment for the Reinforcement Learning framework [OpenAI Gym](https://gym.openai.com/). Instances of the Gym environment connect to a central physics simulation over shared memory in order to allow them to interact with each other during agent training.

The gym environment and physics server are implemented as part of a [Robot Operating System (ROS)](https://www.ros.org/) package in */src*, since they use the ROS parameter server for configuration of environment instances.

## Repository Structure

This is a list of important directories in this repository.

- **/app/MTExperiments**: Contains conducted experiments as part of the thesis. *TrainingScripts* contains blueprints for control signal generation, model training and model evaluation, while *Plots* contains the implementation of training and result visualisation for each conducted experiment.
- **/src/jointcontrol**: Contains the implemented ROS package including the gym environment and robot description. *scripts* contains implementations of the controller model, discrete environment wrapper, physics server (instantiates and synchronises physics simulation) and the shared memory implementation.
- **/ros-ml-container**: Contains Docker build files and is the location, where the application will be built
- **/documentation**: Contains generated documentation for the repository
- **/app/Notebooks**: Contains the notebooks documenting preliminary testing (using an old stable-baselines version)

## Documentation

The documentation for the software is provided in HTML form, with the mainpage being *documentation/documentation.html*. From there, all the relevant sources and documentation are linked.
Additionally, *documentation.html* provides a quickstart guide to get the application up and running. The Docker image is documented in a standalone fashion in a separate [Github repository](https://github.com/SimonSchwaiger/ros-ml-container).

## Compatibility

As long as a means of OpenGL is provided for the Bullet simulator, the application should be able to be started using Docker. The application was tested on the following hardware and software configuration:

- **CPU:** Intel Core i7 12700
- **GPU:** Radeon RX 6700 XT
- **Host OS:** Ubuntu 20.04 LTS
- **GPU Driver:** Proprietary amdgpu Driver Version 21.10
- **Docker Version:** 20.10.16