This repository contains the source code and documentation of the smart-control application. This application aims to allow for deployment of Reinforcement Learning models for robot actuator control and consists of a [Docker](https://www.docker.com/)-based build system, a physics simulation based on the [Bullet](https://pybullet.org/wordpress/) simulator and a custom environment for the Reinforcement Learning framework [OpenAI Gym](https://gym.openai.com/). All components are integrated into a [Robot Operating System (ROS)](https://www.ros.org/) package.

## Documentation

The documentation for the software is provided in HTML form, with the mainpage being *documentation.html*. From there, all the relevant sources and documentation is linked.

Additionally, *documentation.html* provides a quickstart guide to get the application up and running. The build system has been uploaded to a separate Github repository. See [this Github repository](https://github.com/SimonSchwaiger/ros-ml-container) for a detailled explanation of the build system.

## Repository Structure

- **/app** contains the notebooks documenting performed tests
- **/src** contains the implemented jointcontrol ROS package
- **/ros-ml-container** contains Docker build files and is the location, where the application will be built
- **/htmldoc** contains generated documentation for the repository