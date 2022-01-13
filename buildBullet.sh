#!/bin/bash

# Download SDK release
mkdir /bullet && cd /bullet
wget https://github.com/bulletphysics/bullet3/archive/refs/tags/3.21.tar.gz

# Untar and change into directory
tar -xvzf 3.21.tar.gz
cd bullet3-3.21

# Execute build script
chmod +x build_cmake_pybullet_double.sh
./build_cmake_pybullet_double