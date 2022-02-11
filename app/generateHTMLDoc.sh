#!/bin/bash
## Generates complete HTML documentation
# Folder structure for documentation
# /:
# - documentation.html
# /doc:
# - Notebooks.html
# /doc/jointcontrol/html/index.html

# Create folder containing html doc
mkdir /app/htmldoc

# Generate Rosdoc Doxygen documentation
rosdoc_lite -o /app/htmldoc/jointcontrol /catkin_ws/src/jointcontrol/

# Generate HTML Notebooks from ipynb
cd /app/Notebooks
jupyter nbconvert --execute --to html --no-input testControllerBlocks.ipynb
jupyter nbconvert --to html DDPG.ipynb
jupyter nbconvert --execute --to html documentation.ipynb

# Move all files to the right destinations
mv testControllerBlocks.html /app/htmldoc
mv DDPG.html /app/htmldoc
mv documentation.html /app

cd /app