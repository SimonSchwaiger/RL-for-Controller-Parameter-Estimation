#!/bin/bash
## Generates complete HTML documentation
# Folder structure for documentation
# /:
# - documentation.html
# /doc:r

# - Notebooks.html
# /doc/jointcontrol/html/index.html

# Create folder containing html doc
mkdir /app/htmldoc

# Generate Rosdoc Doxygen documentation
rosdoc_lite -o /app/htmldoc/jointcontrol /catkin_ws/src/jointcontrol/

# Change into notebook dir for notebook generation
cd /app/Notebooks
# Copy nbconvert template to the right directory
# Due to a dependency error in the nbconvert inheritance system, we need to replace the exisiting lab template
rm -rf /root/myenv/share/jupyter/nbconvert/templates/lab/static/index.css
cp -r index.css /root/myenv/share/jupyter/nbconvert/templates//lab/static/

# Generate HTML Notebooks from ipynb
jupyter nbconvert --execute --to html --template lab testControllerBlocks.ipynb
jupyter nbconvert --to html --template lab DDPG.ipynb
jupyter nbconvert --execute --to html --template lab documentation.ipynb

# Move all files to the right destinations
mv testControllerBlocks.html /app/htmldoc
mv DDPG.html /app/htmldoc
mv documentation.html /app

cd /app