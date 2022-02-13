#!/bin/bash
# Copy all Notebooks from ros-ml-container/app/Notebooks to /app
cd ros-ml-container/app/Notebooks
for i in *.ipynb; do
    [ -f "$i" ] || break
    cp $i ../../../app/Notebooks
done
cd ../../..

# Copy generated documentation from ros-ml-container/app to .
cp -r ros-ml-container/app/htmldoc .
cp ros-ml-container/app/documentation.html .
