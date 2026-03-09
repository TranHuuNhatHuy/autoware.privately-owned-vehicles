#!/bin/bash

# Run the container
docker run -it --rm \
    -p 6080:6080 \
    -v "$PWD"/model-weights:/autoware/model-weights \
    -v "$PWD"/launch:/autoware/launch \
    -v "$PWD"/../Test:/autoware/test \
    visionpilot \
    /autoware/launch/run_objectFinder.sh
