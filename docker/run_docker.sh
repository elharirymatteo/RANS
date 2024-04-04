#!/bin/bash

xhost +
docker run --name isaac-sim-rans-container -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
-v $HOME/.Xauthority:/root/.Xauthority \
-e DISPLAY \
-e "PRIVACY_CONSENT=Y" \
-v ${PWD}:/workspace/RANS \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
-v ~/.netrc:/root/.netrc \
isaac-sim-rans:latest
