# Enable GUI on docker
XAUTH="${TMPDIR:-"/tmp"}/xauth_docker_${REPOSITORY_NAME}"
touch "${XAUTH}"
chmod a+r "${XAUTH}"
XAUTH_LIST=$(xauth nlist "${DISPLAY}")
if [ -n "${XAUTH_LIST}" ]; then
    echo "${XAUTH_LIST}" | sed -e 's/^..../ffff/' | xauth -f "${XAUTH}" nmerge -
fi

# Build container to run OIGE
docker run --name isaac-sim-oige --entrypoint bash -it -d --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
-e DISPLAY=${DISPLAY} \
-e XAUTHORITY=${XAUTH} \
-v ${XAUTH}:${XAUTH} \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /dev/input:/dev/input \
-v ${PWD}:/workspace/omniisaacgymenvs \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
nvcr.io/nvidia/isaac-sim:2022.2.1

# install OIGE and rl-games
# assumes rl_games repository is also inside $PWD
docker exec -it isaac-sim-oige sh -c "cd /workspace/omniisaacgymenvs && /isaac-sim/python.sh -m pip install --upgrade pip && /isaac-sim/python.sh -m pip install -e . && cd rl_games && /isaac-sim/python.sh -m pip install -e ."
# get inside docker shell
docker exec -it -w /workspace/omniisaacgymenvs/omniisaacgymenvs isaac-sim-oige bash
