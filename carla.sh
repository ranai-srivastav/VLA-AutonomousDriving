docker run \
  --gpus all \
  --net=host \
  --user=$(id -u):$(id -g) \
  --env=DISPLAY=$DISPLAY \
  --env=NVIDIA_VISIBLE_DEVICES=all \
  --env=NVIDIA_DRIVER_CAPABILITIES=all \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  carlasim/carla:0.9.15 /bin/bash -c "./CarlaUE4.sh -nosound"
