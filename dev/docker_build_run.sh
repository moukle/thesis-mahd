#! /bin/sh

# build docker
docker build -t mahd/tf_cv .

# open XSOCK for diplay
XSOCK=/tmp/.X11-unix
xhost +

# run container
docker run \
	--rm --gpus 1 -it \
	-v $PWD:/tmp/dev \
	-v ~/data:/tmp/data \
	-v $PWD:/tf/notebooks \
	-v $XSOCK:$XSOCK -e DISPLAY=$DISPLAY \
	-w /tmp/ \
	-p 0.0.0.0:6006:6006 \
	-p 0.0.0.0:8888:8888 \
	mahd/tf_cv

xhost -
