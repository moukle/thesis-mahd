#! /bin/sh

docker run \
 	-it --rm 
	-v $PWD:/tmp/dev \
 	nvcr.io/nvidia/tensorrt:20.01-py3

# run:
# $ /opt/tensorrt/python/python_setup.sh
# $ pip3.6 install tensorflow=2
# $ python3.6 /tmp/dev/clean/tensorRT.py