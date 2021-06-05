#!/bin/bash

docker run \
	-it --rm --gpus all \
	--name mobilenetv3-train \
	--ipc=host \
	--net=host \
	-v /home/geovision/ILSVRC2012:/dataset \
	-v $(pwd):/workspace \
	nvcr.io/nvidia/pytorch:21.05-py3
