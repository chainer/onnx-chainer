#!/bin/bash

docker run --rm \
-v $PWD:/root/onnx-chainer \
-v $PWD/.chainer:/root/.chainer \
-ti mitmul/onnx-chainer:py36 \
bash -c "cd /root/onnx-chainer && python setup.py develop && py.test -x -s -vvvs -m 'not slow' $1"
