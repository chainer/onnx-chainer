#!/bin/bash

docker run --rm -v $PWD:/root/onnx-chainer \
-ti mitmul/onnx-chainer:py36 \
bash -c "pip install mxnet==1.2.0 && cd /root/onnx-chainer && python setup.py develop && py.test -x -s -vvvs -m 'not slow' $1"
