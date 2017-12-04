#!/bin/bash

docker run --rm -v $PWD/../:/root/onnx-chainer \
mitmul/onnx-chainer:latest \
bash -c "cd /root/onnx-chainer && pip install -e . && py.test -vvvs tests"