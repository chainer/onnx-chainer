#!/bin/bash

docker run --rm \
-v $PWD:/root/onnx-chainer \
-v $PWD/.chainer:/root/.chainer \
-ti mitmul/onnx-chainer:python2.7-chainer4.4.0 \
bash -c "cd /root/onnx-chainer && python setup.py develop && python examples/mxnet/export.py"
