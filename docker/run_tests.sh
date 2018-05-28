#!/bin/bash

docker run --rm -v $PWD:/root/onnx-chainer \
-ti mitmul/onnx-chainer:python2.7 \
bash -c "cd /root/onnx-chainer && python setup.py develop && py.test -vvvs -m 'not slow' tests" && bash
