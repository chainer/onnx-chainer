#!/bin/bash

docker run --rm -v $PWD:/root/onnx-chainer \
-ti mitmul/onnx-chainer:python2.7 \
bash -c "cd /root/onnx-chainer && python setup.py develop && py.test -x -s -vvvs -m 'not slow' tests/functions_tests/test_arrays.py" && bash
