#!/bin/bash

docker run -v $PWD:/root/ \
onnx-chainer \
bash -c "cd /root/ && pip install git+git://github.com/onnx/onnx-caffe2@7aaf817ecf1c4ec34d881633bc494f7f00d2d6b5 && pip install -e . && py.test ."