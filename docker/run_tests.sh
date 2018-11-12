#!/bin/bash

python_version=$1
chainer_version=$2
onnx_runtime_version=0.1.3
tag_name=python${python_version}-chainer${chainer_version}
docker run --rm \
-v $PWD:/root/onnx-chainer \
-v $PWD/.chainer:/root/.chainer \
-ti mitmul/onnx-chainer:${tag_name} \
bash -c "pip install onnxruntime==${onnx_runtime_version} && cd /root/onnx-chainer && python setup.py develop && py.test -x -s -vvvs tests/$3"
