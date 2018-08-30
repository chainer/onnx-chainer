#!/bin/bash

docker_build () {
    python_version=$1
    mxnet_version=$2
    tvm_version=$3
    chainer_version=$4
    chainercv_version=$5
    docker_tag=python${python_version}-chainer${chainer_version}

    docker pull mitmul/onnx-chainer:${docker_tag}

    docker build -t mitmul/onnx-chainer:${docker_tag} \
    --build-arg PYTHON_VERSION=${python_version} \
    --build-arg MXNET_VERSION=${mxnet_version} \
    --build-arg TVM_VERSION=${tvm_version} \
    --build-arg CHAINER_VERSION=${chainer_version} \
    --build-arg CHAINERCV_VERSION=${chainercv_version} .

    docker push mitmul/onnx-chainer:${docker_tag}
}

docker_build 2.7 pre 0.4 3.5.0 0.10.0
# docker_build 2.7 pre 0.4 4.4.0 0.10.0
# docker_build 3.5 pre 0.4 3.5.0 0.10.0
# docker_build 3.5 pre 0.4 4.4.0 0.10.0
# docker_build 3.6 pre 0.4 3.5.0 0.10.0
# docker_build 3.6 pre 0.4 4.4.0 0.10.0
