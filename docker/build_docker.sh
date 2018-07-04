#!/bin/bash

docker_build () {
    python_version=$1
    mxnet_version=$2
    tvm_commit_id=$3
    chainer_version=$4
    chainercv_version=$5
    docker_tag=python${python_version}-chainer${chainer_version}

    docker build -t mitmul/onnx-chainer:${docker_tag} \
    --build-arg PYTHON_VERSION=${python_version} \
    --build-arg MXNET_VERSION=${mxnet_version} \
    --build-arg TVM_COMMIT_ID=${tvm_commit_id} \
    --build-arg CHAINER_VERSION=${chainer_version} \
    --build-arg CHAINERCV_VERSION=${chainercv_version} .
}

docker_build 2.7 1.2.0 ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc 3.5.0 0.10.0
docker_build 2.7 1.2.0 ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc 3.5.0 0.10.0
docker_build 2.7 1.2.0 ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc 3.5.0 0.10.0

docker_build 3.5 1.2.0 ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc 4.2.0 0.10.0
docker_build 3.5 1.2.0 ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc 4.2.0 0.10.0
docker_build 3.5 1.2.0 ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc 4.2.0 0.10.0

docker_build 3.6 1.2.0 ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc 4.2.0 0.10.0
docker_build 3.6 1.2.0 ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc 4.2.0 0.10.0
docker_build 3.6 1.2.0 ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc 4.2.0 0.10.0
