#!/bin/bash

docker_build () {
    python_version=$1
    chainer_version=$2
    chainercv_version=$3
    docker_tag=python${python_version}-chainer${chainer_version}

    docker pull mitmul/onnx-chainer:${docker_tag}

    docker build -t mitmul/onnx-chainer:${docker_tag} \
    --build-arg PYTHON_VERSION=${python_version} \
    --build-arg CHAINER_VERSION=${chainer_version} \
    --build-arg CHAINERCV_VERSION=${chainercv_version} .

    if [ $(id -n -u) = "mitmul" ]; then
        docker push mitmul/onnx-chainer:${docker_tag}
    fi
}

docker_build 3.5 5.0.0 0.11.0
docker_build 3.6 5.0.0 0.11.0

docker_build 3.5 6.0.0a1 0.11.0
docker_build 3.6 6.0.0a1 0.11.0
