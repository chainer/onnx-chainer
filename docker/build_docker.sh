#!/bin/bash

docker build -t mitmul/onnx-chainer:py27 --build-arg PYTHON_VERSION=2.7 .
docker build -t mitmul/onnx-chainer:py35 --build-arg PYTHON_VERSION=3.5 .
docker build -t mitmul/onnx-chainer:py36 --build-arg PYTHON_VERSION=3.6 .
