#!/bin/bash

docker build -t mitmul/python:2.7 python/2.7
docker push mitmul/python:2.7
docker build -t mitmul/python:3.5 python/3.5
docker push mitmul/python:3.5
docker build -t mitmul/python:3.6 python/3.6
docker push mitmul/python:3.6

docker build -t mitmul/caffe2:python2.7 caffe2/python2.7
docker push mitmul/caffe2:python2.7
docker build -t mitmul/caffe2:python3.5 caffe2/python3.5
docker push mitmul/caffe2:python3.5
docker build -t mitmul/caffe2:python3.6 caffe2/python3.6
docker push mitmul/caffe2:python3.6

docker build -t mitmul/onnx-chainer:python2.7 onnx-chainer/python2.7
docker push mitmul/onnx-chainer:python2.7
docker build -t mitmul/onnx-chainer:python3.5 onnx-chainer/python3.5
docker push mitmul/onnx-chainer:python3.5
docker build -t mitmul/onnx-chainer:python3.6 onnx-chainer/python3.6
docker push mitmul/onnx-chainer:python3.6
