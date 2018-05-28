#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import onnx
import chainer
import chainer.functions as F
import chainer.links as L
import onnx_chainer


class SmallCNN(chainer.Chain):

    def __init__(self):
        super(SmallCNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 10, 5)
            self.conv2 = L.Convolution2D(10, 20, 5)
            self.fc1 = L.Linear(None, 50)
            self.fc2 = L.Linear(50, 10, nobias=True)

    def __call__(self, x):
        x = F.relu(F.max_pooling_2d(self.conv1(x), 2))
        x = F.relu(F.max_pooling_2d(F.dropout(self.conv2(x)), 2))
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return x


model = SmallCNN()

# Pseudo input
x = np.zeros((1, 1, 28, 28), dtype=np.float32)

# Don't forget to set train flag off!
chainer.config.train = False

onnx_model = onnx_chainer.export(model, x, filename='test.onnx')

onnx_model = onnx.load('test.onnx')

# from caffe2.python.onnx.backend import Caffe2Backend
# from caffe2.python.onnx.backend import run_model

# print(dir(onnx_model))
# run_model(onnx_model, [x])
