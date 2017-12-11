#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import nnvm
import onnx_chainer

# model = L.VGG16Layers()


class MLP(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, 10)

    def __call__(self, x):
        return F.relu(self.l1(x))


model = MLP()
x = np.random.rand(1, 10).astype(np.float32)

chainer.config.train = False
model_onnx = onnx_chainer.export(model, x)
sym, params = nnvm.frontend.from_onnx(model_onnx.graph)
