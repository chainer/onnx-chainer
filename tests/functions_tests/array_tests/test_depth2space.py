import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import onnx_chainer


class Model(chainer.Chain):

    def __init__(self):
        super(Model, self).__init__()
        with self.init_scope():
            self.l1 = L.Convolution2D(None, 12, 3, 1, 1)

    def __call__(self, x):
        return F.depth2space(self.l1(x), 2)


class TestExport(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        self.x = np.zeros((1, 3, 6, 6), dtype=np.float32)

    def test_export_test(self):
        chainer.config.train = False
        onnx_chainer.export(self.model, self.x)

    def test_export_train(self):
        chainer.config.train = True
        onnx_chainer.export(self.model, self.x)
