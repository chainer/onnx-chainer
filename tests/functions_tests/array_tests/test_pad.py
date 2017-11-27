import unittest

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing
import numpy as np

import onnx_chainer


class Model(chainer.Chain):

    def __init__(self, mode):
        super(Model, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(5, 5)
        self.mode = mode

    def __call__(self, x):
        return F.pad(self.l1(x), (0, 2), self.mode)


@testing.parameterize(
    {'mode': 'constant'},
    {'mode': 'reflect'},
    {'mode': 'edge'},
)
class TestExport(unittest.TestCase):

    def setUp(self):
        self.model = Model(self.mode)
        self.x = np.zeros((1, 5), dtype=np.float32)

    def test_export_test(self):
        chainer.config.train = False
        onnx_chainer.export(self.model, self.x)

    def test_export_train(self):
        chainer.config.train = True
        onnx_chainer.export(self.model, self.x)
