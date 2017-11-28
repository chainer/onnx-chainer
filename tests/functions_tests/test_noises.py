import unittest

import chainer
from chainer import testing
import chainer.functions as F
import numpy as np

import onnx_chainer


@testing.parameterize(
    {'ops': F.dropout, 'args': {'ratio': 0.5}},
)
class TestNoises(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args):
                super(Model, self).__init__()
                self.ops = ops
                self.args = list(args.values())

            def __call__(self, x):
                x = F.identity(x)
                return self.ops(*([x] + self.args))

        self.model = Model(self.ops, self.args)
        self.x = np.zeros((1, 5), dtype=np.float32)

    def test_export_test(self):
        chainer.config.train = False
        onnx_chainer.export(self.model, self.x)

    def test_export_train(self):
        chainer.config.train = True
        onnx_chainer.export(self.model, self.x)
