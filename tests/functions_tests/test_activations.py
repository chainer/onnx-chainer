import unittest

import numpy as np

import chainer
from chainer import testing
import chainer.functions as F
import chainer.links as L
import onnx_chainer
from onnx_chainer.testing import test_mxnet

MXNET_SUPPORT = {
    'elu': True,
    'hard_sigmoid': False,
    'leaky_relu': True,
    'log_softmax': False,
    'relu': True,
    'sigmoid': True,
    'softmax': True,
    'softplus': False,
    'tanh': True,
}


@testing.parameterize(
    {'name': 'elu'},
    {'name': 'hard_sigmoid'},
    {'name': 'leaky_relu'},
    {'name': 'log_softmax'},
    {'name': 'relu'},
    {'name': 'sigmoid'},
    {'name': 'softmax'},
    {'name': 'softplus'},
    {'name': 'tanh'},
)
class TestActivations(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, x):
                return self.ops(x)

        ops = getattr(F, self.name)
        self.model = Model(ops)
        self.x = np.random.randn(1, 5).astype(np.float32)
        self.fn = self.name + '.onnx'

    def test_compatibility(self):
        if MXNET_SUPPORT[self.name]:
            test_mxnet.check_compatibility(self.model, self.x, self.fn)
        else:
            onnx_chainer.export(self.model, self.x)


class TestPReLU(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.prelu = L.PReLU()

            def __call__(self, x):
                return self.prelu(x)

        self.model = Model()
        self.x = np.zeros((1, 5), dtype=np.float32)
        self.fn = 'PReLU.onnx'

    def test_compatibility(self):
        test_mxnet.check_compatibility(self.model, self.x, self.fn)
