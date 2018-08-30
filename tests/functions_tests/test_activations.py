import unittest

import chainer
from chainer import testing
import chainer.functions as F
import chainer.links as L
import numpy as np
import onnx
import onnx_chainer
from onnx_chainer.testing import test_mxnet


MXNET_OPSET_VERSION = {
    'elu': (1, 6),
    'hard_sigmoid': (6,),
    'leaky_relu': (6,),
    'log_softmax': (1,),
    'relu': (1, 6),
    'sigmoid': (1, 6),
    'softmax': (1,),
    'softplus': (1,),
    'tanh': (1, 6),
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
        if MXNET_OPSET_VERSION[self.name] is not None:
            for mxnet_opset_version in MXNET_OPSET_VERSION[self.name]:
                test_mxnet.check_compatibility(
                    self.model, self.x, self.fn, opset_version=mxnet_opset_version)
        for opset_version in range(1, onnx.defs.onnx_opset_version() + 1):
            onnx_chainer.export(
                self.model, self.x, opset_version=opset_version)


@testing.parameterize(
    {'opset_version': 6},
    {'opset_version': 7},
)
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
        test_mxnet.check_compatibility(
            self.model, self.x, self.fn, opset_version=self.opset_version)

        onnx_chainer.export(
            self.model, self.x, opset_version=self.opset_version)
