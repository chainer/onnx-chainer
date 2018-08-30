import unittest

import chainer
from chainer import testing
import chainer.functions as F
import chainer.links as L
import numpy as np
import onnx
import onnx_chainer
from onnx_chainer.testing import test_mxnet


@testing.parameterize(
    {
        'name': 'local_response_normalization',
        'input_argname': 'x',
        'args': {'k': 1, 'n': 3, 'alpha': 1e-4, 'beta': 0.75},
        'opset_version': 1
    },
)
class TestNormalizations(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args, input_argname):
                super(Model, self).__init__()
                self.ops = ops
                self.args = args
                self.input_argname = input_argname

            def __call__(self, x):
                self.args[self.input_argname] = x
                return self.ops(**self.args)

        ops = getattr(F, self.name)
        self.model = Model(ops, self.args, self.input_argname)
        self.x = np.zeros((1, 5, 3, 3), dtype=np.float32)
        self.fn = self.name + '.onnx'

    def test_compatibility(self):
        test_mxnet.check_compatibility(
            self.model, self.x, self.fn, opset_version=self.opset_version)
        for opset_version in range(1, onnx.defs.onnx_opset_version() + 1):
            onnx_chainer.export(
                self.model, self.x, opset_version=self.opset_version)


@testing.parameterize(
    {'opset_version': 1},
    {'opset_version': 6},
    {'opset_version': 7},
)
class TestBatchNormalization(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.bn = L.BatchNormalization(5)

            def __call__(self, x):
                return self.bn(x)

        self.model = Model()
        self.x = np.zeros((1, 5), dtype=np.float32)
        self.fn = 'BatchNormalization.onnx'

    def test_compatibility(self):
        test_mxnet.check_compatibility(
            self.model, self.x, self.fn, opset_version=self.opset_version)
        for opset_version in range(1, onnx.defs.onnx_opset_version() + 1):
            onnx_chainer.export(
                self.model, self.x, opset_version=opset_version)
