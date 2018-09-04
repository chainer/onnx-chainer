import unittest

import numpy as np

import chainer
import chainer.functions as F
import onnx
import onnx_chainer
from chainer import testing
from onnx_chainer.testing import test_mxnet


@testing.parameterize(
    {'name': 'dropout', 'ops': lambda x: F.dropout(x, ratio=0.5)},
)
class TestNoises(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, x):
                with chainer.using_config('train', True):
                    y = self.ops(x)
                return y

        self.model = Model(self.ops)
        self.x = np.zeros((1, 5), dtype=np.float32)
        self.fn = self.name + '.onnx'

    def test_compatibility(self):
        test_mxnet.check_compatibility(self.model, self.x, self.fn)
        for opset_version in range(1, onnx.defs.onnx_opset_version() + 1):
            onnx_chainer.export(self.model, self.x,
                                opset_version=opset_version)
