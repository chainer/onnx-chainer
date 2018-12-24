import unittest

import numpy as np

import chainer
import chainer.functions as F
import onnx
import onnx_chainer
from chainer import testing
from onnx_chainer.testing import test_onnxruntime


@testing.parameterize(
    {'name': 'average_pooling_2d', 'ops': F.average_pooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [2, 1, 0]},
    {'name': 'average_pooling_2d', 'ops': F.average_pooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [3, 2, 1]},
    {'name': 'average_pooling_nd', 'ops': F.average_pooling_nd,
     'in_shape': (1, 3, 6, 6, 6), 'args': [2, 1, 1]},
    {'name': 'max_pooling_2d', 'ops': F.max_pooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [2, 1, 1]},
    {'name': 'max_pooling_nd', 'ops': F.max_pooling_nd,
     'in_shape': (1, 3, 6, 6, 6), 'args': [2, 1, 1]},
)
class TestPoolings(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args):
                super(Model, self).__init__()
                self.ops = ops
                self.args = args

            def __call__(self, x):
                return self.ops(*([x] + self.args))

        self.model = Model(self.ops, self.args)
        self.x = np.ones(self.in_shape, dtype=np.float32)
        self.fn = self.name + '.onnx'

    def test_output(self):
        for opset_version in range(
                test_onnxruntime.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            test_onnxruntime.check_output(
                self.model, self.x, self.fn, opset_version=opset_version)
