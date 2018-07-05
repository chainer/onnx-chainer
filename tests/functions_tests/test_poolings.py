import unittest

import numpy as np

import chainer
from chainer import testing
import chainer.functions as F
import onnx_chainer
from onnx_chainer.testing import test_mxnet

MXNET_SUPPORT = {
    'average_pooling_2d': True,
    'average_pooling_nd': False,
    'max_pooling_2d': True,
    'max_pooling_nd': False,
}


@testing.parameterize(
    {'name': 'average_pooling_2d', 'ops': F.average_pooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [2, 1, 0]},
    {'name': 'average_pooling_nd', 'ops': F.average_pooling_nd,
     'in_shape': (1, 3, 6, 6, 6), 'args': [2, 1, 0]},
    {'name': 'max_pooling_2d', 'ops': F.max_pooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [2, 1, 0]},
    {'name': 'max_pooling_nd', 'ops': F.max_pooling_nd,
     'in_shape': (1, 3, 6, 6, 6), 'args': [2, 1, 0]},
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

    def test_compatibility(self):
        if MXNET_SUPPORT[self.name]:
            test_mxnet.check_compatibility(self.model, self.x, self.fn)
        else:
            onnx_chainer.export(self.model, self.x)
