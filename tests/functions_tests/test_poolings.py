import unittest

import numpy as np

import chainer
import chainer.functions as F
import onnx
import onnx_chainer
from chainer import testing
from onnx_chainer.testing import test_mxnet

MXNET_OPSET_VERSION = {
    'average_pooling_2d': (1, 7),
    'average_pooling_nd': None,
    'max_pooling_2d': (1,),
    'max_pooling_nd': None,
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
        if MXNET_OPSET_VERSION[self.name] is not None:
            for mxnet_opset_version in MXNET_OPSET_VERSION[self.name]:
                test_mxnet.check_compatibility(
                    self.model, self.x, self.fn, opset_version=mxnet_opset_version)
        for opset_version in range(1, onnx.defs.onnx_opset_version() + 1):
            onnx_chainer.export(self.model, self.x,
                                opset_version=opset_version)
