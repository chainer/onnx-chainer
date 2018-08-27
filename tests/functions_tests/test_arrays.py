import unittest

import numpy as np

import chainer
from chainer import testing
import chainer.functions as F
import onnx_chainer
from onnx_chainer.testing import test_mxnet
from onnx.backend.test.case import model

MXNET_SUPPORT = {
    'cast': True,
    'copy': False,
    'depth2space': False,
    'pad': False,
    'reshape': True,
    'space2depth': False,
    'split_axis': False,
    'squeeze': False,
    'tile': False,
    'transpose': True
}

@testing.parameterize(
    {'ops': 'cast', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'typ': np.float16}},
    {'ops': 'cast', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'typ': np.float64}},

    {'ops': 'depth2space', 'input_shape': (1, 12, 6, 6),
     'input_argname': 'X',
     'args': {'r': 2}},

    {'ops': 'pad', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'pad_width': (0, 2), 'mode': 'constant'}},
    {'ops': 'pad', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'pad_width': (0, 2), 'mode': 'reflect'}},
    {'ops': 'pad', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'pad_width': (0, 2), 'mode': 'edge'}},

    {'ops': 'reshape', 'input_shape': (1, 6),
     'input_argname': 'x',
     'args': {'shape': (1, 2, 1, 3)}},
    {'ops': 'reshape', 'input_shape': (1, 6),
     'input_argname': 'x',
     'args': {'shape': (1, 2, 1, 3)}},

    {'ops': 'space2depth', 'input_shape': (1, 12, 6, 6),
     'input_argname': 'X',
     'args': {'r': 2}},

    {'ops': 'split_axis', 'input_shape': (1, 6),
     'input_argname': 'x',
     'args': {'indices_or_sections': 2,
              'axis': 1, 'force_tuple': True}},
    {'ops': 'split_axis', 'input_shape': (1, 6),
     'input_argname': 'x',
     'args': {'indices_or_sections': 2,
              'axis': 1, 'force_tuple': False}},

    {'ops': 'squeeze', 'input_shape': (1, 3, 1, 2),
     'input_argname': 'x',
     'args': {'axis': None}},
    {'ops': 'squeeze', 'input_shape': (1, 3, 1, 2, 1),
     'input_argname': 'x',
     'args': {'axis': (2, 4)}},

    {'ops': 'tile', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'reps': (1, 2)}},

    {'ops': 'transpose', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'axes': None}},

    {'ops': 'copy', 'input_shape': (1, 5),
     'input_argname': 'x',
     'args': {'dst': -1}},
)
class TestArrayOperators(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args, input_argname):
                super(Model, self).__init__()
                self.ops = getattr(F, ops)
                self.args = args
                self.input_argname = input_argname

            def __call__(self, x):
                self.args[self.input_argname] = x
                return self.ops(**self.args)

        self.model = Model(self.ops, self.args, self.input_argname)
        self.x = np.zeros(self.input_shape, dtype=np.float32)
        self.fn = self.ops + '.onnx'

    def test_compatibility(self):
        if MXNET_SUPPORT[self.ops]:
            test_mxnet.check_compatibility(self.model, self.x, self.fn)
        else:
            onnx_chainer.export(self.model, self.x)


class TestConcat(unittest.TestCase):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()

            def __call__(self, x1, x2):
                return F.concat((x1, x2))

        self.model = Model()
        self.x1 = np.zeros((1, 5), dtype=np.float32)
        self.x2 = np.ones((1, 5), dtype=np.float32)
        self.fn = 'Concat.onnx'

    def test_backend(self):
        y = self.model(self.x1, self.x2)
        onnx_model = onnx_chainer.export(self.model, (self.x1, self.x2))
        model.expect(onnx_model, (self.x1,), y)
