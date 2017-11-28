import unittest

import chainer
import chainer.functions as F
import numpy as np

from chainer import testing
import onnx_chainer


@testing.parameterize(
    {'ops': 'cast', 'input_shape': (1, 5),
     'args': {'typ': np.float16}},
    {'ops': 'cast', 'input_shape': (1, 5),
     'args': {'typ': np.float64}},

    {'ops': 'depth2space', 'input_shape': (1, 12, 6, 6),
     'args': {'r': 2}},

    {'ops': 'pad', 'input_shape': (1, 5),
     'args': {'pad_width': (0, 2), 'mode': 'constant'}},
    {'ops': 'pad', 'input_shape': (1, 5),
     'args': {'pad_width': (0, 2), 'mode': 'reflect'}},
    {'ops': 'pad', 'input_shape': (1, 5),
     'args': {'pad_width': (0, 2), 'mode': 'edge'}},

    {'ops': 'reshape', 'input_shape': (1, 6), 'args': {'shape': (1, 2, 1, 3)}},
    {'ops': 'reshape', 'input_shape': (1, 6), 'args': {'shape': (1, 2, 1, 3)}},

    {'ops': 'split_axis', 'input_shape': (1, 6),
     'args': {'indices_or_sections': 2, 'axis': 1, 'force_tuple': True}},
    {'ops': 'split_axis', 'input_shape': (1, 6),
     'args': {'indices_or_sections': 2, 'axis': 1, 'force_tuple': False}},

    {'ops': 'squeeze', 'input_shape': (1, 3, 1, 2),
     'args': {'axis': None}},
    {'ops': 'squeeze', 'input_shape': (1, 3, 1, 2, 1),
     'args': {'axis': (2, 4)}},

    {'ops': 'tile', 'input_shape': (1, 5), 'args': {'reps': (1, 2)}},

    {'ops': 'transpose', 'input_shape': (1, 5), 'args': {'axes': None}},
)
class TestArrayOperators(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args):
                super(Model, self).__init__()
                self.ops = getattr(F, ops)
                self.args = list(args.values())

            def __call__(self, x):
                return self.ops(*([x] + self.args))

        self.model = Model(self.ops, self.args)
        self.x = np.zeros(self.input_shape, dtype=np.float32)

    def test_export_test(self):
        chainer.config.train = False
        onnx_chainer.export(self.model, self.x)

    def test_export_train(self):
        chainer.config.train = True
        onnx_chainer.export(self.model, self.x)


class TestConcat(unittest.TestCase):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()

            def __call__(self, x):
                y = chainer.Variable(np.ones(x.shape).astype(x.dtype))
                return F.concat((x, y))

        self.model = Model()
        self.x = np.zeros((1, 5), dtype=np.float32)

    def test_export_test(self):
        chainer.config.train = False
        onnx_chainer.export(self.model, self.x)

    def test_export_train(self):
        chainer.config.train = True
        onnx_chainer.export(self.model, self.x)
