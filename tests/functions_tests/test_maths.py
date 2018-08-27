import unittest

import numpy as np

import chainer
from chainer import testing
import onnx_chainer
from onnx_chainer.testing import test_mxnet

MXNET_SUPPORT = {
    'Neg': True,
    'Absolute': True,
    'Clip': False,
    'Exp': True,
    'Sqrt': True,
    'PowVarConst': True,
    'Sum': True,
    'Add': True,
    'AddConst': True,
    'Sub': True,
    'Mul': True,
    'Div': True,
    'MatMul': True,
    'Maximum': True,
    'Minimum': True,
}


@testing.parameterize(
    {'info': 'Neg', 'ops': '-a'},
    {'info': 'Absolute', 'ops': 'abs(a)'},
    {'info': 'Clip', 'ops': 'chainer.functions.clip(a, 0.1, 0.2)'},
    {'info': 'Exp', 'ops': 'chainer.functions.exp(a)'},
    {'info': 'Sqrt', 'ops': 'chainer.functions.sqrt(a)'},
    {'info': 'PowVarConst',
     'ops': 'chainer.functions.math.basic_math.pow(a, 2)'},
    {'info': 'Sum',
     'ops': 'chainer.functions.sum(a, axis=1)'},
    {'info': 'Sum',
     'ops': 'chainer.functions.sum(a, axis=0, keepdims=True)'},
    {'info': 'AddConst', 'ops': 'a + 1'},
)
class TestUnaryMathOperators(unittest.TestCase):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a):
                if not isinstance(a, chainer.Variable):
                    a = chainer.Varaible(a)
                return eval(self.ops)

        self.model = Model(self.ops)
        self.a = chainer.Variable(np.ones((2, 3), dtype=np.float32))
        self.fn = self.info + '.onnx'

    def test_compatibility(self):
        if MXNET_SUPPORT[self.info]:
            test_mxnet.check_compatibility(self.model, self.a, self.fn)
        else:
            onnx_chainer.export(self.model, self.a)


@testing.parameterize(
    {'info': 'Add', 'ops': 'a + b'},
    {'info': 'Sub', 'ops': 'a - b'},
    {'info': 'Mul', 'ops': 'a * b'},
    {'info': 'Div', 'ops': 'a / b'},
    {'info': 'MatMul', 'ops': 'chainer.functions.matmul(a, b, transb=True)'},
    {'info': 'Maximum', 'ops': 'chainer.functions.maximum(a, b)'},
    {'info': 'Minimum', 'ops': 'chainer.functions.minimum(a, b)'},
)
class TestBinaryMathOperators(unittest.TestCase):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a, b):
                if not isinstance(a, chainer.Variable):
                    a = chainer.Varaible(a)
                if not isinstance(b, chainer.Variable):
                    b = chainer.Varaible(b)
                return eval(self.ops)

        self.model = Model(self.ops)
        a = chainer.Variable(np.ones((2, 3), dtype=np.float32))
        b = chainer.Variable(np.ones((2, 3), dtype=np.float32) * 2)
        self.x = (a, b)
        print(self.x)
        self.fn = self.info + '.onnx'

    def test_compatibility(self):
        if MXNET_SUPPORT[self.info]:
            test_mxnet.check_compatibility(self.model, self.x, self.fn)
        else:
            onnx_chainer.export(self.model, self.x)
