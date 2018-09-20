import unittest

import numpy as np

import chainer
import onnx
import onnx_chainer
from chainer import testing
from onnx_chainer.testing import test_mxnet

MXNET_OPSET_VERSION = {
    'Add': (1, 6, 7),
    'AddConst': (1, 6, 7),
    'Absolute': (1, 6),
    'Div': (1, 6, 7),
    'Mul': (1, 6, 7),
    'Neg': (1, 6),
    'PowVarConst': (1, 7),
    'Sub': (1, 6, 7),
    'Clip': (6,),
    'Exp': (1, 6),
    'MatMul': (1, 6, 7),
    'Maximum': (1, 6),
    'Minimum': (1, 6),
    'Sqrt': (1, 6),
    'Sum': (1,),
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
    {'info': 'Max', 'ops': 'chainer.functions.max(a)'},
    {'info': 'Mean', 'ops': 'chainer.functions.mean(a)'},
    {'info': 'Min', 'ops': 'chainer.functions.min(a)'},
    {'info': 'Prod', 'ops': 'chainer.functions.prod(a)'},
    {'info': 'LogSumExp', 'ops': 'chainer.functions.logsumexp(a)'},
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
        if MXNET_OPSET_VERSION[self.info] is not None:
            for opset_version in MXNET_OPSET_VERSION[self.info]:
                test_mxnet.check_compatibility(
                    self.model, self.a, self.fn, opset_version=opset_version)
        for opset_version in range(1, onnx.defs.onnx_opset_version() + 1):
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
        self.fn = self.info + '.onnx'

    def test_compatibility(self):
        if MXNET_OPSET_VERSION[self.info] is not None:
            for opset_version in MXNET_OPSET_VERSION[self.info]:
                test_mxnet.check_compatibility(
                    self.model, self.x, self.fn, opset_version=opset_version)
        for opset_version in range(1, onnx.defs.onnx_opset_version() + 1):
            onnx_chainer.export(self.model, self.x)
