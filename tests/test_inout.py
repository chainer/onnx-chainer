import unittest

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing
import numpy as np

from onnx_chainer.testing import input_generator
from onnx_chainer.testing import test_onnxruntime


class TestMultipleInputs(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.prelu = L.PReLU()

            def __call__(self, x, y, z):
                return F.relu(x) + self.prelu(y) * z

        self.model = Model()
        self.ins = (input_generator.increasing(1, 5),
                    input_generator.increasing(1, 5),
                    input_generator.increasing(1, 5))
        self.fn = 'MultiInputs.onnx'

    def test_arrays(self):
        test_onnxruntime.check_output(self.model, self.ins, self.fn)

    def test_variables(self):
        ins = [chainer.Variable(i) for i in self.ins]
        test_onnxruntime.check_output(self.model, ins, self.fn)

    def test_array_dicts(self):
        arg_names = ['x', 'y', 'z']  # current exporter ignores these names
        ins = {arg_names[i]: v for i, v in enumerate(self.ins)}
        test_onnxruntime.check_output(self.model, ins, self.fn)

    def test_variable_dicts(self):
        arg_names = ['x', 'y', 'z']  # current exporter ignores these names
        ins = {arg_names[i]: chainer.Variable(v)
               for i, v in enumerate(self.ins)}
        test_onnxruntime.check_output(self.model, ins, self.fn)


class TestImplicitInput(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()

                self.frac = chainer.Parameter(np.array(2, dtype=np.float32))

            def __call__(self, x):
                return x / self.frac

        self.model = Model()
        self.fn = 'ImplicitInput.onnx'

    def test_implicit_input(self):
        x = chainer.Variable(np.array(1, dtype=np.float32))
        test_onnxruntime.check_output(self.model, x, self.fn)


@testing.parameterize(
    {'use_bn': True},
    {'use_bn': False},
)
class TestMultipleOutput(unittest.TestCase):

    def get_model(self, use_bn=False):
        class Model(chainer.Chain):

            def __init__(self, use_bn=False):
                super(Model, self).__init__()

                self._use_bn = use_bn
                with self.init_scope():
                    self.conv = L.Convolution2D(None, 32, ksize=3, stride=1)
                    if self._use_bn:
                        self.bn = L.BatchNormalization(32)

            def __call__(self, x):
                h = self.conv(x)
                if self._use_bn:
                    h = self.bn(h)
                return {
                    'Tanh_0': F.tanh(h),
                    'Sigmoid_0': F.sigmoid(h)
                }

        return Model(use_bn=use_bn)

    def test_multiple_outputs(self):
        model = self.get_model(use_bn=self.use_bn)
        x = np.zeros((1, 3, 32, 32), dtype=np.float32)
        test_onnxruntime.check_output(
            model, x, 'MultipleOutputs.onnx', out_keys=['Tanh_0', 'Sigmoid_0'])
