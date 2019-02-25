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
    {'use_bn': True, 'out_type': 'dict'},
    {'use_bn': False, 'out_type': 'dict'},
    {'use_bn': True, 'out_type': 'tuple'},
    {'use_bn': True, 'out_type': 'list'},
)
class TestMultipleOutput(unittest.TestCase):

    def get_model(self, use_bn=False, out_type=None):
        class Model(chainer.Chain):

            def __init__(self, use_bn=False, out_type=None):
                super(Model, self).__init__()

                self._use_bn = use_bn
                self._out_type = out_type
                with self.init_scope():
                    self.conv = L.Convolution2D(None, 32, ksize=3, stride=1)
                    if self._use_bn:
                        self.bn = L.BatchNormalization(32)

            def __call__(self, x):
                h = self.conv(x)
                if self._use_bn:
                    h = self.bn(h)
                o1 = F.tanh(h)
                o2 = F.sigmoid(h)
                if self._out_type == 'dict':
                    return {
                        'Tanh_0': o1,
                        'Sigmoid_0': o2
                    }
                elif self._out_type == 'tuple':
                    return o1, o2
                elif self._out_type == 'list':
                    return [o1, o2]

        return Model(use_bn=use_bn, out_type=out_type)

    def test_multiple_outputs(self):
        model = self.get_model(use_bn=self.use_bn, out_type=self.out_type)
        x = np.zeros((1, 3, 32, 32), dtype=np.float32)
        # 'out_keys' is necessary even if self.out_type is tuple or list
        # because onnxruntime does not guarantee the order of outputs when
        # output keys are None.
        test_onnxruntime.check_output(
            model, x, 'MultipleOutputs.onnx', out_keys=['Tanh_0', 'Sigmoid_0'])


class TestIntermidiateOutput(unittest.TestCase):

    def get_model(self):
        class Model(chainer.Chain):

            def __init__(self):
                super().__init__()
                with self.init_scope():
                    self.l1 = L.Linear(4)
                    self.l2 = L.Linear(5, initial_bias=0.1)

            def __call__(self, x):
                y = self.l1(x)
                z = self.l2(y)
                return y, z
        return Model()

    def test_outputs(self):
        model = self.get_model()
        x = np.ones((1, 3), dtype=np.float32)
        # TODO(disktnk) output keys are not intuitive
        test_onnxruntime.check_output(
            model, x, 'IntermidiateOutput.onnx',
            out_keys=['Identity_0', 'Gemm_1'])
