import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

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
        self.ins = (np.zeros((1, 5), dtype=np.float32),
                    np.zeros((1, 5), dtype=np.float32),
                    np.zeros((1, 5), dtype=np.float32))
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


class TestOutScopeNodeParam(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()

                self.frac = chainer.Parameter(np.array(2, dtype=np.float32))

            def __call__(self, x):
                return x / self.frac

        self.model = Model()
        self.fn = 'OutScopeNodeParam.onnx'

    def test_out_scope_node_input(self):
        x = chainer.Variable(np.array(1, dtype=np.float32))
        test_onnxruntime.check_output(self.model, x, self.fn)
