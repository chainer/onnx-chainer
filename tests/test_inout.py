import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import onnx

import chainercv.links as C
import onnx_chainer
from onnx_chainer.testing import test_onnxruntime


class TestMultipleInputs(unittest.TestCase):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.prelu = L.PReLU()

            def __call__(self, x, y, z):
                return F.relu(x) + y * z

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
