import unittest

import chainer
import numpy as np

from onnx_chainer.testing import test_onnxruntime


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
