import os
import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import onnx_chainer


class TestExportTestCase(unittest.TestCase):

    def setUp(self):
        self.model = chainer.Sequential(
            L.Convolution2D(None, 16, 5, 1, 2),
            F.relu,
            L.Convolution2D(16, 8, 5, 1, 2),
            F.relu,
            L.Convolution2D(8, 5, 5, 1, 2),
            F.relu,
            L.Linear(None, 100),
            F.relu,
            L.Linear(100, 10)
        )
        x = np.zeros((1, 3, 28, 28), dtype=np.float32)
        self.x = chainer.as_variable(x)

    def test_export_testcase(self):
        # Just check the existence of pb files
        path = 'out/test_export_testcase'
        onnx_chainer.export_testcase(
            self.model, (self.x,), path)

        assert os.path.isfile(os.path.join(path, 'model.onnx'))
        assert os.path.isfile(os.path.join(path, 'test_data_set_0',
                                           'input_0.pb'))
        assert os.path.isfile(os.path.join(path, 'test_data_set_0',
                                           'output_0.pb'))

    def test_output_grad(self):
        path = 'out/test_export_testcase_with_grad'
        onnx_chainer.export_testcase(
            self.model, (self.x,), path, output_grad=True, train=True)

        assert os.path.isfile(os.path.join(path, 'model.onnx'))
        assert os.path.isfile(os.path.join(path, 'test_data_set_0',
                                           'input_0.pb'))
        assert os.path.isfile(os.path.join(path, 'test_data_set_0',
                                           'output_0.pb'))
        # 10 gradient files should be there
        for i in range(10):
            assert os.path.isfile(os.path.join(path, 'test_data_set_0',
                                               'gradient_{}.pb'.format(i)))
        assert i == 9
