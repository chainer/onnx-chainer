import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

import onnx_chainer


class TestLeNet5(unittest.TestCase):

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
        self.x = np.zeros((1, 3, 28, 28), dtype=np.float32)

    def test_export(self):
        onnx_chainer.export(self.model, self.x, export_params=False)


class TestVGG16(unittest.TestCase):

    def setUp(self):

        self.model = L.VGG16Layers(None)
        self.x = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def test_export(self):
        onnx_chainer.export(self.model, self.x, export_params=False)


class TestResNet50(unittest.TestCase):

    def setUp(self):

        self.model = L.ResNet50Layers(None)
        self.x = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def test_export(self):
        onnx_chainer.export(self.model, self.x, export_params=False)
