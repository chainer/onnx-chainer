import unittest

import chainer
import chainer.links as L
import numpy as np
import onnx_chainer


class TestVGG16(unittest.TestCase):

    def setUp(self):

        self.model = L.VGG16Layers(None)
        self.x = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def test_export(self):
        onnx_chainer.export(self.model, self.x)


class TestResNet50(unittest.TestCase):

    def setUp(self):

        self.model = L.ResNet50Layers(None)
        self.x = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def test_export(self):
        onnx_chainer.export(self.model, self.x)
