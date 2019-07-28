import chainer
import chainer.functions as F
import chainer.links as L
from chainer.testing import attr
import numpy as np

import chainercv.links as C
from tests.helper import ONNXModelTest


class TestLeNet5(ONNXModelTest):

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

    def test_output(self):
        self.expect(self.model, self.x)

    @attr.gpu
    def test_output_gpu(self):
        model, x = self.to_gpu(self.model, self.x)
        self.expect(model, x)


class TestVGG16(ONNXModelTest):

    def setUp(self):
        self.model = C.VGG16(
            pretrained_model=None, initialW=chainer.initializers.Uniform(1))
        self.x = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def test_output(self):
        self.expect(self.model, self.x)

    @attr.gpu
    def test_output_gpu(self):
        model, x = self.to_gpu(self.model, self.x)
        self.expect(model, x)


class TestResNet50(ONNXModelTest):

    def setUp(self):
        self.model = C.ResNet50(
            pretrained_model=None, initialW=chainer.initializers.Uniform(1),
            arch='he')
        self.model.pool1 = lambda x: F.max_pooling_2d(
            x, ksize=3, stride=2, cover_all=False)
        self.x = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def test_output(self):
        self.expect(self.model, self.x)

    @attr.gpu
    def test_output_gpu(self):
        model, x = self.to_gpu(self.model, self.x)
        self.expect(model, x)
