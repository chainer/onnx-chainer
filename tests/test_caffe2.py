import os
import tempfile
import unittest

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from onnx_caffe2.backend import Caffe2Backend
from onnx_caffe2.helper import benchmark_caffe2_model
from onnx_caffe2.helper import benchmark_pytorch_model
from onnx_caffe2.helper import c2_native_run_net
from onnx_caffe2.helper import load_caffe2_net
from onnx_caffe2.helper import save_caffe2_net

import onnx
import onnx_chainer


class TestVGG16(unittest.TestCase):

    def setUp(self):

        self.model = L.VGG16Layers(None)
        self.x = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def test_export_test(self):
        chainer.config.train = False
        with tempfile.NamedTemporaryFile('wb') as fp:
            onnx_chainer.export(self.model, self.x, fp)
            onnx_model = onnx.ModelProto.FromString(open(fp.name, 'rb').read())
            init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(
                onnx_model.graph, device='CPU')

            print(dir(fp))
            print(fp.name)
            print(os.path.exists(fp.name))

    def test_export_train(self):
        chainer.config.train = True
        onnx_chainer.export(self.model, self.x)


class TestResNet50(unittest.TestCase):

    def setUp(self):

        self.model = L.ResNet50Layers(None)
        self.x = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def test_export_test(self):
        chainer.config.train = False
        onnx_chainer.export(self.model, self.x)

    def test_export_train(self):
        chainer.config.train = True
        onnx_chainer.export(self.model, self.x)
