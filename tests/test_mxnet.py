from collections import namedtuple
import unittest

from chainer import testing
import chainer.links as L
import numpy as np
import os
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
import onnx_chainer


@testing.parameterize(
    {'net': 'VGG16Layers'},
    {'net': 'ResNet50Layers'},
)
class TestOutputWithMXNetBackend(unittest.TestCase):

    def setUp(self):
        self.model = getattr(L, self.net)()
        self.x = np.random.rand(1, 3, 224, 224).astype(np.float32)

    def test_compatibility(self):
        self.save_as_onnx_then_import_from_mxnet(self.model, self.x, self.net)

    def save_as_onnx_then_import_from_mxnet(self, model, x, fn):
        chainer_out = model(x)['prob'].array

        onnx_model = onnx_chainer.export(model, x, fn)
        data_names = [onnx_model.graph.input[-1].name]
        print(onnx_model.graph.input[-1])
        sym, arg, aux = onnx_mxnet.import_model(fn)

        mod = mx.mod.Module(
            symbol=sym, data_names=data_names, context=mx.cpu(),
            label_names=None)
        mod.bind(
            for_training=False, data_shapes=[(data_names[0], x.shape)],
            label_shapes=None)
        mod.set_params(
            arg_params=arg, aux_params=aux, allow_missing=True,
            allow_extra=True)

        Batch = namedtuple('Batch', ['data'])
        mod.forward(Batch([mx.nd.array(x)]))

        mxnet_out = mod.get_outputs()[0].asnumpy()

        print(mxnet_out.shape)

        np.testing.assert_almost_equal(
            chainer_out, mxnet_out, decimal=5)

        os.remove(fn)
