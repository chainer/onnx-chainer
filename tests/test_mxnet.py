from collections import namedtuple
import os
import unittest

import chainer
from chainer import testing
import chainer.links as L
import numpy as np

import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
import onnx_chainer


@testing.parameterize(
    # {'net': 'VGG16Layers'},
    {'net': 'ResNet50Layers'},
)
class TestOutputWithMXNetBackend(unittest.TestCase):

    def setUp(self):
        self.model = getattr(L, self.net)()
        self.x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    def test_compatibility(self):
        self.save_as_onnx_then_import_from_mxnet(self.model, self.x, self.net)

    def save_as_onnx_then_import_from_mxnet(self, model, x, fn):
        chainer.config.train = False
        chainer_out_all = model(x, layers=['res5', 'prob'])
        chainer_out = chainer_out_all['prob'].array

        onnx_chainer.export(model, x, fn)

        sym, arg, aux = onnx_mxnet.import_model(fn)
        data_names = [graph_input for graph_input in sym.list_inputs()
                      if graph_input not in arg and graph_input not in aux]

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

        # Extract intermediate layer outputs
        relu48 = None
        for mxnet_internal_symbol in mod.symbol.get_internals():
            if mxnet_internal_symbol.name == 'relu48':
                relu48 = mxnet_internal_symbol
            print(mxnet_internal_symbol)
        data_names = [
            graph_input for graph_input in relu48.list_inputs()
            if graph_input not in arg and graph_input not in aux]
        mod3 = mx.mod.Module(
            symbol=relu48, data_names=data_names,
            label_names=None, context=mx.cpu())
        mod3.bind(
            for_training=False, data_shapes=[(data_names[0], x.shape)],
            label_shapes=None)
        mod3.set_params(
            arg_params=arg, aux_params=aux, allow_missing=True,
            allow_extra=True)
        mod3.forward(Batch([mx.nd.array(x)]))
        mxnet_internal_output = mod3.get_outputs()[0].asnumpy()

        # Chainer's intermediate layer output
        chainer_internal_output = chainer_out_all['res5'].array

        np.testing.assert_almost_equal(
            mxnet_internal_output, chainer_internal_output, decimal=5)

        mxnet_outs = mod.get_outputs()
        mxnet_out = mxnet_outs[0].asnumpy()

        print('Chainer:', np.argmax(chainer_out[0]))
        print('MXNet:', np.argmax(mxnet_out[0]))

        np.testing.assert_almost_equal(
            chainer_out, mxnet_out, decimal=5)

        os.remove(fn)
