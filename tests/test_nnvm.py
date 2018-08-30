#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

import chainer
from chainer import testing
import chainer.functions as F
import numpy as np
import onnx
import onnx_chainer

import chainercv.links as C
import nnvm
import nnvm.compiler
import tvm


@testing.parameterize(*testing.product({
    'model': [
        {'mod': C, 'arch': 'VGG16', 'kwargs': {'pretrained_model': None}},
        {'mod': C, 'arch': 'ResNet50', 'kwargs': {'pretrained_model': None, 'arch': 'he'}},
    ],
    'opset_version': [
        # 1, 2, 3, 4, 5, 6, 7
        7
    ]
}))
class TestWithNNVMBackend(unittest.TestCase):

    def setUp(self):
        m = self.model
        self.model = getattr(m['mod'], m['arch'])(**m['kwargs'])

        # To match the behavior with MXNet's default max pooling
        if m['arch'] == 'ResNet50':
            self.model.pool1 = lambda x: F.max_pooling_2d(
                x, ksize=3, stride=2, cover_all=False)

        self.x = np.random.uniform(
            -5, 5, size=(1, 3, 224, 224)).astype(np.float32)
        with chainer.using_config('train', True):
            self.model(self.x)  # Prevent all NaN output
        self.fn = '{}.onnx'.format(m['arch'])

    def test_compatibility(self):
        chainer.config.train = False
        with chainer.using_config('train', False):
            chainer_out = self.model(self.x).array

        onnx_chainer.export(self.model, self.x, self.fn, opset_version=self.opset_version)

        model_onnx = onnx.load(self.fn)
        sym, params = nnvm.frontend.from_onnx(model_onnx)

        target = 'llvm'
        input_name = sym.list_input_names()[0]

        shape_dict = {input_name: self.x.shape}
        graph, lib, params = nnvm.compiler.build(
            sym, target, shape_dict, params=params, dtype={input_name: 'float32'})
        module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu(0))
        module.set_input(input_name, tvm.nd.array(self.x))
        module.set_input(**params)
        module.run()

        out_shape = (1, 1000)
        output = tvm.nd.empty(out_shape, ctx=tvm.cpu(0))
        nnvm_output = module.get_output(0, output).asnumpy()

        np.testing.assert_almost_equal(
            chainer_out, nnvm_output, decimal=5)

        os.remove(self.fn)
