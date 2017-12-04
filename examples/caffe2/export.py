#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import chainer
import chainer.links as L
import numpy as np
from onnx_caffe2.backend import Caffe2Backend
from onnx_caffe2.backend import run_model
from onnx_caffe2.helper import save_caffe2_net

import onnx_chainer

chainer.config.train = False

model = L.VGG16Layers()
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
onnx_model = onnx_chainer.export(model, x)

print(x)
y = model(x)
st = time.time()
y = model(x)
print('Chainer:', time.time() - st, 'sec')
if isinstance(y, dict):
    y = y['prob']
chainer_out = y.array

init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(
    onnx_model.graph, device='CPU')
init_file = "./vgg16_init.pb"
predict_file = "./vgg16_predict.pb"
save_caffe2_net(init_net, init_file, output_txt=False)
save_caffe2_net(predict_net, predict_file, output_txt=True)

print(x)
caffe2_out = run_model(onnx_model, [x])[0]

# prepared_backend = prepare(onnx_model)
# st = time.time()
# # caffe2_out = prepared_backend.run()
# # p = workspace.Predictor(init_net, predict_net)
# # st = time.time()
# # p = workspace.Predictor(init_net, predict_net)
# # caffe2_out = p.run([x])[0]
# print('Caffe2:', time.time() - st, 'sec')

np.testing.assert_almost_equal(
    chainer_out, caffe2_out, decimal=5)
