#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import chainercv.links as C
import onnx_chainer
import onnxruntime as rt
import chainer
import chainer.functions as F

model = C.VGG16(pretrained_model='imagenet')

model = chainer.Sequential(F.elu)

# Pseudo input
x = (np.zeros((1, 3, 224, 224), dtype=np.float32),)

onnx_model = onnx_chainer.export(model, x)

sess = rt.InferenceSession(onnx_model.SerializeToString())

input_names = [i.name for i in sess.get_inputs()]
rt_out = sess.run(
    None, {name: array for name, array in zip(input_names, x)})
