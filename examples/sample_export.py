#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import chainer
import numpy as np

import chainercv.links as C
import onnx_chainer

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

model = C.VGG16(pretrained_model='imagenet')
if args.gpu >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

# Pseudo input
x = model.xp.zeros((1, 3, 224, 224), dtype=np.float32)

onnx_chainer.export(model, x, filename='vgg16.onnx')
