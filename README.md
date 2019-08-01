# ONNX-Chainer
[![PyPI](https://img.shields.io/pypi/v/onnx-chainer.svg)](https://pypi.org/project/onnx-chainer/)
[![GitHub license](https://img.shields.io/github/license/chainer/onnx-chainer.svg)](https://github.com/chainer/onnx-chainer)
[![Build Status](https://travis-ci.org/chainer/onnx-chainer.svg?branch=master)](https://travis-ci.org/chainer/onnx-chainer)
[![codecov](https://codecov.io/gh/chainer/onnx-chainer/branch/master/graph/badge.svg)](https://codecov.io/gh/chainer/onnx-chainer)

This is an add-on package for ONNX support by Chainer.

## Tested environment

- Python 3.5.5, 3.6.7, 3.7.2
- ONNX 1.4.1, 1.5.0
    - opset version 7, 8, 9, 10
- Chainer stable, preview
- ONNX-Runtime 0.4.0

**(You can still specify all opset versions <= 9, but please noted that opset versions <= 6 are not tested)**

## Installation

### On Ubuntu 14.04/16.04

```bash
pip install onnx-chainer
```

## Run Test

### 1. Install test modules

```bash
$ pip install onnx-chainer[test-cpu]
```

Or, on GPU environment

```bash
$ pip install cupy  # or cupy-cudaXX is useful
$ pip install onnx-chainer[test-gpu]
```

### 2. Run tests

```bash
$ pytest -m "not gpu"
```

Or, on GPU environment

```bash
$ pytest
```


## Quick Start

First, install [ChainerCV](https://github.com/chainer/chainercv) to get the pre-trained models.

```python
import numpy as np

import chainer
import chainercv.links as C
import onnx_chainer

model = C.VGG16(pretrained_model='imagenet')

# Pseudo input
x = np.zeros((1, 3, 224, 224), dtype=np.float32)

onnx_chainer.export(model, x, filename='vgg16.onnx')
```


## Supported Functions

Currently 82 Chainer Functions are supported to export in ONNX format.

### Activation

- ClippedReLU
- ELU
- HardSigmoid
- LeakyReLU
- LogSoftmax
- PReLUFunction
- ReLU
- Sigmoid
- Softmax
- Softplus
- Tanh

### Array

- Cast
- Concat
- Copy
- Depth2Space
- Dstack
- ExpandDims
- GetItem
- Hstack
- Pad <sup>[1](#pad1)</sup><sup>[2](#pad2)</sup>
- Repeat
- Reshape
- ResizeImages
- Separate
- Shape <sup>[5](#shape1)</sup>
- Space2Depth
- SplitAxis
- Squeeze
- Stack
- Swapaxes
- Tile
- Transpose
- Vstack
- Where

### Connection

- Convolution2DFunction
- ConvolutionND
- Deconvolution2DFunction
- DeconvolutionND
- EmbedIDFunction <sup>[3](#embed1)</sup>
- LinearFunction

### Loss

- SoftmaxCrossEntropy

### Math

- Absolute
- Add
- AddConstant
- ArgMax
- ArgMin
- BroadcastTo
- Clip
- Div
- DivFromConstant
- Exp
- Identity
- LinearInterpolate
- LogSumExp
- MatMul
- Max
- Maximum
- Mean
- Min
- Minimum
- Mul
- MulConstant
- Neg
- PowVarConst
- Prod
- RsqrtGPU
- Sqrt
- Square
- Sub
- SubFromConstant
- Sum

### Noise

- Dropout <sup>[4](#dropout1)</sup>

### Normalization

- BatchNormalization
- FixedBatchNormalization
- LocalResponseNormalization
- NormalizeL2

### Pooling

- AveragePooling2D
- AveragePoolingND
- MaxPooling2D
- MaxPoolingND
- ROIPooling2D
- Unpooling2D


## Contribution

Any contribution to ONNX-Chainer is welcome!

- Python codes follow [Chainer Coding Guidelines](https://docs.chainer.org/en/stable/contribution.html#coding-guidelines)

---

<a name="pad1">1</a>: mode should be either 'constant', 'reflect', or 'edge'<br />
<a name="pad2">2</a>: ONNX doesn't support multiple constant values for Pad operation<br />
<a name="embed1">3</a>: Current ONNX doesn't support ignore_label for EmbedID<br />
<a name="dropout1">4</a>: In test mode, all dropout layers aren't included in the exported file<br />
<a name="shape1">5</a>: Chainer doesn't support Shape function<br />
