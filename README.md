# ONNX-Chainer
![pypi](https://img.shields.io/pypi/v/onnx-chainer.svg)
![Build Status](https://travis-ci.org/mitmul/onnx-chainer.svg?branch=master)
![MIT License](https://img.shields.io/github/license/mitmul/onnx-chainer.svg)

This is an add-on package for ONNX support by Chainer.

## Requirements

- onnx==0.2.1
- chainer>=3.1.0

## Installation

See [INSTALL.md](INSTALL.md)

## Quick Start

```python
import numpy as np
import chainer
import chainer.links as L
import onnx_chainer

model = L.VGG16Layers()

# Pseudo input
x = np.zeros((1, 3, 224, 224), dtype=np.float32)

# Don't forget to set train flag off!
chainer.config.train = False

onnx_chainer.export(model, x, filename='VGG16.onnx')
```

## Supported Functions

Currently 50 Chainer Functions are supported to export in ONNX format.

### Activation

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
- Depth2Space
- Pad <sup>[1](#pad1)</sup><sup>[2](#pad2)</sup>
- Reshape
- Space2Depth
- SplitAxis
- Squeeze
- Tile
- Transpose

### Connection

- Convolution2DFunction
- ConvolutionND
- Deconvolution2DFunction
- DeconvolutionND
- EmbedIDFunction <sup>[3](#embed1)</sup>
- LinearFunction

### Math

- Add
- Absolute
- Div
- Mul
- Neg
- PowVarConst
- Sub
- Clip
- Exp
- Identity
- MatMul <sup>[4](#matmul1)</sup>
- Maximum
- Minimum
- Sqrt
- SquaredDifference
- Sum

### Noise

- Dropout <sup>[5](#dropout1)</sup>

### Pooling

- AveragePooling2D
- AveragePoolingND
- MaxPooling2D
- MaxPoolingND

### Normalization

- BatchNormalization
- FixedBatchNormalization
- LocalResponseNormalization

---

<a name="pad1">1</a>: mode should be either 'constant', 'reflect', or 'edge'<br />
<a name="pad2">2</a>: ONNX doesn't support multiple constant values for Pad operation<br />
<a name="embed1">3</a>: Current ONNX doesn't support ignore_label for EmbedID<br />
<a name="matmul1">4</a>: Current ONNX doesn't support transpose options for matmul ops<br />
<a name="dropout1">5</a>: In test mode, all dropout layers aren't included in the exported file<br />
