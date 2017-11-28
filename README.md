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
import chainer.links as L
import onnx_chainer

model = L.VGG16Layers()

# Pseudo input
x = np.zeros((1, 3, 224, 224), dtype=np.float32)

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
- Pad
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
- EmbedIDFunction
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
- MatMul
- Maximum
- Minimum
- Sqrt
- SquaredDifference
- Sum

### Noise

- Dropout (NOTE: In test mode, it's not exported)

### Pooling

- AveragePooling2D
- AveragePoolingND
- MaxPooling2D
- MaxPoolingND

### Normalization

- BatchNormalization
- FixedBatchNormalization
- LocalResponseNormalization
