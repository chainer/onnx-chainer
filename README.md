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
- Pad [^pad1] [^pad2]
- Reshape
- Space2Depth
- SplitAxis
- Squeeze
- Tile
- Transpose

[^pad1]: mode should be either 'constant', 'reflect', or 'edge'
[^pad2]: ONNX doesn't support multiple constant values for Pad operation

### Connection

- Convolution2DFunction
- ConvolutionND
- Deconvolution2DFunction
- DeconvolutionND
- EmbedIDFunction [^embed1]
- LinearFunction

[^embed1]: Current ONNX doesn't support ignore_label for EmbedID

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
- MatMul [^matmul1]
- Maximum
- Minimum
- Sqrt
- SquaredDifference
- Sum

[^matmul1]: Current ONNX doesn't support transpose options for matmul ops

### Noise

- Dropout [^dropout1]

[^dropout1]: In test mode, all dropout layers aren't included in the exported file

### Pooling

- AveragePooling2D
- AveragePoolingND
- MaxPooling2D
- MaxPoolingND

### Normalization

- BatchNormalization
- FixedBatchNormalization
- LocalResponseNormalization
