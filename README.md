# ONNX-Chainer

This is an add-on package for ONNX support by Chainer.

## Requirements

- onnx==0.2.1
- chainer>=2.1.0

## Installation

```bash
pip install onnx-chainer
```

## Quick Start

```python
import numpy as np
import chainer.links as L
import onnx_chainer

model = L.VGG16()

# Pseudo input
x = np.zeros((1, 3, 224, 224), stype=np.float32)

onnx_chainer.export(model, x, filename='MLP.onnx')
```