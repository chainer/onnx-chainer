# ONNX-Chainer
[![PyPI](https://img.shields.io/pypi/v/onnx-chainer.svg)](https://pypi.org/project/onnx-chainer/)
[![GitHub license](https://img.shields.io/github/license/chainer/onnx-chainer.svg)](https://github.com/chainer/onnx-chainer)
[![Build Status](https://travis-ci.org/chainer/onnx-chainer.svg?branch=master)](https://travis-ci.org/chainer/onnx-chainer)
[![codecov](https://codecov.io/gh/chainer/onnx-chainer/branch/master/graph/badge.svg)](https://codecov.io/gh/chainer/onnx-chainer)
[![Documentation Status](https://readthedocs.org/projects/onnx-chainer/badge/?version=latest)](https://onnx-chainer.readthedocs.io/en/latest/?badge=latest)

All code and functionalities of ONNX-Chainer have been merged into [Chainer](https://chainer.org/) v7rc1 and this repository supports only bug fixes.

This is an add-on package for ONNX support by Chainer.

## Tested environment

see [Tested environments](https://onnx-chainer.readthedocs.io/en/latest/introduction/index.html#tested-environments)

## Installation

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

see [Supported Functions](https://onnx-chainer.readthedocs.io/en/latest/introduction/index.html#supported-functions)


## Contribution

Any contribution to ONNX-Chainer is welcome!

- Python codes follow [Chainer Coding Guidelines](https://docs.chainer.org/en/stable/contribution.html#coding-guidelines)
