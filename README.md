# ONNX-Chainer
![pypi](https://img.shields.io/pypi/v/onnx-chainer.svg)
![Build Status](https://travis-ci.org/chainer/onnx-chainer.svg?branch=master)
![MIT License](https://img.shields.io/github/license/mitmul/onnx-chainer.svg)

This is an add-on package for ONNX support by Chainer.

## Tested environment

- ONNX 1.1.1
- Chainer 3.5.0, 4.2.0
- Python 2.7.14, 3.5.5, 3.6.5

### Compatibility tests

- with MXNet 1.2.0
- with NNVM (under TVM repository at commit ID = `ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc`)

## Installation

### On Ubuntu 14.04/16.04

**Please install Chainer first.**

```bash
pip install chainer
pip install onnx-chainer
```

## Run Test

### 1. Build Docker images

```bash
cd docker
bash build_docker.sh
```

### 2. Run tests

```bash
bash docker/run_tests.sh
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

## Load models from MXNet

Install [MXNet](https://github.com/apache/incubator-mxnet) first, then try the following code:

```python
import collections

import mxnet
import numpy as np

import chainer
import chainer.functions as F
import chainercv.links as C
import onnx_chainer

# Prepare an input tensor
x = np.random.rand(1, 3, 224, 224).astype(np.float32) * 255

# Run the model on the data
with chainer.using_config('train', False):
    chainer_out = model(x).array

# Export Chainer model into ONNX
onnx_chainer.export(model, x, fn)

# Load ONNX model into MXNet symbol
sym, arg, aux = mxnet.contrib.onnx.import_model(fn)

# Find the name of input tensor
data_names = [graph_input for graph_input in sym.list_inputs()
                if graph_input not in arg and graph_input not in aux]
data_shapes = [(data_names[0], x.shape)]

# Create MXNet model
mod = mxnet.mod.Module(
    symbol=sym, data_names=data_names, context=mxnet.cpu(),
    label_names=None)
mod.bind(
    for_training=False, data_shapes=data_shapes,
    label_shapes=None)
mod.set_params(
    arg_params=arg, aux_params=aux, allow_missing=True,
    allow_extra=True)

# Create input data
Batch = collections.namedtuple('Batch', ['data'])
input_data = Batch([mxnet.nd.array(x)])

# Forward computation using MXNet
mod.forward(input_data)

# Retrieve the output of forward result
mxnet_out = mod.get_outputs()[0].asnumpy()

# Check the prediction results are same
assert np.argmax(chainer_out) == np.argmax(mxnet_out)

# Check both outputs have same values
np.testing.assert_almost_equal(chainer_out, mxnet_out, decimal=5)
```

## Compile the Chainer model via ONNX

Please install [TVM](https://github.com/dmlc/tvm/tree/ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc) at a specified commit ID (ebdde3c277a9807a67b233cecfaf6d9f96c0c1bc) with NNVM first.

```python
import collections

import numpy as np
import onnx

import chainer
import chainer.functions as F
import chainercv.links as C
import nnvm
import onnx_chainer
import tvm

model = C.ResNet50(pretrained_model='imagenet', arch='he')
# Change cover_all option to False to match the default behavior of MXNet's pooling
model.pool1 = lambda x: F.max_pooling_2d(x, ksize=3, stride=2, cover_all=False)
save_as_onnx_then_import_from_nnvm(model, 'resnet50.onnx')

# Prepare an input tensor
x = np.random.rand(1, 3, 224, 224).astype(np.float32) * 255

# Run the model on the data
with chainer.using_config('train', False):
    chainer_out = model(x).array

# Export Chainer model into ONNX
onnx_chainer.export(model, x, fn)

# Load the saved ONNX file using ONNX module
model_onnx = onnx.load(fn)

# Convert the ONNX model object into NNVM symbol
sym, params = nnvm.frontend.from_onnx(model_onnx)

# Choose the compilation target
target = 'llvm'

# Extract the name of input variable in the ONNX graph
input_name = sym.list_input_names()[0]
shape_dict = {input_name: x.shape}

# Compile the model using NNVM
graph, lib, params = nnvm.compiler.build(
    sym, target, shape_dict, params=params)

# Convert the compiled model into TVM module
module = tvm.contrib.graph_runtime.create(graph, lib, tvm.cpu(0))

# Set the input tensor x
module.set_input(input_name, tvm.nd.array(x))
module.set_input(**params)

# Run the model
module.run()

# Retrieve the inference result
out_shape = (1, 1000)
output = tvm.nd.empty(out_shape, ctx=tvm.cpu(0))
nnvm_output = module.get_output(0, output).asnumpy()

# Check both outputs have same values
np.testing.assert_almost_equal(chainer_out, nnvm_output, decimal=5)
```

## Supported Functions

Currently 49 Chainer Functions are supported to export in ONNX format.

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
- MatMul
- Maximum
- Minimum
- Sqrt
- Sum

### Noise

- Dropout <sup>[4](#dropout1)</sup>

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
<a name="dropout1">4</a>: In test mode, all dropout layers aren't included in the exported file<br />
