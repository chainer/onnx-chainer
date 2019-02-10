import warnings

import chainer
import numpy as np
import onnx

import onnx_chainer

try:
    import onnxruntime as rt
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    warnings.warn(
        'ONNXRuntime is not installed. Please install it to use '
        ' the testing utility for ONNX-Chainer\'s converters.',
        ImportWarning)
    ONNXRUNTIME_AVAILABLE = False


MINIMUM_OPSET_VERSION = 7


def check_output(model, x, fn, out_key='prob', opset_version=None):
    if opset_version is None:
        opset_version = onnx.defs.onnx_opset_version()
    if not ONNXRUNTIME_AVAILABLE:
        raise ImportError('check_output requires onnxruntime.')

    chainer.config.train = False

    # Forward computation
    if isinstance(x, (list, tuple)):
        for i in x:
            assert isinstance(i, (np.ndarray, chainer.Variable))
        chainer_out = model(*x)
        x_rt = tuple(
            _x.array if isinstance(_x, chainer.Variable) else _x for _x in x)
    elif isinstance(x, dict):
        chainer_out = model(**x)
        x_rt = tuple(_x.array if isinstance(_x, chainer.Variable) else _x
                     for _, _x in x.items())
    elif isinstance(x, np.ndarray):
        chainer_out = model(chainer.Variable(x))
        x_rt = x,
    elif isinstance(x, chainer.Variable):
        chainer_out = model(x)
        x_rt = x.array,
    else:
        raise ValueError(
            'The \'x\' argument should be a list, tuple or dict of '
            'numpy.ndarray or chainer.Variable, or simply a numpy.ndarray or a'
            ' chainer.Variable itself. But a {} object was given.'.format(
                type(x)))

    if isinstance(chainer_out, (list, tuple)):
        chainer_out = (y.array for y in chainer_out)
    elif isinstance(chainer_out, dict):
        chainer_out = chainer_out[out_key]
        if isinstance(chainer_out, chainer.Variable):
            chainer_out = (chainer_out.array,)
    elif isinstance(chainer_out, chainer.Variable):
        chainer_out = (chainer_out.array,)
    else:
        raise ValueError('Unknown output type: {}'.format(type(chainer_out)))

    onnx_model = onnx_chainer.export(model, x, fn, opset_version=opset_version)
    sess = rt.InferenceSession(onnx_model.SerializeToString())
    input_names = [i.name for i in sess.get_inputs()]

    # To detect unexpected inputs created by exporter, check input names
    # TODO(disktnk): `input_names` got from onnxruntime session includes only
    #                network inputs, does not include internal inputs such as
    #                weight attribute etc. so that need to collect network
    #                inputs from `onnx_model`.
    initialized_graph_input_names = {
        i.name for i in onnx_model.graph.initializer}
    graph_input_names = [i.name for i in onnx_model.graph.input
                         if i.name not in initialized_graph_input_names]
    assert input_names == list(sorted(graph_input_names))

    rt_out = sess.run(
        None, {name: array for name, array in zip(input_names, x_rt)})

    for cy, my in zip(chainer_out, rt_out):
        np.testing.assert_almost_equal(cy, my, decimal=5)
