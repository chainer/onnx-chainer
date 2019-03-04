import os
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

TEST_OUT_DIR = 'out'


def check_model_expect(test_path, input_names=None):
    if not ONNXRUNTIME_AVAILABLE:
        raise ImportError('check_output requires onnxruntime.')

    model_path = os.path.join(test_path, 'model.onnx')
    with open(model_path, 'rb') as f:
        onnx_model = onnx.load_model(f)

    test_data_sets = sorted([
        p for p in os.listdir(test_path) if p.startswith('test_data_set_')])
    for test_data in test_data_sets:
        test_data_path = os.path.join(test_path, test_data)
        assert os.path.isdir(test_data_path)

        file_list = sorted(os.listdir(test_data_path))
        inputs, outputs = [], []

        for file_name in file_list:
            if not file_name.endswith('.pb'):
                continue
            path = os.path.join(test_path, test_data, file_name)
            with open(path, 'rb') as f:
                array = onnx.numpy_helper.to_array(onnx.load_tensor(path))
            if file_name.startswith('input_'):
                inputs.append(array)
            else:
                outputs.append(array)

        sess = rt.InferenceSession(onnx_model.SerializeToString())

        # To detect unexpected inputs created by exporter, check input names
        # TODO(disktnk): `input_names` got from onnxruntime session includes
        # only network inputs, does not include internal inputs such as weight
        # attribute etc. so that need to collect network inputs from
        # `onnx_model`.
        rt_input_names = [i.name for i in sess.get_inputs()]
        if input_names is not None:
            assert list(sorted(input_names)) == list(sorted(rt_input_names))

        rt_out = sess.run(
            None, {name: array for name, array in zip(rt_input_names, inputs)})
        for cy, my in zip(outputs, rt_out):
            np.testing.assert_allclose(cy, my, rtol=1e-5, atol=1e-5)


def check_output(model, x, filename, out_keys=None, opset_version=None):
    model.xp.random.seed(42)

    os.makedirs(TEST_OUT_DIR, exist_ok=True)
    filename = os.path.join(TEST_OUT_DIR, filename)

    if opset_version is None:
        opset_version = onnx.defs.onnx_opset_version()
    if not ONNXRUNTIME_AVAILABLE:
        raise ImportError('check_output requires onnxruntime.')

    chainer.config.train = False

    # Forward computation
    if isinstance(x, (list, tuple)):
        for i in x:
            assert isinstance(i,
                              chainer.get_array_types() + (chainer.Variable,))
        chainer_out = model(*x)
        x_rt = tuple(
            _x.array if isinstance(_x, chainer.Variable) else _x for _x in x)
    elif isinstance(x, dict):
        chainer_out = model(**x)
        x_rt = tuple(_x.array if isinstance(_x, chainer.Variable) else _x
                     for _, _x in x.items())
    elif isinstance(x, chainer.get_array_types()):
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

    rt_out_keys = None
    if isinstance(chainer_out, (list, tuple)):
        chainer_out = tuple(y.array for y in chainer_out)
        if out_keys is not None:
            assert len(out_keys) == len(chainer_out)
            rt_out_keys = out_keys
    elif isinstance(chainer_out, dict):
        if len(out_keys) > 1:
            rt_out_keys = out_keys
        chainer_outs = [chainer_out[k] for k in out_keys]
        chainer_out = tuple(out.array if isinstance(out, chainer.Variable) else
                            out for out in chainer_outs)
    elif isinstance(chainer_out, chainer.Variable):
        chainer_out = (chainer_out.array,)
    else:
        raise ValueError('Unknown output type: {}'.format(type(chainer_out)))

    x_rt = tuple(chainer.cuda.to_cpu(x) for x in x_rt)
    chainer_out = tuple(chainer.cuda.to_cpu(x) for x in chainer_out)

    onnx_model = onnx_chainer.export(model, x, filename,
                                     opset_version=opset_version)
    check_all_connected_from_inputs(onnx_model)

    sess = rt.InferenceSession(onnx_model.SerializeToString())
    input_names = [i.name for i in sess.get_inputs()]

    # To detect unexpected inputs created by exporter, check input names
    # TODO(disktnk): `input_names` got from onnxruntime session includes only
    # network inputs, does not include internal inputs such as weight attribute
    # etc. so that need to collect network inputs from `onnx_model`.
    graph_input_names = _get_graph_input_names(onnx_model)
    assert list(sorted(input_names)) == list(sorted(graph_input_names))

    rt_out = sess.run(
        rt_out_keys, {name: array for name, array in zip(input_names, x_rt)})

    for cy, my in zip(chainer_out, rt_out):
        np.testing.assert_allclose(cy, my, rtol=1e-5, atol=1e-5)


def check_all_connected_from_inputs(onnx_model):
    edge_names = set(_get_graph_input_names(onnx_model))
    # Nodes which are not connected from the network inputs.
    orphan_nodes = []
    for node in onnx_model.graph.node:
        if not edge_names.intersection(node.input):
            orphan_nodes.append(node)
        for output_name in node.output:
            edge_names.add(output_name)
    assert not(orphan_nodes), '{}'.format(orphan_nodes)


def _get_graph_input_names(onnx_model):
    initialized_graph_input_names = {
        i.name for i in onnx_model.graph.initializer}
    return [i.name for i in onnx_model.graph.input if i.name not in
            initialized_graph_input_names]
