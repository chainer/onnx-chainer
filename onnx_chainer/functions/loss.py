import numpy as np
from onnx import helper
from onnx.numpy_helper import from_array


dummy_objects = []


def create_new_name():
    dummy = object()
    dummy_objects.append(dummy)
    return str(id(dummy))


def convert_SoftmaxCrossEntropy(
        func, onnx_op_name, opset_version, input_names,
        output_names, parameters):
    nodes = []
    x, t = input_names
    y_log = create_new_name()
    th = create_new_name()
    s0 = create_new_name()
    sn = create_new_name()
    sr = create_new_name()
    depth = create_new_name()
    zeroone = create_new_name()

    nodes.append(helper.make_node(
        'LogSoftmax', [x], [y_log]))
    nodes.append(helper.make_node(
        'Constant', [], [depth], value=from_array(np.array([5], dtype=np.int32))))  # FIXME
    nodes.append(helper.make_node(
        'Constant', [], [zeroone], value=from_array(np.array([0, 1], dtype='f'))))
    nodes.append(helper.make_node(
        'OneHot', [t, depth, zeroone], [th]))
    nodes.append(helper.make_node(
        'Mul', [y_log, th], [s0]))
    nodes.append(helper.make_node(
        'Neg', [s0], [sn]))
    nodes.append(helper.make_node(
        'ReduceSum', [sn], [sr], axes=[1], keepdims=0))
    nodes.append(helper.make_node(
        'ReduceMean', [sr], output_names, axes=[0], keepdims=0))

    return tuple(reversed(nodes))
