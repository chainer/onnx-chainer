import numpy as np
from onnx import helper
from onnx.numpy_helper import from_array

from onnx_chainer.onnx_helper import gensym
from onnx_chainer import onnx_helper


def convert_SoftmaxCrossEntropy(
        func, opset_version, input_names,
        num_outputs, parameters):
    nodes = []
    x, t = input_names
    y_log = gensym()
    th = gensym()
    s0 = gensym()
    sn = gensym()
    sr = gensym()
    depth = gensym()
    zeroone = gensym()

    nodes.append(helper.make_node(
        'LogSoftmax', [x], [y_log]))
    nodes.append(helper.make_node(
        'Constant', [], [depth], value=from_array(
            np.array([5], dtype=np.int32))))  # FIXME
    nodes.append(helper.make_node(
        'Constant', [], [zeroone], value=from_array(
            np.array([0, 1], dtype='f'))))
    nodes.append(helper.make_node(
        'OneHot', [t, depth, zeroone], [th]))
    nodes.append(helper.make_node(
        'Mul', [y_log, th], [s0]))
    nodes.append(helper.make_node(
        'Neg', [s0], [sn]))
    nodes.append(helper.make_node(
        'ReduceSum', [sn], [sr], axes=[1], keepdims=0))
    nodes.append(onnx_helper.make_node(
        'ReduceMean', [sr], num_outputs, axes=[0], keepdims=0))

    return tuple(nodes)
