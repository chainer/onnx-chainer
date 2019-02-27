import numpy as np
from onnx import helper
from onnx.numpy_helper import from_array

from onnx_chainer import onnx_helper
from onnx_chainer.onnx_helper import gensym


def convert_SoftmaxCrossEntropy(
        func, opset_version, input_names,
        num_outputs, parameters):
    # obtain input variable
    x_var, t_var = func.get_retained_inputs()
    if len(x_var.shape) != 2:
        raise NotImplementedError(
            'ONNX-Chainer currently handles SoftmaxCrossEntropy only when '
            'the dimension of input variable x is exactly two.')
    if np.any(t_var.array == func.ignore_label):
        raise NotImplementedError(
            'ONNX-Chainer currently handles SoftmaxCrossEntropy only when '
            'ignore_label is not used in input variable t.')
    if (not func.normalize) or (func.class_weight is not None) or\
       (func.ignore_label != -1) or (func.reduce != 'mean'):
        raise NotImplementedError(
            'ONNX-Chainer currently handles SoftmaxCrossEntropy only when '
            'argument parameters are default setting.')

    # create intermediate values
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
            np.array([x_var.shape[1]], dtype=np.int32))))
    nodes.append(helper.make_node(
        'Constant', [], [zeroone], value=from_array(
            np.array([0, 1], dtype=x_var.dtype))))
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
