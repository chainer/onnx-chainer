import chainer
import numpy as np

from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


def convert_Convolution2DFunction(func, opset_version,
                                  input_names, num_outputs, context,
                                  parameters):
    if hasattr(func, 'dy') and hasattr(func, 'dx'):
        node = onnx_helper.make_node(
            'Conv', input_names, num_outputs,
            dilations=(func.dy, func.dx),
            kernel_shape=func.inputs[1].shape[2:],
            # pads: [x1_begin, x2_begin...x1_end, x2_end,...]
            pads=(func.ph, func.pw, func.ph, func.pw),
            strides=(func.sy, func.sx),
            group=func.groups,
        )
    else:
        node = onnx_helper.make_node(
            'Conv', input_names, num_outputs,
            dilations=(1, 1),
            kernel_shape=func.inputs[1].shape[2:],
            pads=(func.ph, func.pw, func.ph, func.pw),
            strides=(func.sy, func.sx),
            group=func.groups,
        )
    return node,


def convert_ConvolutionND(func, opset_version, input_names,
                          num_outputs, context,
                          parameters):
    pad = []
    x_ndim = len(func.inputs[0].shape)
    w_ndim = len(func.inputs[1].shape)
    for _ in range(x_ndim - w_ndim):
        pad.append(0)
    for p in func.pad:
        pad.append(p)
    pad = pad * 2

    return onnx_helper.make_node(
        'Conv', input_names, num_outputs,
        kernel_shape=func.inputs[1].shape[2:],
        pads=pad,
        strides=func.stride,
    ),


def convert_Deconvolution2DFunction(func, opset_version,
                                    input_names, num_outputs, context,
                                    parameters):
    return onnx_helper.make_node(
        'ConvTranspose', input_names, num_outputs,
        kernel_shape=func.inputs[1].shape[2:],
        output_shape=(func.outh, func.outw),
        # pads: [x1_begin, x2_begin...x1_end, x2_end,...]
        pads=(func.ph, func.pw, func.ph, func.pw),
        strides=(func.sy, func.sx),
    ),


def convert_DeconvolutionND(func, opset_version, input_names,
                            num_outputs, context, parameters):
    pad = []
    x_ndim = len(func.inputs[0].shape)
    w_ndim = len(func.inputs[1].shape)
    for _ in range(x_ndim - w_ndim):
        pad.append(0)
    for p in func.pad:
        pad.append(p)
    pad = pad * 2

    return onnx_helper.make_node(
        'ConvTranspose', input_names, num_outputs,
        kernel_shape=func.inputs[1].shape[2:],
        output_shape=func.outs,
        pads=pad,
        strides=func.stride,
    ),


def convert_EmbedIDFunction(func, opset_version, input_names,
                            num_outputs, context, parameters):
    x_index_name, W_name = input_names
    input_names = [W_name, x_index_name]

    if func.ignore_label is not None:
        raise ValueError(
            'Current ONNX doesn\'t support ignore_label for EmbedID.')

    return onnx_helper.make_node(
        'Gather', input_names, num_outputs, axis=0),


@support((1, 6, 7))
def convert_LinearFunction(func, opset_version, input_names,
                           num_outputs, context, parameters):
    # When the func has bias
    if len(func.inputs) == 2:
        bias_dim = func.inputs[1].shape[0]
        bias = np.zeros((bias_dim,), dtype=func.inputs[0].dtype)
        bias_param = chainer.Parameter(bias)
        parameters.append(bias_param)
        input_names.append(context.get_name(bias_param))

    if opset_version == 1 or opset_version == 6:
        return onnx_helper.make_node(
            'Gemm', input_names, num_outputs,
            alpha=1.0, beta=1.0, broadcast=1, transA=0, transB=1),
    elif opset_version == 7:
        return onnx_helper.make_node(
            'Gemm', input_names, num_outputs,
            alpha=1.0, beta=1.0, transA=0, transB=1),
