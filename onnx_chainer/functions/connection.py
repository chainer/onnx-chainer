import chainer
import numpy as np
from onnx import helper

from onnx_chainer import mapping


def convert_Convolution2DFunction(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    if hasattr(func, 'dy') and hasattr(func, 'dx'):
        node = helper.make_node(
            onnx_op_name, input_names, output_names,
            dilations=(func.dy, func.dx),
            kernel_shape=func.inputs[1].shape[2:],
            # pads: [x1_begin, x2_begin...x1_end, x2_end,...]
            pads=(func.ph, func.pw, func.ph, func.pw),
            strides=(func.sy, func.sx),
        )
    else:
        node = helper.make_node(
            onnx_op_name, input_names, output_names,
            dilations=(1, 1),
            kernel_shape=func.inputs[1].shape[2:],
            pads=(func.ph, func.ph, func.pw, func.pw),
            strides=(func.sy, func.sx),
        )

    return node,


def convert_ConvolutionND(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        kernel_shape=func.inputs[1].shape[2:],
        pads=func.pad,
        strides=func.stride,
    ),


def convert_Deconvolution2DFunction(
        func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        kernel_shape=func.inputs[1].shape[2:],
        output_shape=(func.outh, func.outw),
        pads=(func.ph, func.pw),
        strides=(func.sy, func.sx),
    ),


def convert_DeconvolutionND(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        auto_pad='VALID',
        kernel_shape=func.inputs[1].shape[2:],
        output_shape=func.outs,
        pads=func.pad,
        strides=func.stride,
    ),


def convert_EmbedIDFunction(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    x_index_name, W_name = input_names
    input_names = [W_name, x_index_name]

    if func.ignore_label is not None:
        raise ValueError(
            'Current ONNX doesn\'t support ignore_label for EmbedID.')

    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_LinearFunction(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    if len(func.inputs) == 2:
        batchsize = func.inputs[0].shape[0]
        bias_dim = func.inputs[1].shape[0]
        bias = np.zeros((batchsize, bias_dim), dtype=np.float32)
        bias_param = chainer.Parameter(bias)
        parameters.append(bias_param)
        input_names.append(str(id(bias_param)))
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        alpha=1.0, beta=1.0, transA=0, transB=1),
