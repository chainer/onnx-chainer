from onnx import helper

from onnx_chainer import mapping


def convert_AveragePooling2D(func, onnx_op_name, input_names, output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        kernel_shape=(func.kh, func.kw),
        pads=(func.ph, func.pw, func.ph, func.pw),
        strides=(func.sy, func.sx)
    ),


def convert_AveragePoolingND(func, onnx_op_name, input_names, output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        kernel_shape=func.ksize,
        pads=func.pad,
        strides=func.stride
    ),


def convert_MaxPooling2D(func, onnx_op_name, input_names, output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        kernel_shape=(func.kh, func.kw),
        pads=(func.ph, func.pw, func.ph, func.pw),
        strides=(func.sy, func.sx)
    ),


def convert_MaxPoolingND(func, onnx_op_name, input_names, output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        kernel_shape=func.ksize,
        pads=func.pad,
        strides=func.stride
    ),
