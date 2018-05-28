from onnx import helper

from onnx_chainer import mapping


def convert_AveragePooling2D(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    gpool = func.inputs[0].shape[2:] == (func.kh, func.kw)

    if not gpool:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=(func.kh, func.kw),
            pads=(func.ph, func.pw),
            strides=(func.sy, func.sx)
        ),
    else:
        return helper.make_node(
            'Global' + onnx_op_name, input_names, output_names),


def convert_AveragePoolingND(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    gpool = func.inputs[0].shape[2:] == func.ksize

    if not gpool:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            auto_pad='VALID',
            kernel_shape=func.ksize,
            pads=func.pad,
            strides=func.stride
        ),
    else:
        return helper.make_node(
            'Global' + onnx_op_name, input_names, output_names),


def convert_MaxPooling2D(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    gpool = func.inputs[0].shape[2:] == (func.kh, func.kw)

    if not gpool:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=(func.kh, func.kw),
            pads=(func.ph, func.pw),
            strides=(func.sy, func.sx)
        ),
    else:
        return helper.make_node(
            'Global' + onnx_op_name, input_names, output_names),


def convert_MaxPoolingND(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    gpool = func.inputs[0].shape[2:] == func.ksize

    if not gpool:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            auto_pad='VALID',
            kernel_shape=func.ksize,
            pads=func.pad,
            strides=func.stride
        ),
    else:
        return helper.make_node(
            'Global' + onnx_op_name, input_names, output_names),
