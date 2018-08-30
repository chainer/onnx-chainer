from onnx import helper


def convert_AveragePooling2D(
        func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1 or opset_version == 7:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=(func.kh, func.kw),
            pads=(func.ph, func.pw, func.ph, func.pw),
            strides=(func.sy, func.sx)
        ),


def convert_AveragePoolingND(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    pad = []
    x_ndim = len(func.inputs[0].shape)
    k_ndim = len(func.ksize)
    for _ in range(x_ndim - k_ndim):
        pad.append(0)
    for p in func.pad:
        pad.append(p)
    pad = pad * 2
    if opset_version == 1 or opset_version == 7:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=func.ksize,
            pads=pad,
            strides=func.stride
        ),


def convert_MaxPooling2D(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=(func.kh, func.kw),
            pads=(func.ph, func.pw, func.ph, func.pw),
            strides=(func.sy, func.sx)
        ),


def convert_MaxPoolingND(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    pad = []
    x_ndim = len(func.inputs[0].shape)
    k_ndim = len(func.ksize)
    for _ in range(x_ndim - k_ndim):
        pad.append(0)
    for p in func.pad:
        pad.append(p)
    pad = pad * 2

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=func.ksize,
            pads=pad,
            strides=func.stride
        ),
