from onnx import helper


def convert_AveragePooling2D(
        func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1 or opset_version == 7:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=(func.kh, func.kw),
            pads=(func.ph, func.pw, func.ph, func.pw),
            strides=(func.sy, func.sx),
            count_include_pad=1,
        ),


def convert_AveragePoolingND(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    pad = list(func.pad) * 2
    if opset_version == 1 or opset_version == 7:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=func.ksize,
            pads=pad,
            strides=func.stride,
            count_include_pad=1,
        ),


def convert_MaxPooling2D(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=(func.kh, func.kw),
            pads=(func.ph, func.pw, func.ph, func.pw),
            strides=(func.sy, func.sx)
        ),
    elif opset_version == 8:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=(func.kh, func.kw),
            pads=(func.ph, func.pw, func.ph, func.pw),
            strides=(func.sy, func.sx),
            storage_order=0,  # row major
        ),


def convert_MaxPoolingND(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    pad = list(func.pad) * 2
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=func.ksize,
            pads=pad,
            strides=func.stride
        ),
    elif opset_version == 8:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            kernel_shape=func.ksize,
            pads=pad,
            strides=func.stride,
            storage_order=0,  # row major
        ),

