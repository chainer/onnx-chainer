import numpy as np

import chainer
from chainer.functions.math import basic_math
from onnx import helper
from onnx import numpy_helper
from onnx_chainer import mapping


def convert_Add(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_AddConstant(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    value = np.asarray([func.value], dtype=func.inputs[0].dtype)
    value = np.broadcast_to(value, func.inputs[0].shape)
    value_param = chainer.Parameter(value)
    parameters.append(value_param)
    input_names.append(str(id(value_param)))

    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Sub(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Mul(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Neg(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Div(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Absolute(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(onnx_op_name, input_names, output_names, consumed_inputs=[1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_PowVarConst(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    value = np.asarray([func.value], dtype=func.inputs[0].dtype)
    value = np.broadcast_to(value, func.inputs[0].shape)
    value_param = chainer.Parameter(value)
    parameters.append(value_param)
    input_names.append(str(id(value_param)))

    if opset_version == 1 or opset_version == 7:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Clip(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            max=func.x_max,
            min=func.x_min,
            consumed_inputs=[1]
        ),
    elif opset_version == 6:
        return helper.make_node(
            onnx_op_name, input_names, output_names,
            max=func.x_max,
            min=func.x_min,
        ),


def convert_Exp(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Identity(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_MatMul(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    bias_shape = (
        func.inputs[0].shape[-1] if func.transa else func.inputs[0].shape[-2],
        func.inputs[1].shape[-2] if func.transb else func.inputs[1].shape[-1]
    )
    bias_tensor = np.zeros(bias_shape, dtype=np.float32)
    bias_param = chainer.Parameter(bias_tensor)
    parameters.append(bias_param)
    input_names.append(str(id(bias_param)))

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        transA=func.transa,
        transB=func.transb
    ),


def convert_Maximum(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Minimum(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Sqrt(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    if opset_version == 1:
        return helper.make_node(onnx_op_name, input_names, output_names, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Sum(func, onnx_op_name, opset_version, input_names, output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axes=func.axis,
        keepdims=func.keepdims,
    ),
