import numpy as np
import chainer
from onnx import helper
from onnx import numpy_helper
from onnx_chainer import mapping


def convert_Add(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Sub(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Mul(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Neg(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Div(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Absolute(
        func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_PowVarConst(
        func, input_names, output_names, parameters):
    value = np.asarray([func.value], dtype=func.inputs[0].dtype)
    value_param = chainer.Parameter(value)
    parameters.append(value_param)
    input_names.append(str(id(value_param)))
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Clip(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        max=func.x_max,
        min=func.x_min,
    ),


def convert_Exp(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Identity(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_MatMul(func, input_names, output_names, parameters):
    if func.transa or func.transb:
        raise ValueError(
            'Current ONNX doesn\'t support transpose options for matmul ops.')
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Maximum(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Minimum(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Sqrt(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_SquaredDifference(func, input_names, output_names, parameters):
    sub_output_names = [input_names[0] + '_sub']
    onnx_op_name = mapping.operators[basic_math.Sub.__name__]
    sub_node = helper.make_node(onnx_op_name, input_names, sub_output_names)

    pow_node, = convert_PowVarConst(
        basic_math.PowVarConst(2), sub_output_names, output_names,
        parameters)

    return list(reversed([sub_node, pow_node]))


def convert_Sum(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axes=func.axis,
        keepdims=func.keepdims,
    ),
