import chainer
from chainer.functions.math import basic_math
import numpy as np
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
    value = np.broadcast_to(value, func.inputs[0].shape)
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
    onnx_op_name = mapping.operators[func.__class__.__name__]

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


def convert_Maximum(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Minimum(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Sqrt(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Sum(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axes=func.axis,
        keepdims=func.keepdims,
    ),


def convert_LinearInterpolate(func, input_names, output_names, parameters):
    typ = func.inputs[0].dtype if isinstance(
        func.inputs[0].dtype, np.dtype) else np.dtype(func.inputs[0].dtype)

    one = chainer.Parameter(np.array(1, dtype=typ))
    #one = chainer.Parameter(np.ones(dtype=typ, shape=[1]*len(func.inputs[0].shape)))
    parameters.append(one)

    n1_out_name = gensym()
    n2_out_name = gensym()
    n3_out_name = gensym()
    n4_out_name = gensym()

    n1 = helper.make_node("Neg", [input_names[0]], [n1_out_name])
    n2 = helper.make_node("Add", [n1_out_name, str(id(one))], [n2_out_name])
    n3 = helper.make_node("Mul", [input_names[0], input_names[1]], [n3_out_name])
    n4 = helper.make_node("Mul", [n2_out_name, input_names[2]], [n4_out_name])
    n5 = helper.make_node("Add", [n3_out_name, n4_out_name], [output_names[0]])

    return n5, n4, n3, n2, n1


dummy_objects = []


def gensym():
    o = object()
    dummy_objects.append(o)
    return str(id(o))
