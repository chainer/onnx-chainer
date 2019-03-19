import chainer
import numpy as np

from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


@support((1, 6, 7))
def convert_Add(func, opset_version, input_names, num_outputs,
                context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Add', input_names, num_outputs, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Add', input_names, num_outputs),


@support((1, 6, 7))
def convert_AddConstant(func, opset_version, input_names,
                        num_outputs, context, parameters):
    value = np.array(func.value, dtype=func.inputs[0].dtype)
    value_param = chainer.Parameter(value)
    parameters.append(value_param)
    input_names.append(context.get_name(value_param))

    if opset_version == 1:
        return onnx_helper.make_node(
            'Add', input_names, num_outputs, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Add', input_names, num_outputs),


@support((1, 6, 7))
def convert_Sub(func, opset_version, input_names, num_outputs,
                context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Sub', input_names, num_outputs, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Sub', input_names, num_outputs),


@support((1, 6, 7))
def convert_Mul(func, opset_version, input_names, num_outputs,
                context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Mul', input_names, num_outputs, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Mul', input_names, num_outputs),


@support((1, 6, 7))
def convert_MulConstant(func, opset_version, input_names,
                        num_outputs, context, parameters):
    value = np.array(func.value, dtype=func.inputs[0].dtype)
    value_param = chainer.Parameter(value)
    parameters.append(value_param)
    input_names.append(context.get_name(value_param))

    if opset_version == 1:
        return onnx_helper.make_node(
            'Mul', input_names, num_outputs, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Mul', input_names, num_outputs),


@support((1, 6))
def convert_Neg(func, opset_version, input_names, num_outputs,
                context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Neg', input_names, num_outputs, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Neg', input_names, num_outputs),


@support((1, 6, 7))
def convert_Div(func, opset_version, input_names, num_outputs,
                context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Div', input_names, num_outputs, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node('Div', input_names, num_outputs),


@support((1, 6))
def convert_Absolute(func, opset_version, input_names,
                     num_outputs, context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Abs', input_names, num_outputs, consumed_inputs=[1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Abs', input_names, num_outputs),


@support((1, 7))
def convert_PowVarConst(func, opset_version, input_names,
                        num_outputs, context, parameters):
    value = np.array(func.value, dtype=func.inputs[0].dtype)
    value_param = chainer.Parameter(value)
    parameters.append(value_param)
    input_names.append(context.get_name(value_param))

    if opset_version == 1 or opset_version == 7:
        return onnx_helper.make_node('Pow', input_names, num_outputs),


@support((1, 6))
def convert_Clip(func, opset_version, input_names, num_outputs,
                 context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Clip', input_names, num_outputs,
            max=func.x_max,
            min=func.x_min,
            consumed_inputs=[1]
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'Clip', input_names, num_outputs,
            max=func.x_max,
            min=func.x_min,
        ),


@support((1, 6))
def convert_Exp(func, opset_version, input_names, num_outputs,
                context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Exp', input_names, num_outputs, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Exp', input_names, num_outputs),


def convert_Identity(func, opset_version, input_names,
                     num_outputs, context, parameters):
    return onnx_helper.make_node('Identity', input_names, num_outputs),


@support((1, 6, 7))
def convert_MatMul(func, opset_version, input_names,
                   num_outputs, context, parameters):
    bias_shape = (
        func.inputs[0].shape[-1] if func.transa else func.inputs[0].shape[-2],
        func.inputs[1].shape[-2] if func.transb else func.inputs[1].shape[-1]
    )
    bias_tensor = np.zeros(bias_shape, dtype=func.inputs[0].dtype)
    bias_param = chainer.Parameter(bias_tensor)
    parameters.append(bias_param)
    input_names.append(context.get_name(bias_param))

    return onnx_helper.make_node(
        'Gemm', input_names, num_outputs,
        transA=func.transa,
        transB=func.transb
    ),


@support((1, 6, 8))
def convert_Maximum(func, opset_version, input_names,
                    num_outputs, context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Max', input_names, num_outputs, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 8:
        return onnx_helper.make_node('Max', input_names, num_outputs),


@support((1, 6, 8))
def convert_Minimum(func, opset_version, input_names,
                    num_outputs, context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Min', input_names, num_outputs, consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 8:
        return onnx_helper.make_node('Min', input_names, num_outputs),


@support((1, 6))
def convert_Sqrt(func, opset_version, input_names, num_outputs,
                 context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Sqrt', input_names, num_outputs, consumed_inputs=[1, 1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Sqrt', input_names, num_outputs),


def convert_LogSumExp(func, opset_version, input_names,
                      num_outputs, context, parameters):
    # Use keepdims=False by default
    # since the chainer does not support keepdims option
    kwargs = {'keepdims': False}
    if hasattr(func, 'keepdims'):
        kwargs['keepdims'] = func.keepdims
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceLogSumExp', input_names, num_outputs, **kwargs),


def convert_Max(func, opset_version, input_names, num_outputs,
                context, parameters):
    kwargs = {'keepdims': func.keepdims}
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceMax', input_names, num_outputs, **kwargs),


def convert_Mean(func, opset_version, input_names, num_outputs,
                 context, parameters):
    kwargs = {'keepdims': func.keepdims}
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceMean', input_names, num_outputs, **kwargs),


def convert_Min(func, opset_version, input_names, num_outputs,
                context, parameters):
    kwargs = {'keepdims': func.keepdims}
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceMin', input_names, num_outputs, **kwargs),


def convert_Prod(func, opset_version, input_names, num_outputs,
                 context, parameters):
    kwargs = {'keepdims': func.keepdims}
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceProd', input_names, num_outputs, **kwargs),


def convert_Sum(func, opset_version, input_names, num_outputs,
                context, parameters):
    kwargs = {'keepdims': func.keepdims}
    if func.axis is not None:
        kwargs['axes'] = func.axis
    return onnx_helper.make_node(
        'ReduceSum', input_names, num_outputs, **kwargs),


@support((1, 6, 7))
def convert_LinearInterpolate(func, opset_version, input_names,
                              num_outputs, context, parameters):
    typ = func.inputs[0].dtype if isinstance(
        func.inputs[0].dtype, np.dtype) else np.dtype(func.inputs[0].dtype)

    one = chainer.Parameter(np.array(1, dtype=typ))
    parameters.append(one)

    kwargs = {'consumed_inputs': [1, 1]} if opset_version == 1 else {}
    kwargs2 = {} if opset_version >= 7 else {'broadcast': 1}

    gb = onnx_helper.GraphBuilder()
    p, x, y = input_names
    n1 = gb.op('Sub', [context.get_name(one), p], **kwargs, **kwargs2)
    n2 = gb.op('Mul', [p, x], **kwargs)
    n3 = gb.op('Mul', [n1, y], **kwargs)
    gb.op('Add', [n2, n3], num_outputs, **kwargs)

    return gb.nodes()


@support((1, 6, 7))
def convert_Square(func, opset_version, input_names,
                   num_outputs, context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Mul', [input_names[0], input_names[0]], num_outputs,
            consumed_inputs=[1, 1]),
    elif opset_version == 6 or opset_version == 7:
        return onnx_helper.make_node(
            'Mul', [input_names[0], input_names[0]], num_outputs),


@support((8,))
def convert_BroadcastTo(func, opset_version, input_names,
                        num_outputs, context, parameters):
    shape = np.array(func._shape)
    parameters.append(shape)
    input_names.append(context.get_name(shape))
    return onnx_helper.make_node('Expand', input_names, num_outputs),
