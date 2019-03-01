import sys

import chainer
import numpy as np
from onnx_chainer import onnx_helper


def convert_BatchNormalization(func, opset_version, input_names,
                               num_outputs, context, parameters):
    # Add running_mean and running_var to graph
    running_mean = chainer.Parameter(func.running_mean)
    parameters.append(running_mean)
    input_names.append(context.get_name(running_mean))

    running_var = chainer.Parameter(func.running_var)
    parameters.append(running_var)
    input_names.append(context.get_name(running_var))

    unique_layer_name = '{}_{}'.format(func.__class__.__name__,
                                       context.get_name(func))
    num_outputs += [
        unique_layer_name + '_mean',
        unique_layer_name + '_var',
        unique_layer_name + '_saved_mean',
        unique_layer_name + '_saved_var'
    ]

    if opset_version == 1:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, num_outputs,
            epsilon=func.eps,
            momentum=func.decay,
            is_test=not chainer.config.train,
            consumed_inputs=[False, False, False, True, True],
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, num_outputs,
            epsilon=func.eps,
            momentum=func.decay,
            is_test=not chainer.config.train,
        ),
    elif opset_version == 7:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, num_outputs,
            epsilon=func.eps,
            momentum=func.decay,
        ),


def convert_FixedBatchNormalization(func, opset_version,
                                    input_names, num_outputs, context,
                                    parameters):
    # Add avg_mean and avg_var to graph
    mean_arr, var_arr = [i.get_variable().array for i in func.inputs[3:]]

    mean_arr_param = chainer.Parameter(mean_arr)
    parameters.append(mean_arr_param)
    input_names[3] = context.get_name(mean_arr_param)

    var_arr_param = chainer.Parameter(var_arr)
    parameters.append(var_arr_param)
    input_names[4] = context.get_name(var_arr_param)

    # if `use_beta=False`, passed None value to the functions
    if func.inputs[2].get_variable_or_none() is None:
        beta = chainer.Parameter(np.zeros_like(mean_arr, dtype=mean_arr.dtype))
        parameters.append(beta)
        input_names[2] = context.get_name(beta)
    # `use_gamma=False` is same
    if func.inputs[1].get_variable_or_none() is None:
        gamma = chainer.Parameter(np.ones_like(mean_arr, dtype=mean_arr.dtype))
        parameters.append(gamma)
        input_names[1] = context.get_name(gamma)

    if opset_version == 1:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, num_outputs,
            epsilon=func.eps,
            momentum=0.,
            is_test=not chainer.config.train,
            consumed_inputs=[False, False, False, True, True],
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, num_outputs,
            epsilon=func.eps,
            momentum=0.,
            is_test=not chainer.config.train,
        ),
    elif opset_version == 7:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, num_outputs,
            epsilon=func.eps,
            momentum=0.,
        ),


def convert_LocalResponseNormalization(func, opset_version,
                                       input_names, num_outputs, context,
                                       parameters):
    if opset_version == 1:
        size = int(func.n)
        return onnx_helper.make_node(
            'LRN', input_names, num_outputs,
            alpha=float(func.alpha) * size,
            beta=float(func.beta),
            bias=float(func.k),
            size=size,
        ),


def convert_NormalizeL2(func, opset_version, input_names,
                        num_outputs, context, parameters):
    if isinstance(func.axis, tuple) and len(func.axis) != 1:
        raise ValueError(
            'Normalization along with multiple axes ({}) are not supported in '
            'the ONNX\'s LpNormalization operator.'.format(func.axis))
    if abs(func.eps - 1e-5) > sys.float_info.epsilon:
        # default value of F.normaize eps is 1e-5
        raise ValueError(
            '\'eps\' is not supported in the ONNX\'s LpNormalization operator,'
            ' so that ONNX-Chainer does not accepct custom values for \'eps\' '
            '({})'.format(func.eps))
    if opset_version == 1:
        return onnx_helper.make_node(
            'LpNormalization', input_names, num_outputs,
            axis=int(func.axis[0]),
            p=2,
        ),
