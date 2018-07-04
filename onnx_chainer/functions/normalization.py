import os

import chainer
from onnx import helper
from onnx import numpy_helper

from onnx_chainer import mapping


def convert_BatchNormalization(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    # Add running_mean and running_var to graph
    running_mean = chainer.Parameter(func.running_mean)
    parameters.append(running_mean)
    input_names.append(str(id(running_mean)))

    running_var = chainer.Parameter(func.running_var)
    parameters.append(running_var)
    input_names.append(str(id(running_var)))

    unique_layer_name = '{}_{}'.format(func.__class__.__name__, str(id(func)))
    output_names += [
        unique_layer_name + '_mean',
        unique_layer_name + '_var',
        unique_layer_name + '_saved_mean',
        unique_layer_name + '_saved_var'
    ]

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        epsilon=func.eps,
        momentum=func.decay,
        spatial=True,
        is_test=not chainer.config.train,
        consumed_inputs=[False, False, False, True, True],
    ),


def convert_FixedBatchNormalization(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    # # Add avg_mean and avg_var to graph
    mean_arr, var_arr = [i.get_variable().array for i in func.inputs[3:]]

    mean_arr_param = chainer.Parameter(mean_arr)
    parameters.append(mean_arr_param)
    input_names[3] = str(id(mean_arr_param))

    var_arr_param = chainer.Parameter(var_arr)
    parameters.append(var_arr_param)
    input_names[4] = str(id(var_arr_param))

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        epsilon=func.eps,
        spatial=True,
        is_test=not chainer.config.train,
        consumed_inputs=[False, False, False, True, True],
    ),


def convert_LocalResponseNormalization(
        func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        alpha=float(func.alpha),
        beta=float(func.beta),
        bias=float(func.k),
        size=int(func.n),
    ),
