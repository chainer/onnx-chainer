import sys

import chainer
import numpy as np
from onnx_chainer import onnx_helper


def convert_BatchNormalization(func, opset_version, input_names,
                               num_outputs, parameters):
    x = func.inputs[0].get_variable().data
    mean = chainer.Parameter(x.mean(axis=func.axis))
    parameters.append(mean)
    input_names.append(str(id(mean)))
    var = chainer.Parameter(x.var(axis=func.axis))
    parameters.append(var)
    input_names.append(str(id(var)))

    # TODO(disktnk): ONNX's BatchNormalization operator outputs one required
    # output and four optional outputs. This converter must make 5 values for
    # output and return them.

    # if `use_beta=False`, passed None value to the functions
    if func.inputs[2].get_variable_or_none() is None:
        beta = chainer.Parameter(np.zeros_like(mean, dtype=mean.dtype))
        parameters.append(beta)
        input_names[2] = str(id(beta))
    # `use_gamma=False` is same
    if func.inputs[1].get_variable_or_none() is None:
        gamma = chainer.Parameter(np.ones_like(mean, dtype=mean.dtype))
        parameters.append(gamma)
        input_names[1] = str(id(gamma))

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
                                    input_names, num_outputs, parameters):
    # Add avg_mean and avg_var to graph
    mean_arr, var_arr = [i.get_variable().array for i in func.inputs[3:]]

    mean_arr_param = chainer.Parameter(mean_arr)
    parameters.append(mean_arr_param)
    input_names[3] = str(id(mean_arr_param))

    var_arr_param = chainer.Parameter(var_arr)
    parameters.append(var_arr_param)
    input_names[4] = str(id(var_arr_param))

    # if `use_beta=False`, passed None value to the functions
    if func.inputs[2].get_variable_or_none() is None:
        beta = chainer.Parameter(np.zeros_like(mean_arr, dtype=mean_arr.dtype))
        parameters.append(beta)
        input_names[2] = str(id(beta))
    # `use_gamma=False` is same
    if func.inputs[1].get_variable_or_none() is None:
        gamma = chainer.Parameter(np.ones_like(mean_arr, dtype=mean_arr.dtype))
        parameters.append(gamma)
        input_names[1] = str(id(gamma))

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
                                       input_names, num_outputs, parameters):
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
                        num_outputs, parameters):
    if isinstance(func.axis, tuple) and len(func.axis) != 1:
        raise ValueError(
            'Normalization along with multiple axes ({}) are not supported in '
            'the ONNX\'s LpNormalization operator.'.format(func.axis))
    if abs(func.eps - 1e-5) > sys.float_info.epsilon:
        # default value of F.normaize eps is 1e-5
        raise ValueError(
            '\'eps\' is not supported in the ONNX\'s LpNormalization operator,'
            ' so that ONNX-Chainer does not accept custom values for \'eps\' '
            '({})'.format(func.eps))
    if opset_version == 1:
        return onnx_helper.make_node(
            'LpNormalization', input_names, num_outputs,
            axis=int(func.axis[0]),
            p=2,
        ),
