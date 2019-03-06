import sys

import chainer
import numpy as np

from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


@support((1, 6, 7))
def convert_BatchNormalization(func, opset_version, input_names,
                               num_outputs, context, parameters):
    if len(func.inputs) <= 3:
        # expect this `func` is F.batch_normalization
        x = func.inputs[0].get_variable().array
        mean = chainer.Parameter(x.mean(axis=func.axis))
        input_names.append(context.get_name(mean))
        var = chainer.Parameter(x.var(axis=func.axis))
        input_names.append(context.get_name(var))
    else:
        # expect this `func` is F.fixed_batch_normalization
        mean = chainer.Parameter(func.inputs[3].get_variable().array)
        input_names[3] = context.get_name(mean)
        var = chainer.Parameter(func.inputs[4].get_variable().array)
        input_names[4] = context.get_name(var)
    parameters.append(mean)
    parameters.append(var)

    momentum = getattr(func, 'decay', 0.)

    # if `use_beta=False`, passed None value to the functions
    if func.inputs[2].get_variable_or_none() is None:
        beta = chainer.Parameter(np.zeros_like(mean, dtype=mean.dtype))
        parameters.append(beta)
        input_names[2] = context.get_name(beta)
    # `use_gamma=False` is same
    if func.inputs[1].get_variable_or_none() is None:
        gamma = chainer.Parameter(np.ones_like(mean, dtype=mean.dtype))
        parameters.append(gamma)
        input_names[1] = context.get_name(gamma)

    # TODO(disktnk): On definition of ONNX's BatchNormalization operator,
    # outputs one required output and four optional outputs. This converter
    # must make 5 values for output and return them.

    if opset_version == 1:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, num_outputs,
            epsilon=func.eps,
            momentum=momentum,
            is_test=not chainer.config.train,
            consumed_inputs=[False, False, False, True, True],
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, num_outputs,
            epsilon=func.eps,
            momentum=momentum,
            is_test=not chainer.config.train,
        ),
    elif opset_version == 7:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, num_outputs,
            epsilon=func.eps,
            momentum=momentum,
        ),


@support((1, 6, 7))
def convert_FixedBatchNormalization(func, opset_version,
                                    input_names, num_outputs, context,
                                    parameters):
    return convert_BatchNormalization(
        func, opset_version, input_names, num_outputs, context, parameters)


def convert_LocalResponseNormalization(func, opset_version,
                                       input_names, num_outputs, context,
                                       parameters):
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
            ' so that ONNX-Chainer does not accept custom values for \'eps\' '
            '({})'.format(func.eps))

    return onnx_helper.make_node(
        'LpNormalization', input_names, num_outputs,
        axis=int(func.axis[0]),
        p=2,
    ),
