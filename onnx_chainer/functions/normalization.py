import sys

import chainer
import numpy as np

from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


@support((1, 6, 7))
def convert_BatchNormalization(
        func, opset_version, input_names, output_names, context):
    is_fixed_bn = len(func.inputs) > 3

    # NOTE(disktnk):
    # if `use_beta=False`, beta_param is None, `use_gamma=False` is same.
    beta_param = func.inputs[2].get_variable_or_none()
    gamma_param = func.inputs[1].get_variable_or_none()
    namedlink = context.get_link(beta_param) or context.get_link(gamma_param)

    if namedlink is not None:
        prefix, link = namedlink
        if is_fixed_bn:
            mean = link.avg_mean
            var = link.avg_var
        else:
            # on train mode, avg_mean would be updated, so make them from x
            x = func.inputs[0].get_variable().array
            mean = x.mean(axis=func.axis)
            var = x.var(axis=func.axis)
    else:
        prefix = None
        if is_fixed_bn:
            mean = func.inputs[3].get_variable().array
            var = func.inputs[4].get_variable().array
        else:
            x = func.inputs[0].get_variable().array
            mean = x.mean(axis=func.axis)
            var = x.var(axis=func.axis)

    def add_param(v, suffix):
        if prefix is None:
            return context.add_param(v, suffix)
        else:
            return context.add_param(
                v, '{}_{}'.format(prefix, suffix), use_original_name=True)

    maen_name = add_param(mean, 'avg_mean')
    var_name = add_param(var, 'avg_var')
    if is_fixed_bn:
        input_names[3] = maen_name
        input_names[4] = var_name
    else:
        input_names.extend([maen_name, var_name])

    if beta_param is None:
        beta_name = add_param(np.zeros_like(mean, dtype=mean.dtype), 'beta')
        input_names[2] = beta_name
    if gamma_param is None:
        gamma_name = add_param(np.ones_like(mean, dtype=mean.dtype), 'gamma')
        input_names[1] = gamma_name

    momentum = getattr(func, 'decay', 0.)

    # TODO(disktnk): On definition of ONNX's BatchNormalization operator,
    # outputs one required output and four optional outputs. This converter
    # must make 5 values for output and return them.

    if opset_version == 1:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, output_names,
            epsilon=func.eps,
            momentum=momentum,
            is_test=not chainer.config.train,
            consumed_inputs=[False, False, False, True, True],
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, output_names,
            epsilon=func.eps,
            momentum=momentum,
            is_test=not chainer.config.train,
        ),
    elif opset_version == 7:
        return onnx_helper.make_node(
            'BatchNormalization', input_names, output_names,
            epsilon=func.eps,
            momentum=momentum,
        ),


@support((1, 6, 7))
def convert_FixedBatchNormalization(
        func, opset_version, input_names, output_names, context):
    return convert_BatchNormalization(
        func, opset_version, input_names, output_names, context)


def convert_LocalResponseNormalization(
        func, opset_version, input_names, output_names, context):
    size = int(func.n)
    return onnx_helper.make_node(
        'LRN', input_names, output_names,
        alpha=float(func.alpha) * size,
        beta=float(func.beta),
        bias=float(func.k),
        size=size,
    ),


def convert_NormalizeL2(
        func, opset_version, input_names, output_names, context):
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
        'LpNormalization', input_names, output_names,
        axis=int(func.axis[0]),
        p=2,
    ),
