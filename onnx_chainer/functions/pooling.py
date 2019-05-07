import warnings

import chainer
from chainer.utils import conv
import numpy as np

from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


@support((1, 7))
def convert_AveragePooling2D(func, opset_version, input_names,
                             num_outputs, context, parameters):
    pad = [func.ph, func.pw]
    stride = [func.sy, func.sx]
    ksize = [func.kh, func.kw]
    if func.cover_all:
        # Supports cover_all by setting extra padding
        for p, s, k in zip(pad, stride, ksize):
            if k <= p + s - 1:
                raise RuntimeError(
                    'Could not correctly export in the current setting'
                    ' (ksize={} pad={} stride={}). Please set pad or stride to'
                    'lower value.'.format(k, p, s))
        pad.extend([p + s - 1 for p, s in zip(pad, stride)])
    else:
        pad = pad * 2

    if opset_version == 1:
        raise ValueError(
            'AveragePooling2D is not compatible with ONNX\'s AveragePool-1. '
            'Use operation set version >= 7.')
    elif opset_version == 7:
        return onnx_helper.make_node(
            'AveragePool', input_names, num_outputs,
            kernel_shape=ksize,
            pads=pad,
            strides=stride,
            count_include_pad=1,
        ),


@support((1, 7))
def convert_AveragePoolingND(func, opset_version, input_names,
                             num_outputs, context, parameters):
    pad = list(func.pad[:])
    if func.cover_all:
        # Supports cover_all by setting extra padding
        for p, s, k in zip(pad, func.stride, func.ksize):
            # Raise exception because a virtual pad for cover_all must be
            # smaller than ksize in the current ONNX
            if k <= p + s - 1:
                raise RuntimeError(
                    'Could not correctly export in the current setting'
                    ' (ksize={} pad={} stride={}). Please set pad or stride to'
                    'lower value.'.format(k, p, s))
        pad.extend([p + s - 1 for p, s in zip(pad, func.stride)])
    else:
        pad = pad * 2

    if opset_version == 1:
        raise ValueError(
            'AveragePoolingND is not compatible with ONNX\'s AveragePool-1. '
            'Use operation set version >= 7.')
    elif opset_version == 7:
        return onnx_helper.make_node(
            'AveragePool', input_names, num_outputs,
            kernel_shape=func.ksize,
            pads=pad,
            strides=func.stride,
            count_include_pad=1,
        ),


@support((1, 8))
def convert_MaxPooling2D(func, opset_version, input_names,
                         num_outputs, context, parameters):
    pad = [func.ph, func.pw]
    stride = [func.sy, func.sx]
    ksize = [func.kh, func.kw]
    if func.cover_all:
        # Supports cover_all by setting extra padding
        for p, s, k in zip(pad, stride, ksize):
            if k <= p + s - 1:
                raise RuntimeError(
                    'Could not correctly export in the current setting'
                    ' (ksize={} pad={} stride={}). Please set pad or stride to'
                    'lower value.'.format(k, p, s))
        pad.extend([p + s - 1 for p, s in zip(pad, stride)])
    else:
        pad = pad * 2

    if opset_version == 1:
        return onnx_helper.make_node(
            'MaxPool', input_names, num_outputs,
            kernel_shape=ksize,
            pads=pad,
            strides=stride
        ),
    elif opset_version == 8:
        return onnx_helper.make_node(
            'MaxPool', input_names, num_outputs,
            kernel_shape=ksize,
            pads=pad,
            strides=stride,
            storage_order=0,  # row major
        ),


@support((1, 8))
def convert_MaxPoolingND(func, opset_version, input_names,
                         num_outputs, context, parameters):
    pad = list(func.pad[:])
    if func.cover_all:
        # Supports cover_all by setting extra padding
        for p, s, k in zip(pad, func.stride, func.ksize):
            # Raise exception because a virtual pad for cover_all must be
            # smaller than ksize in the current ONNX
            if k <= p + s - 1:
                raise RuntimeError(
                    'Could not correctly export in the current setting'
                    ' (ksize={} pad={} stride={}). Please set pad or stride to'
                    'lower value.'.format(k, p, s))
        pad.extend([p + s - 1 for p, s in zip(pad, func.stride)])
    else:
        pad = pad * 2

    if opset_version == 1:
        return onnx_helper.make_node(
            'MaxPool', input_names, num_outputs,
            kernel_shape=func.ksize,
            pads=pad,
            strides=func.stride
        ),
    elif opset_version == 8:
        return onnx_helper.make_node(
            'MaxPool', input_names, num_outputs,
            kernel_shape=func.ksize,
            pads=pad,
            strides=func.stride,
            storage_order=0,  # row major
        ),


def convert_ROIPooling2D(func, opset_version, input_names,
                         num_outputs, context, parameters):
    warnings.warn(
        'It\'s possible that output does not match with Chainer, please check '
        'each runtime\'s implementation. For example, when input x has '
        'negative values, some runtimes set max(output, 0) unlike Chainer.',
        UserWarning)
    return onnx_helper.make_node(
        'MaxRoiPool', input_names, num_outputs,
        pooled_shape=[func.outh, func.outw],
        spatial_scale=func.spatial_scale,
    ),


@support((7, 9))
def convert_Unpooling2D(func, opset_version, input_names, num_outputs,
                        context, parameters):
    pad = [func.ph, func.pw]
    stride = [func.sy, func.sx]
    ksize = [func.kh, func.kw]
    outsize = [func.outh, func.outw]
    # TODO(hamaji): These could be implemented by `Slice` and `Pad`.
    if func.cover_all:
        raise RuntimeError('ONNX-chainer does not support `cover_all=True` '
                           'for Unpooling2D')
    h, w = func.inputs[0].shape[2:]
    expected_outsize = [
        conv.get_deconv_outsize(
            h, func.kh, func.sy, func.ph, cover_all=func.cover_all),
        conv.get_deconv_outsize(
            w, func.kh, func.sy, func.ph, cover_all=func.cover_all)
    ]
    if outsize != expected_outsize:
        raise RuntimeError('ONNX-chainer does not support `outsize!=None` '
                           'for Unpooling2D: expected={} actual={}'.format(
                               expected_outsize, outsize))
    if pad != [0, 0]:
        raise RuntimeError('ONNX-chainer does not support `pad!=0` '
                           'for Unpooling2D')
    # This one would require an extra 1x1 MaxPool.
    if stride != ksize:
        raise RuntimeError('ONNX-chainer does not support `stride!=ksize` '
                           'for Unpooling2D: stride={} ksize={}'.format(
                               stride, ksize))
    scales = [1.0, 1.0, float(func.kh), float(func.kw)]
    if opset_version == 7:
        return onnx_helper.make_node('Upsample', input_names, num_outputs,
                                     scales=scales),
    if opset_version == 9:
        scales = np.array(scales, dtype=np.float32)
        scales_param = chainer.Parameter(scales)
        parameters.append(scales_param)
        input_names.append(context.get_name(scales_param))
        return onnx_helper.make_node('Upsample', input_names, num_outputs),
