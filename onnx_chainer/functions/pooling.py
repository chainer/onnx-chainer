import warnings

from onnx_chainer import onnx_helper


def convert_AveragePooling2D(func, opset_version, input_names,
                             num_outputs, parameters):
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


def convert_AveragePoolingND(func, opset_version, input_names,
                             num_outputs, parameters):
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


def convert_MaxPooling2D(func, opset_version, input_names,
                         num_outputs, parameters):
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


def convert_MaxPoolingND(func, opset_version, input_names,
                         num_outputs, parameters):
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
                         num_outputs, parameters):
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
