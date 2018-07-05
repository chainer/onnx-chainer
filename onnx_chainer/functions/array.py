import chainer
import numpy as np
from onnx import helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from onnx_chainer import mapping


def convert_Cast(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    typ = func.type if isinstance(func.type, np.dtype) else np.dtype(func.type)
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        to=mapping.TENSOR_TYPE_TO_NAME[NP_TYPE_TO_TENSOR_TYPE[typ]]
    ),


def convert_Concat(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axis=func.axis
    ),


def convert_Depth2Space(
        func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        blocksize=func.r
    ),


def convert_GetItem(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        slice=func.slices
    ),


def convert_Pad(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    if func.mode not in ['constant', 'reflect', 'edge']:
        raise ValueError(
            '{} mode is not supported in ONNX\'s Pad operation'.format(
                func.mode))

    if 'constant_values' in func.keywords:
        values = func.keywords['constant_values']
        if not isinstance(values, int) and len(values) > 1:
            raise ValueError(
                'ONNX doesn\'t support multiple constant values for Pad '
                'operation')
        elif not isinstance(values, int):
            values = values[0]

        node = helper.make_node(
            onnx_op_name, input_names, output_names,
            mode=func.mode,
            pads=func.pad_bw.tolist(),
            value=values
        )
    else:
        node = helper.make_node(
            onnx_op_name, input_names, output_names,
            mode=func.mode,
            pads=func.pad_bw.ravel().tolist(),
        )

    return node,


def convert_Reshape(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    # TODO(mitmul): This part is needed for opset_version > 1
    # # Add tiles and axis to graph
    # shape = np.asarray(func.shape, dtype=np.int64)
    # shape_param = chainer.Parameter(shape)
    # parameters.append(shape_param)
    # input_names.append(str(id(shape_param)))

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        shape=func.shape
    ),


def convert_Space2Depth(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        blocksize=func.r
    ),


def convert_SplitAxis(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    if func.indices is not None:
        indices_or_sections = func.indices
    else:
        indices_or_sections = func.sections

    if hasattr(indices_or_sections, '__iter__'):
        split = []
        prev_i = 0
        for i in indices_or_sections:
            split.append(i - prev_i)
            prev_i = i
    else:
        length = func.inputs[0].shape[func.axis] // indices_or_sections
        split = [length for _ in range(indices_or_sections)]

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axis=func.axis,
        split=split
    ),


def convert_Squeeze(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    if func.axis is None:
        axis = []
        for s in func.inputs[0].shape:
            if s == 1:
                axis.append(s)
    else:
        axis = func.axis

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axes=axis
    ),


def convert_Tile(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    # Add tiles and axis to graph
    if isinstance(func.reps, int):
        func.reps = [func.reps]
    tiles = np.asarray(func.reps, dtype=np.float32)

    tiles_param = chainer.Parameter(tiles)
    parameters.append(tiles_param)
    input_names.append(str(id(tiles_param)))

    # In operater version = 1, axis also should be given
    axis = np.array([i for i, _ in enumerate(func.reps)], dtype=np.float32)
    axis_param = chainer.Parameter(axis)
    parameters.append(axis_param)
    input_names.append(str(id(axis_param)))

    node = helper.make_node(onnx_op_name, input_names, output_names)
    return node,


def convert_Transpose(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    if func.axes is None:
        node = helper.make_node(onnx_op_name, input_names, output_names)
    else:
        node = helper.make_node(
            onnx_op_name, input_names, output_names,
            perm=func.axes
        )

    return node,
