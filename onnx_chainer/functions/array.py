import chainer
import numpy as np
from onnx import helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from onnx_chainer import mapping


def convert_Cast(func, onnx_op_name, input_names, output_names, parameters):
    typ = func.type if isinstance(func.type, np.dtype) else np.dtype(func.type)
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        to=mapping.TENSOR_TYPE_TO_NAME[NP_TYPE_TO_TENSOR_TYPE[typ]]
    ),


def convert_Concat(func, onnx_op_name, input_names, output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axis=func.axis
    ),


def convert_Copy(func, onnx_op_name, input_names, output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names
    ),


def convert_Depth2Space(
        func, onnx_op_name, input_names, output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        blocksize=func.r
    ),


def convert_GetItem(func, onnx_op_name, input_names, output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        slice=func.slices
    ),


def convert_Pad(func, onnx_op_name, input_names, output_names, parameters):

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


def convert_Reshape(func, onnx_op_name, input_names, output_names, parameters):

    shape = np.asarray(list(func.shape), dtype=np.int64)
    shape_param = chainer.Parameter(shape)
    parameters.append(shape_param)
    input_names.append(str(id(shape_param)))

    return helper.make_node(
        onnx_op_name, input_names, output_names,
    ),


def convert_Space2Depth(func, onnx_op_name, input_names, output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        blocksize=func.r
    ),


def convert_SplitAxis(func, onnx_op_name, input_names, output_names, parameters):

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


def convert_Squeeze(func, onnx_op_name, input_names, output_names, parameters):

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


def convert_Tile(func, onnx_op_name, input_names, output_names, parameters):

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


def convert_Transpose(func, onnx_op_name, input_names, output_names, parameters):

    if func.axes is None:
        node = helper.make_node(onnx_op_name, input_names, output_names)
    else:
        node = helper.make_node(
            onnx_op_name, input_names, output_names,
            perm=func.axes
        )

    return node,
