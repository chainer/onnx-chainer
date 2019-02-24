import chainer
import numpy as np
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from onnx_chainer import mapping
from onnx_chainer import onnx_helper


def convert_Cast(func, opset_version, input_names, output_names,
                 parameters):
    typ = func.type if isinstance(func.type, np.dtype) else np.dtype(func.type)
    if opset_version == 1:
        return onnx_helper.make_node(
            'Cast', input_names, output_names,
            to=mapping.TENSOR_TYPE_TO_NAME[NP_TYPE_TO_TENSOR_TYPE[typ]]
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'Cast', input_names, output_names,
            to=NP_TYPE_TO_TENSOR_TYPE[typ]
        ),


def convert_Concat(func, opset_version, input_names,
                   output_names, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Concat', input_names, output_names,
            axis=func.axis
        ),
    elif opset_version == 4:
        return onnx_helper.make_node(
            'Concat', input_names, output_names,
            axis=func.axis
        ),


def convert_Copy(func, opset_version, input_names, output_names,
                 parameters):
    return onnx_helper.make_node(
        'Identity', input_names, output_names
    ),


def convert_Depth2Space(func, opset_version, input_names,
                        output_names, parameters):
    return onnx_helper.make_node(
        'DepthToSpace', input_names, output_names,
        blocksize=func.r
    ),


def convert_GetItem(func, opset_version, input_names,
                    output_names, parameters):
    x = func.inputs[0]
    axes, starts, ends = [], [], []
    squeeze_idxs, unsqueeze_idxs = [], []
    skipped = 0  # when set ellipsis, need to skip index rolling

    for i, idx in enumerate(func.slices):
        # axis means the index of input x, adjust None and Ellipsis counts
        axis = i - len(unsqueeze_idxs) + skipped
        if isinstance(idx, slice):
            if idx.step is not None and idx.step != 1:
                raise ValueError(
                    'GetItem with {}step slicing is not supported in ONNX '
                    'Slice operator'.format(idx.step))
            axes.append(axis)
            starts.append(0 if idx.start is None else idx.start)
            ends.append(x.shape[axis] if idx.stop is None else idx.stop)
        elif isinstance(idx, int):
            axes.append(axis)
            starts.append(idx)
            ends.append(idx+1)
            squeeze_idxs.append(axis)
        elif isinstance(idx, np.ndarray) and idx.ndim == 0:
            scalar_idx = np.asscalar(idx)
            axes.append(axis)
            starts.append(scalar_idx)
            ends.append(scalar_idx+1)
            squeeze_idxs.append(axis)
        elif idx is None:
            unsqueeze_idxs.append(i - len(squeeze_idxs) + skipped)
        elif idx is Ellipsis:
            # calculate rest slice number except None, GetItem does not allow
            # multiple Ellipsis, so ignore latter Ellipsis count
            rest_slice_len = len(
                [idx_ for idx_ in func.slices[i+1:] if idx_ is not None])
            assert skipped == 0
            skipped = len(x.shape) - axis - rest_slice_len - 1
        else:
            # not support advanced index like `array[[0,1], [0, 1]]`
            raise ValueError(
                'GetItem with type {} cannot handle in ONNX Slice, so that '
                'ONNX-Chainer does not accept the type'.format(type(idx)))
    nodes = []
    nodes.append(onnx_helper.make_node(
        'Slice', input_names, 1,
        axes=axes, starts=starts, ends=ends))

    if squeeze_idxs:
        nodes.append(onnx_helper.make_node(
            'Squeeze', nodes[-1].output, 1,
            axes=squeeze_idxs))

    if unsqueeze_idxs:
        nodes.append(onnx_helper.make_node(
            'Unsqueeze', nodes[-1].output, 1,
            axes=unsqueeze_idxs))

    return tuple(nodes)


def convert_Pad(func, opset_version, input_names, output_names,
                parameters):
    if func.mode not in ['constant', 'reflect', 'edge']:
        raise ValueError(
            '{} mode is not supported in ONNX\'s Pad operation'.format(
                func.mode))

    pad_begin = []
    pad_end = []
    for pp in func.pad_bw.tolist():
        pad_begin.append(pp[0])
        pad_end.append(pp[1])
    pad = pad_begin + pad_end

    if 'constant_values' in func.keywords:
        values = func.keywords['constant_values']
        if not isinstance(values, int) and len(values) > 1:
            raise ValueError(
                'ONNX doesn\'t support multiple constant values for Pad '
                'operation')
        elif not isinstance(values, int):
            values = values[0]

        if opset_version == 1:
            node = onnx_helper.make_node(
                'Pad', input_names, output_names,
                mode=func.mode,
                paddings=pad,
                value=values
            )
        elif opset_version == 2:
            node = onnx_helper.make_node(
                'Pad', input_names, output_names,
                mode=func.mode,
                pads=pad,
                value=values
            )
    else:
        if opset_version == 1:
            node = onnx_helper.make_node(
                'Pad', input_names, output_names,
                mode=func.mode,
                paddings=pad,
                value=0.,
            )
        elif opset_version == 2:
            node = onnx_helper.make_node(
                'Pad', input_names, output_names,
                mode=func.mode,
                pads=pad,
                value=0.,
            )

    return node,


def convert_Reshape(func, opset_version, input_names,
                    output_names, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Reshape', input_names, output_names,
            shape=func.shape
        ),
    elif opset_version == 5:
        shape = np.asarray(list(func.shape), dtype=np.int64)
        shape_param = chainer.Parameter(shape)
        parameters.append(shape_param)
        input_names.append(str(id(shape_param)))

        return onnx_helper.make_node(
            'Reshape', input_names, output_names,
        ),


def convert_Space2Depth(func, opset_version, input_names,
                        output_names, parameters):
    return onnx_helper.make_node(
        'SpaceToDepth', input_names, output_names,
        blocksize=func.r
    ),


def convert_SplitAxis(func, opset_version, input_names,
                      output_names, parameters):
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

    if opset_version == 1:
        return onnx_helper.make_node(
            'Split', input_names, output_names,
            axis=func.axis,
            split=split
        ),
    elif opset_version == 2:
        return onnx_helper.make_node(
            'Split', input_names, output_names,
            axis=func.axis,
            split=split
        ),


def convert_Squeeze(func, opset_version, input_names,
                    output_names, parameters):
    if func.axis is None:
        axis = []
        for i, s in enumerate(func.inputs[0].shape):
            if s == 1:
                axis.append(i)
    else:
        axis = func.axis

    return onnx_helper.make_node(
        'Squeeze', input_names, output_names,
        axes=axis
    ),


def convert_Tile(func, opset_version, input_names, output_names,
                 parameters):
    # Add tiles and axis to graph
    if isinstance(func.reps, int):
        func.reps = [func.reps]
    tiles = np.asarray(func.reps, dtype=np.int64)

    tiles_param = chainer.Parameter(tiles)
    parameters.append(tiles_param)
    input_names.append(str(id(tiles_param)))

    # In operater version = 1, axis also should be given
    if opset_version == 1:
        axis = np.array([i for i, _ in enumerate(func.reps)], dtype=np.float32)
        axis_param = chainer.Parameter(axis)
        parameters.append(axis_param)
        input_names.append(str(id(axis_param)))

    return onnx_helper.make_node('Tile', input_names, output_names),


def convert_Transpose(func, opset_version, input_names,
                      output_names, parameters):

    if func.axes is None:
        node = onnx_helper.make_node('Transpose', input_names, output_names)
    else:
        node = onnx_helper.make_node(
            'Transpose', input_names, output_names,
            perm=func.axes
        )

    return node,


def convert_ExpandDims(func, opset_version, input_names,
                       output_names, parameters):
    axis = func.axis
    if axis < 0:
        axis = len(func.inputs[0].shape) + 1 + axis

    return onnx_helper.make_node(
        'Unsqueeze', input_names, output_names, axes=[axis]),
