import chainer
from onnx import helper

from onnx_chainer import mapping


def convert_Dropout(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]

    return helper.make_node(
        onnx_op_name, input_names, output_names,
        is_test=chainer.config.train,
        ratio=func.dropout_ratio,
    ),
