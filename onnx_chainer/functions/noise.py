import chainer
from onnx import helper

from onnx_chainer import mapping


def convert_Dropout(func, onnx_op_name, input_names, output_names, parameters):
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        is_test=0 if chainer.config.train else 1,
        ratio=func.dropout_ratio,
    ),
