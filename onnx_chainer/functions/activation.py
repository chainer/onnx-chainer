from onnx import helper

from onnx_chainer import mapping


def convert_ELU(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        alpha=func.alpha
    ),


def convert_HardSigmoid(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        alpha=0.2,
        beta=0.5
    ),


def convert_LeakyReLU(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        alpha=func.slope
    ),


def convert_LogSoftmax(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axis=1
    ),


def convert_PReLUFunction(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_ReLU(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Sigmoid(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Softmax(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(
        onnx_op_name, input_names, output_names,
        axis=func.axis
    ),


def convert_Softplus(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),


def convert_Tanh(func, input_names, output_names, parameters):
    onnx_op_name = mapping.operators[func.__class__.__name__]
    return helper.make_node(onnx_op_name, input_names, output_names),
