from onnx_chainer.functions.opset_version import support
from onnx_chainer import onnx_helper


@support((1, 6))
def convert_ClippedReLU(func, opset_version, input_names,
                        num_outputs, context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Clip', input_names, num_outputs,
            min=0.0, max=func.cap,
            consumed_inputs=[1]
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'Clip', input_names, num_outputs,
            min=0.0, max=func.cap,
        ),


@support((1, 6))
def convert_ELU(func, opset_version, input_names, num_outputs,
                context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Elu', input_names, num_outputs,
            alpha=func.alpha,
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'Elu', input_names, num_outputs,
            alpha=func.alpha
        ),


@support((1, 6))
def convert_HardSigmoid(func, opset_version, input_names,
                        num_outputs, context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'HardSigmoid', input_names, num_outputs,
            alpha=0.2,
            beta=0.5,
            consumed_inputs=[1],
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'HardSigmoid', input_names, num_outputs,
            alpha=0.2,
            beta=0.5
        ),


@support((1, 6))
def convert_LeakyReLU(func, opset_version, input_names,
                      num_outputs, context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'LeakyRelu', input_names, num_outputs,
            alpha=func.slope,
            consumed_inputs=[1],
        ),
    elif opset_version == 6:
        return onnx_helper.make_node(
            'LeakyRelu', input_names, num_outputs,
            alpha=func.slope
        ),


def convert_LogSoftmax(func, opset_version, input_names,
                       num_outputs, context, parameters):
    return onnx_helper.make_node(
        'LogSoftmax', input_names, num_outputs,
        axis=1
    ),


@support((1, 6, 7))
def convert_PReLUFunction(func, opset_version, input_names,
                          num_outputs, context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'PRelu', input_names, num_outputs, consumed_inputs=[1]),
    elif opset_version == 6:
        return onnx_helper.make_node('PRelu', input_names, num_outputs),
    elif opset_version == 7:
        return onnx_helper.make_node('PRelu', input_names, num_outputs),


@support((1, 6))
def convert_ReLU(func, opset_version, input_names, num_outputs,
                 context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Relu', input_names, num_outputs,
            consumed_inputs=[1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Relu', input_names, num_outputs),


@support((1, 6))
def convert_Sigmoid(func, opset_version, input_names,
                    num_outputs, context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Sigmoid', input_names, num_outputs,
            consumed_inputs=[1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Sigmoid', input_names, num_outputs),


def convert_Softmax(func, opset_version, input_names,
                    num_outputs, context, parameters):
    return onnx_helper.make_node(
        'Softmax', input_names, num_outputs,
        axis=func.axis
    ),


def convert_Softplus(func, opset_version, input_names,
                     num_outputs, context, parameters):
    return onnx_helper.make_node('Softplus', input_names, num_outputs),


@support((1, 6))
def convert_Tanh(func, opset_version, input_names, num_outputs,
                 context, parameters):
    if opset_version == 1:
        return onnx_helper.make_node(
            'Tanh', input_names, num_outputs,
            consumed_inputs=[1]),
    elif opset_version == 6:
        return onnx_helper.make_node('Tanh', input_names, num_outputs),
