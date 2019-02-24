import onnx


def make_node(op_name, input_names, output_spec=1, **kwargs):
    if isinstance(output_spec, int):
        output_names = ['output_%d' % i for i in range(output_spec)]
    else:
        output_names = output_spec
    return onnx.helper.make_node(op_name, input_names, output_names, **kwargs)
