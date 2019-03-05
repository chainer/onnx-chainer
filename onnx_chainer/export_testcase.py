import os

import chainer

from onnx_chainer.export import export
from onnx_chainer.onnx_helper import write_tensor_pb


def export_testcase(
        model, args, out_dir, graph_name='Graph', output_grad=False,
        opset_version=None, train=False):
    """Export model and I/O tensors of the model in protobuf format.

    Similar to the `export` function, this function first performs a forward
    computation to a given input for obtaining an output. Then, this function
    saves the pair of input and output in Protobuf format, which is a
    defacto-standard format in ONNX.

    This function also saves the model with the name "model.onnx".

    Args:
        model (~chainer.Chain): The model object.
        args (list): The arguments which are given to the model
            directly. Unlike `export` function, only `list` type is accepted.
        out_dir (str): The directory name used for saving the input and output.
        graph_name (str): A string to be used for the ``name`` field of the
            graph in the exported ONNX model.
        output_grad (bool): If True, this function will output model's
            gradient with names 'gradient_%d.pb'.
        train (bool): If True, output computational graph with train mode.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.cleargrads()
    onnx_model, inputs, outputs = export(
        model, args, filename=os.path.join(out_dir, 'model.onnx'),
        graph_name=graph_name, opset_version=opset_version,
        train=train, return_flat_inout=True)

    test_data_dir = os.path.join(out_dir, 'test_data_set_0')
    os.makedirs(test_data_dir, exist_ok=True)
    # TODO(disktnk): consider to resolve input names smarter
    input_names = _get_graph_input_names(onnx_model)
    for i, var in enumerate(inputs):
        pb_name = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
        array = chainer.cuda.to_cpu(var.array)
        write_tensor_pb(pb_name, input_names[i], array)

    for i, var in enumerate(outputs):
        pb_name = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
        array = chainer.cuda.to_cpu(var.array)
        # TODO(disktnk): set customized output name
        write_tensor_pb(pb_name, '', array)

    if output_grad:
        # Perform backward computation
        if len(outputs) > 1:
            outputs = chainer.functions.identity(*outputs)
        for out in outputs:
            out.grad = model.xp.ones_like(out.array)
        outputs[0].backward()

        for i, (name, param) in enumerate(model.namedparams()):
            pb_name = os.path.join(test_data_dir, 'gradient_{}.pb'.format(i))
            grad = chainer.cuda.to_cpu(param.grad)
            write_tensor_pb(pb_name, '', grad)


def _get_graph_input_names(onnx_model):
    initialized_graph_input_names = {
        i.name for i in onnx_model.graph.initializer}
    return [i.name for i in onnx_model.graph.input if i.name not in
            initialized_graph_input_names]
