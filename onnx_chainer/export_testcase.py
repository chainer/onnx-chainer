import os

from onnx import numpy_helper
from onnx_chainer.export import export


def export_testcase(model, args, out_dir, graph_name='Graph'):
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
    """
    os.makedirs(out_dir, exist_ok=True)
    _, inputs, outputs = export(
        model, args, filename=os.path.join(out_dir, 'model.onnx'),
        graph_name=graph_name, return_flat_inout=True)

    test_data_dir = os.path.join(out_dir, 'test_data_set_0')
    os.makedirs(test_data_dir, exist_ok=True)
    for i, var in enumerate(inputs):
        with open(os.path.join(test_data_dir, 'input_%d.pb' % i), 'wb') as f:
            t = numpy_helper.from_array(var.data, 'Input_%d' % i)
            f.write(t.SerializeToString())

    for i, var in enumerate(outputs):
        with open(os.path.join(test_data_dir, 'output_%d.pb' % i), 'wb') as f:
            t = numpy_helper.from_array(var.data, '')
            f.write(t.SerializeToString())
