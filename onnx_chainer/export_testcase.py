import chainer
import os

from onnx import numpy_helper


def makedirs(d):
    if not os.path.exists(d):
        os.makedirs(d)


def export_testcase(model, args, out_dir):
    """Export input and output of a model in protobuf format.

    Similar to the `export` function, this function first performs a forward
    computation to a given input for obtaining an output. Then, this function
    saves the pair of input and output in Protobuf format, which is a
    defacto-standard format in ONNX.

    Args:
        model (~chainer.Chain): The model object.
        args (list): The arguments which are given to the model
            directly. Unlike `export` function, only `list` type is accepted.
        out_dir (str): The directory name used for saving the input and output.
    """
    makedirs(out_dir)
    onnx_extra_inputs = []
    if hasattr(model, 'extra_inputs'):
        onnx_extra_inputs = model.extra_inputs

    test_data_dir = '%s/test_data_set_0' % out_dir
    makedirs(test_data_dir)
    for i, var in enumerate(list(args) + list(onnx_extra_inputs)):
        with open(os.path.join(test_data_dir, 'input_%d.pb' % i), 'wb') as f:
            t = numpy_helper.from_array(var.data, 'Input_%d' % i)
            f.write(t.SerializeToString())

    chainer.config.train = True
    model.cleargrads()
    result = model(*args)

    with open(os.path.join(test_data_dir, 'output_0.pb'), 'wb') as f:
        t = numpy_helper.from_array(result.array, '')
        f.write(t.SerializeToString())
