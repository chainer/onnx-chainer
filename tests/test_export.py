import os

import chainer
import numpy as np

from onnx_chainer import export_testcase
from onnx_chainer import onnx_helper
from onnx_chainer.testing import input_generator
from onnx_chainer.testing.test_onnxruntime import check_model_expect


def test_export_external_converters_overwrite(tmpdir):
    path = tmpdir.mkdir('test_export_external_converters_overwrite').dirname

    model = chainer.Sequential(chainer.functions.sigmoid)
    x = input_generator.positive_increasing(2, 5)

    def custom_converter(func, opset_version, input_names, num_outputs,
                         context, parameters):
        return onnx_helper.make_node('Tanh', input_names, num_outputs),

    addon_converters = {'Sigmoid': custom_converter}
    export_testcase(model, x, path, external_converters=addon_converters)

    tanh_outputs = chainer.functions.tanh(x).array
    output_path = os.path.join(path, 'test_data_set_0', 'output_0.pb')
    onnx_helper.write_tensor_pb(output_path, '', tanh_outputs)  # overwrite

    check_model_expect(path)


def test_export_external_converters_custom_op(tmpdir):
    path = tmpdir.mkdir('test_export_external_converters_custom_op').dirname

    class Dummy(chainer.FunctionNode):

        def forward_cpu(self, inputs):
            self.x = inputs[0]
            return np.ones_like(inputs[0]),

        def backward(self, indexes, grad_outputs):
            return np.zeros_like(self.x),

    def dummy_function(x):
        return Dummy().apply((x,))[0]

    model = chainer.Sequential(dummy_function)
    x = input_generator.increasing(2, 5)

    def custom_converter(func, opset_version, input_names, num_outputs,
                         context, parameters):
        return onnx_helper.make_node('Dummy', input_names, num_outputs),

    addon_converters = {'Dummy': custom_converter}
    export_testcase(model, x, path, external_converters=addon_converters)
