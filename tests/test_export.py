import os
import warnings

import chainer
import numpy as np
import onnx
import pytest

from onnx_chainer import export_testcase
from onnx_chainer import onnx_helper
from onnx_chainer.testing import input_generator
from onnx_chainer.testing.test_onnxruntime import check_model_expect


def test_export_external_converters_overwrite(tmpdir):
    path = tmpdir.mkdir('test_export_external_converters_overwrite').dirname

    model = chainer.Sequential(chainer.functions.sigmoid)
    x = input_generator.positive_increasing(2, 5)

    def custom_converter(params):
        return onnx_helper.make_node(
            'Tanh', params.input_names, len(params.output_names)),

    addon_converters = {'Sigmoid': custom_converter}
    export_testcase(model, x, path, external_converters=addon_converters)

    tanh_outputs = chainer.functions.tanh(x).array
    output_path = os.path.join(path, 'test_data_set_0', 'output_0.pb')
    onnx_helper.write_tensor_pb(output_path, '', tanh_outputs)  # overwrite

    check_model_expect(path)


@pytest.mark.parametrize('domain,version', [(None, 0), ('domain', 0)])
def test_export_external_converters_custom_op(tmpdir, domain, version):
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

    def custom_converter(params):
        return onnx_helper.make_node(
            'Dummy', params.input_names, len(params.output_names),
            domain=domain),

    addon_converters = {'Dummy': custom_converter}

    external_opset_imports = {}
    if domain is not None:
        external_opset_imports[domain] = version
    with warnings.catch_warnings(record=True) as w:
        export_testcase(
            model, x, path, external_converters=addon_converters,
            external_opset_imports=external_opset_imports)
        assert len(w) == 1
        assert '"Dummy"' in str(w[-1].message)

    output_path = os.path.join(path, 'test_data_set_0', 'output_0.pb')
    assert os.path.isfile(output_path)
    output = onnx.numpy_helper.to_array(onnx.load_tensor(output_path))
    expected_output = np.ones_like(x)
    np.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-5)
