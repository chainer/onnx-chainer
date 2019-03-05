import os
import warnings

import numpy as np
import onnx

try:
    import onnxruntime as rt
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    warnings.warn(
        'ONNXRuntime is not installed. Please install it to use '
        ' the testing utility for ONNX-Chainer\'s converters.',
        ImportWarning)
    ONNXRUNTIME_AVAILABLE = False


def check_model_expect(test_path, input_names=None):
    if not ONNXRUNTIME_AVAILABLE:
        raise ImportError('check_output requires onnxruntime.')

    model_path = os.path.join(test_path, 'model.onnx')
    with open(model_path, 'rb') as f:
        onnx_model = onnx.load_model(f)

    test_data_sets = sorted([
        p for p in os.listdir(test_path) if p.startswith('test_data_set_')])
    for test_data in test_data_sets:
        test_data_path = os.path.join(test_path, test_data)
        assert os.path.isdir(test_data_path)

        file_list = sorted(os.listdir(test_data_path))
        inputs, outputs = [], []

        for file_name in file_list:
            if not file_name.endswith('.pb'):
                continue
            path = os.path.join(test_path, test_data, file_name)
            with open(path, 'rb') as f:
                array = onnx.numpy_helper.to_array(onnx.load_tensor(path))
            if file_name.startswith('input_'):
                inputs.append(array)
            else:
                outputs.append(array)

        sess = rt.InferenceSession(onnx_model.SerializeToString())

        # To detect unexpected inputs created by exporter, check input names
        rt_input_names = [i.name for i in sess.get_inputs()]
        if input_names is not None:
            assert list(sorted(input_names)) == list(sorted(rt_input_names))

        rt_out = sess.run(
            None, {name: array for name, array in zip(rt_input_names, inputs)})
        for cy, my in zip(outputs, rt_out):
            np.testing.assert_allclose(cy, my, rtol=1e-5, atol=1e-5)
