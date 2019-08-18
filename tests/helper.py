import glob
import os
import unittest
import warnings

import chainer
import numpy as np
import onnx
import pytest

import onnx_chainer
from onnx_chainer.testing.get_test_data_set import gen_test_data_set


def load_input_data(data_dir):
    input_data = []
    for pb in sorted(glob.glob(os.path.join(
            data_dir, 'test_data_set_0', 'input_*.pb'))):
        tensor = onnx.load_tensor(pb)
        ndarray = onnx.numpy_helper.to_array(tensor)
        input_data.append(ndarray)
    return input_data


class ONNXModelTest(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def set_config(self, disable_experimental_warning):
        pass

    @pytest.fixture(autouse=True, scope='function')
    def set_name(self, request, check_model_expect):
        cls_name = request.cls.__name__
        self.default_name = cls_name[len('Test'):].lower()
        self.check_out_values = check_model_expect

    def expect(self, model, args, name=None, skip_opset_version=None,
               skip_outvalue_version=None, with_warning=False,
               custom_model_test_func=None, expected_num_initializers=None,
               **kwargs):
        """Compare model output and test runtime output.

        Make an ONNX model from target model with args, and put output
        directory. Then test runtime load the model, and compare.

        Arguments:
            model (~chainer.Chain): The target model.
            args (list or dict): Arguments of the target model.
            name (str): name of test. Set class name on default.
            skip_opset_version (list): Versions to skip test.
            skip_outvalue_version (list): Versions to skip output value check.
            with_warning (bool): If True, check warnings.
            custom_model_test_func (func): A function to check generated
                model. The functions is called before checking output values.
                ONNX model is passed to arguments.
            expected_num_initializers (int): The expected number of
                initializers in the output ONNX model.
            **kwargs (dict): keyward arguments for ``onnx_chainer.export``.
        """

        test_name = name
        if test_name is None:
            test_name = self.default_name

        for opset_version in range(onnx_chainer.MINIMUM_OPSET_VERSION,
                                   onnx.defs.onnx_opset_version() + 1):
            if skip_opset_version is not None and\
                    opset_version in skip_opset_version:
                continue

            dir_name = 'test_' + test_name
            if with_warning:
                with warnings.catch_warnings(record=True) as w:
                    test_path = gen_test_data_set(
                        model, args, dir_name, opset_version, **kwargs)
                assert len(w) == 1
            else:
                test_path = gen_test_data_set(
                    model, args, dir_name, opset_version, **kwargs)

            onnx_model_path = os.path.join(test_path, 'model.onnx')
            assert os.path.isfile(onnx_model_path)
            with open(onnx_model_path, 'rb') as f:
                onnx_model = onnx.load_model(f)
            check_all_connected_from_inputs(onnx_model)

            if expected_num_initializers is not None:
                actual_num_initializers = len(onnx_model.graph.initializer)
                assert expected_num_initializers == actual_num_initializers

            graph_input_names = _get_graph_input_names(onnx_model)
            if kwargs.get('input_names', {}):
                input_names = kwargs['input_names']
                if isinstance(input_names, dict):
                    expected_names = list(sorted(input_names.values()))
                else:
                    expected_names = list(sorted(input_names))
                assert list(sorted(graph_input_names)) == expected_names
            if kwargs.get('output_names', {}):
                output_names = kwargs['output_names']
                if isinstance(output_names, dict):
                    expected_names = list(sorted(output_names.values()))
                else:
                    expected_names = list(sorted(output_names))
                graph_output_names = [v.name for v in onnx_model.graph.output]
                assert list(sorted(graph_output_names)) == expected_names

            # Input data is generaged by `network_inputs` dict, this can
            # introduce unexpected conversions. Check values of input PB with
            # test args.
            if isinstance(args, (tuple, list)):
                flat_args = args
            elif isinstance(args, dict):
                flat_args = args.values()
            else:
                flat_args = [args]
            input_data = load_input_data(test_path)
            assert len(input_data) == len(flat_args)
            for i, arg in enumerate(flat_args):
                array = arg.array if isinstance(arg, chainer.Variable) else arg
                array = chainer.cuda.to_cpu(array)
                np.testing.assert_allclose(
                    array, input_data[i], rtol=1e-5, atol=1e-5)

            if custom_model_test_func is not None:
                custom_model_test_func(onnx_model)

            if skip_outvalue_version is not None and\
                    opset_version in skip_outvalue_version:
                continue

            # Export function can be add unexpected inputs. Collect inputs
            # from ONNX model, and compare with another input list got from
            # test runtime.
            if self.check_out_values is not None:
                self.check_out_values(test_path, input_names=graph_input_names)

    def to_gpu(self, model, x):
        model = model.copy()
        model.to_gpu()
        x = chainer.cuda.to_gpu(x)
        return model, x


def check_all_connected_from_inputs(onnx_model):
    edge_names = get_initializer_names(onnx_model) |\
        _get_input_names(onnx_model)
    # Nodes which are not connected from the network inputs.
    orphan_nodes = []
    for node in onnx_model.graph.node:
        if not node.input:
            for output_name in node.output:
                edge_names.add(output_name)
            continue
        if not edge_names.intersection(node.input):
            orphan_nodes.append(node)
        for output_name in node.output:
            edge_names.add(output_name)
    assert not(orphan_nodes), '{}'.format(orphan_nodes)


def get_initializer_names(onnx_model):
    return {i.name for i in onnx_model.graph.initializer}


def _get_input_names(onnx_model):
    return {i.name for i in onnx_model.graph.input}


def _get_graph_input_names(onnx_model):
    return list(
        _get_input_names(onnx_model) - get_initializer_names(onnx_model))
