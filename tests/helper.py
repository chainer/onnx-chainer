import os
import unittest
import warnings

import onnx
import pytest

import onnx_chainer
from onnx_chainer.testing.get_test_data_set import gen_test_data_set


class ONNXModelTest(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def set_config(self, disable_experimental_warning):
        pass

    @pytest.fixture(autouse=True, scope='function')
    def set_name(self, request):
        cls_name = request.cls.__name__
        self.default_name = cls_name[len('Test'):].lower()
        self.check_out_values = None
        selected_runtime = request.config.getoption('value-check-runtime')
        if selected_runtime == 'onnxruntime':
            from onnx_chainer.testing.test_onnxruntime import check_model_expect  # NOQA
            self.check_out_values = check_model_expect
        elif selected_runtime == 'mxnet':
            from onnx_chainer.testing.test_mxnet import check_model_expect
            self.check_out_values = check_model_expect
        else:
            self.check_out_values = None

    def expect(self, model, args, name=None, skip_opset_version=None,
               with_warning=False, train=False, input_names=None,
               output_names=None):
        """Compare model output and test runtime output.

        Make an ONNX model from target model with args, and put output
        directory. Then test runtime load the model, and compare.

        Arguments:
            model (~chainer.Chain): The target model.
            args (list or dict): Arguments of the target model.
            name (str): name of test. Set class name on default.
            skip_opset_version (list): Versions to skip test.
            with_warning (bool): If True, check warnings.
            train (bool): If True, output computational graph with train mode.
            input_names (str, list or dict): Customize input names. If set,
                check input names generated by exported ONNX model.
            output_names (str, list or dict): Customize output names. If set,
                check output names generated by exported ONNX model.
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
                        model, args, dir_name, opset_version, train,
                        input_names, output_names)
                assert len(w) == 1
            else:
                test_path = gen_test_data_set(
                    model, args, dir_name, opset_version, train, input_names,
                    output_names)

            onnx_model_path = os.path.join(test_path, 'model.onnx')
            assert os.path.isfile(onnx_model_path)
            with open(onnx_model_path, 'rb') as f:
                onnx_model = onnx.load_model(f)
            check_all_connected_from_inputs(onnx_model)

            graph_input_names = _get_graph_input_names(onnx_model)
            if input_names:
                if isinstance(input_names, dict):
                    expected_names = list(sorted(input_names.values()))
                else:
                    expected_names = list(sorted(input_names))
                assert list(sorted(graph_input_names)) == expected_names
            if output_names:
                if isinstance(output_names, dict):
                    expected_names = list(sorted(output_names.values()))
                else:
                    expected_names = list(sorted(output_names))
                graph_output_names = [v.name for v in onnx_model.graph.output]
                assert list(sorted(graph_output_names)) == expected_names

            # Export function can be add unexpected inputs. Collect inputs
            # from ONNX model, and compare with another input list got from
            # test runtime.
            if self.check_out_values is not None:
                self.check_out_values(test_path, input_names=graph_input_names)


def check_all_connected_from_inputs(onnx_model):
    edge_names = set(_get_graph_input_names(onnx_model))
    # Nodes which are not connected from the network inputs.
    orphan_nodes = []
    for node in onnx_model.graph.node:
        if not edge_names.intersection(node.input):
            orphan_nodes.append(node)
        for output_name in node.output:
            edge_names.add(output_name)
    assert not(orphan_nodes), '{}'.format(orphan_nodes)


def _get_graph_input_names(onnx_model):
    initialized_graph_input_names = {
        i.name for i in onnx_model.graph.initializer}
    return [i.name for i in onnx_model.graph.input if i.name not in
            initialized_graph_input_names]
