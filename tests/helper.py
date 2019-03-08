import os
import unittest
import warnings

import onnx
import pytest

import onnx_chainer
from onnx_chainer.testing.get_test_data_set import gen_test_data_set
from onnx_chainer.testing.test_onnxruntime import check_model_expect


class ONNXModelTest(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def set_config(self, desable_experimental_warning):
        pass

    @pytest.fixture(autouse=True, scope='function')
    def set_name(self, request):
        cls_name = request.cls.__name__
        self.default_name = cls_name[len('Test'):].lower()

    def expect(self, model, args, name=None, skip_opset_version=None,
               with_warning=False, train=False):
        """Compare model output and test runtime output.

        Make an ONNX model from target model with args, and put output
        directory. Then test runtime load the model, and compare.

        Arguments:
            model (~chainer.Chain): the target model.
            args (list or dict): arguments of the target model.
            name (str): name of test. set class name on default.
            skip_opset_version (list): versions to skip test.
            with_warning (bool): if True, check warnings.
            train (bool): If True, output computational graph with train mode.
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
                        model, args, dir_name, opset_version, train)
                assert len(w) == 1
            else:
                test_path = gen_test_data_set(
                    model, args, dir_name, opset_version, train)

            onnx_model_path = os.path.join(test_path, 'model.onnx')
            assert os.path.isfile(onnx_model_path)
            with open(onnx_model_path, 'rb') as f:
                onnx_model = onnx.load_model(f)
            check_all_connected_from_inputs(onnx_model)

            # TODO(disktnk): some operators such as BatchNormalization are not
            # supported on latest onnxruntime, should skip ONLY not supported
            # operators, but it's hard to write down skip op list.
            if opset_version >= 9:
                continue
            # TODO(disktnk): `input_names` got from onnxruntime session
            # includes only network inputs, does not include internal inputs
            # such as weight attribute etc. so that need to collect network
            # inputs from `onnx_model`.
            graph_input_names = _get_graph_input_names(onnx_model)
            check_model_expect(test_path, input_names=graph_input_names)


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
