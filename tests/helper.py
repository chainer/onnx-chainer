import unittest
import warnings

import onnx
import pytest

import onnx_chainer
from onnx_chainer.testing.get_test_data_set import gen_test_data_set
from onnx_chainer.testing.test_onnxruntime import check_model_expect


class ONNXModelTest(unittest.TestCase):

    @pytest.fixture(autouse=True, scope='function')
    def set_name(self, request):
        cls_name = request.cls.__name__
        self.default_name = cls_name[len('Test'):].lower()

    def expect(self, model, args, name=None, op_name=None,
               skip_opset_version=None, with_warning=False):
        """Compare model output and test runtime output.

        Make an ONNX model from target model with args, and put output
        directory. Then test runtime load the model, and compare.

        Arguments:
            model (~chainer.Chain): the target model.
            args (list or dict): arguments of the target model.
            name (str): name of test. set class name on default.
            op_name (str): name of operator. use for getting opset verseion.
            skip_opset_version (list): versions to skip test.
            with_warning (bool): if set `True`, check warnings.
        """

        minimum_version = onnx_chainer.MINIMUM_OPSET_VERSION
        if op_name is not None:
            opset_ids = onnx_chainer.mapping.operators[op_name]
            minimum_version = max(minimum_version, opset_ids[0])
        test_name = name
        if test_name is None:
            test_name = self.default_name

        for opset_version in range(minimum_version,
                                   onnx.defs.onnx_opset_version() + 1):
            if skip_opset_version is not None and\
                    opset_version in skip_opset_version:
                continue

            dir_name = 'test_' + test_name
            if with_warning:
                with warnings.catch_warnings(record=True) as w:
                    test_path = gen_test_data_set(
                        model, args, dir_name, opset_version)
                assert len(w) == 1
            else:
                test_path = gen_test_data_set(
                    model, args, dir_name, opset_version)

            check_model_expect(test_path)
