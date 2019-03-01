import unittest

import onnx
import pytest

import onnx_chainer


class ONNXModelTest(unittest.TestCase):

    @pytest.fixture(autouse=True, scope='function')
    def set_name(self, request):
        cls_name = request.cls.__name__
        self.default_name = cls_name[len('Test'):].lower()

    def expect(self, model, args, name=None, op_name=None):
        minimum_version = onnx_chainer.MINIMUM_OPSET_VERSION
        if op_name is not None:
            opset_ids = onnx_chainer.mapping.operators(op_name)
            minimum_version = max(minimum_version, opset_ids[0])
        test_name = name
        if test_name is None:
            test_name = self.default_name

        for opset_version in range(minimum_version,
                                   onnx.defs.onnx_opset_version() + 1):
            dir_name = 'test_{}_opset{}'.format(test_name, opset_version)
            onnx_chainer.testing.test_onnxruntime.check(
                self.model, args, dir_name)
