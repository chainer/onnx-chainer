import unittest

import onnx

import onnx_chainer


class ONNXModelTest(unittest.TestCase):

    def expect(self, model, args, name, op_name=None):
        minimum_version = onnx_chainer.MINIMUM_OPSET_VERSION
        if op_name is not None:
            opset_ids = onnx_chainer.mapping.operators(op_name)
            minimum_version = max(minimum_version, opset_ids[0])
        for opset_version in range(minimum_version,
                                   onnx.defs.onnx_opset_version() + 1):
            test_name = 'test_{}_opset{}'.format(name, opset_version)
            onnx_chainer.testing.test_onnxruntime.check(
                self.model, args, test_name)
