# Currently, the test for SoftmaxCrossEntropy is disabled since onnxruntime
# does not support OneHot node. After OneHot node becomes available, we may
# use the comment-out part for testing.

# import unittest
#
# import chainer
# from chainer import testing
# import numpy as np
#
# from onnx_chainer.testing import test_onnxruntime
#
#
# @testing.parameterize(
#     {'in_shape': (3, 5)},
# )
# class TestSoftmaxCrossEntropy(unittest.TestCase):
#
#     def setUp(self):
#
#         class Model(chainer.Chain):
#             def __init__(self):
#                 super(Model, self).__init__()
#
#             def __call__(self, x, t):
#                 return chainer.functions.softmax_cross_entropy(x, t)
#
#         self.model = Model()
#         self.x = np.random.uniform(size=self.in_shape).astype('f')
#         self.t = np.random.randint(size=self.in_shape[0], low=0,
#                                    high=self.in_shape[1]).astype(np.int32)
#
#     def test_output(self):
#         for opset_version in (9,):
#             test_onnxruntime.check_output(
#                 self.model, [self.x, self.t],
#                 'softmax_cross_entropy.onnx',
#                 opset_version=opset_version)
