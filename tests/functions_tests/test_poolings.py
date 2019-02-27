import unittest
import warnings

import chainer
import chainer.functions as F
from chainer import testing
import numpy as np
import onnx

import onnx_chainer
from onnx_chainer.testing import input_generator
from onnx_chainer.testing import test_onnxruntime


@testing.parameterize(
    {'name': 'average_pooling_2d', 'ops': F.average_pooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [2, 1, 0], 'cover_all': None},
    {'name': 'average_pooling_2d', 'ops': F.average_pooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [3, 2, 1], 'cover_all': None},
    {'name': 'average_pooling_nd', 'ops': F.average_pooling_nd,
     'in_shape': (1, 3, 6, 6, 6), 'args': [2, 1, 1], 'cover_all': None},
    {'name': 'max_pooling_2d', 'ops': F.max_pooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [2, 1, 1], 'cover_all': False},
    {'name': 'max_pooling_2d', 'ops': F.max_pooling_2d,
     'in_shape': (1, 3, 6, 5), 'args': [3, (2, 1), 1], 'cover_all': True},
    {'name': 'max_pooling_nd', 'ops': F.max_pooling_nd,
     'in_shape': (1, 3, 6, 6, 6), 'args': [2, 1, 1], 'cover_all': False},
    {'name': 'max_pooling_nd', 'ops': F.max_pooling_nd,
     'in_shape': (1, 3, 6, 5, 4), 'args': [3, 2, 1], 'cover_all': True},
    {'name': 'unpooling_2d', 'ops': F.unpooling_2d,
     'in_shape': (1, 3, 6, 6), 'args': [3, None, 0], 'cover_all': False},
)
class TestPoolings(unittest.TestCase):

    def setUp(self):
        ops = getattr(F, self.name)
        self.model = Model(ops, self.args, self.cover_all)
        self.x = input_generator.increasing(*self.in_shape)
        self.fn = self.name + '.onnx'

    def test_output(self):
        for opset_version in range(
                onnx_chainer.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            # TODO(hamaji): onnxruntime does not support Upsample-9 yet.
            if self.name == 'unpooling_2d' and opset_version == 9:
                continue
            test_onnxruntime.check_output(
                self.model, self.x, self.fn, opset_version=opset_version)


@testing.parameterize(
    {'name': 'max_pooling_2d', 'ops': F.max_pooling_2d,
     'in_shape': (1, 3, 6, 5), 'args': [2, 2, 1], 'cover_all': True},
    {'name': 'max_pooling_nd', 'ops': F.max_pooling_nd,
     'in_shape': (1, 3, 6, 5, 4), 'args': [2, 2, 1], 'cover_all': True},
)
class TestPoolingsWithUnsupportedSettings(unittest.TestCase):

    def setUp(self):
        ops = getattr(F, self.name)
        self.model = Model(ops, self.args, self.cover_all)
        self.x = input_generator.increasing(*self.in_shape)
        self.fn = self.name + '.onnx'

    def test_output(self):
        for opset_version in range(
                onnx_chainer.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            with self.assertRaises(RuntimeError):
                test_onnxruntime.check_output(
                    self.model, self.x, self.fn, opset_version=opset_version)


class Model(chainer.Chain):

    def __init__(self, ops, args, cover_all):
        super(Model, self).__init__()
        self.ops = ops
        self.args = args
        self.cover_all = cover_all

    def __call__(self, x):
        if self.cover_all is not None:
            return self.ops(*([x] + self.args), cover_all=self.cover_all)
        else:
            return self.ops(*([x] + self.args))


class TestROIPooling2D(unittest.TestCase):

    def setUp(self):
        # these parameters are referenced from chainer test
        in_shape = (3, 3, 12, 8)
        self.x = input_generator.positive_increasing(*in_shape)
        # In chainer test, x is shuffled and normalize-like conversion,
        # In this test, those operations are skipped.
        # If x includes negative value, not match with onnxruntime output.
        # You can reproduce this issue by changing `positive_increasing` to
        # `increase`
        self.rois = np.array([
            [0, 1, 1, 6, 6],
            [2, 6, 2, 7, 11],
            [1, 3, 1, 5, 10],
            [0, 3, 3, 3, 3]], dtype=np.float32)
        kwargs = {
            'outh': 3,
            'outw': 7,
            'spatial_scale': 0.6
        }

        class Model(chainer.Chain):
            def __init__(self, kwargs):
                super(Model, self).__init__()
                self.kwargs = kwargs

            def __call__(self, x, rois):
                return F.roi_pooling_2d(x, rois, **self.kwargs)

        self.model = Model(kwargs)
        self.fn = 'roi_pooling_2d.onnx'

    def test_output(self):
        for opset_version in range(
                test_onnxruntime.MINIMUM_OPSET_VERSION,
                onnx.defs.onnx_opset_version() + 1):
            with warnings.catch_warnings(record=True) as w:
                test_onnxruntime.check_output(
                    self.model, [self.x, self.rois], self.fn,
                    opset_version=opset_version)
                assert len(w) == 1
