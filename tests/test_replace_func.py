import os
import unittest
import warnings

import chainer
import chainer.functions as F
import onnx
import pytest

from onnx_chainer import export_testcase
from onnx_chainer import onnx_helper
import onnx_chainer.replace_func as FX
from onnx_chainer.replace_func import as_funcnode
from onnx_chainer.replace_func import fake_as_funcnode
from onnx_chainer.testing import input_generator
from tests.helper import ONNXModelTest


@pytest.fixture(scope='function')
def model():
    # this model, variable node is cut on fn1 and fn2
    class Model(chainer.Chain):
        def __init__(self):
            super().__init__()
            self.fn1 = self.square
            self.fn2 = self.matmul

        def square(self, x):
            return x.array ** 2

        def matmul(self, x1, x2, b):
            return x1@x2.array + b

        def __call__(self, x1, x2):
            h = self.fn1(x1)
            h2 = self.fn2(h, x2, 0.1)
            return F.sigmoid(h2)
    return Model()


@pytest.fixture(scope='function')
def model_dec():
    # similar to model(), decorated
    class Model(chainer.Chain):
        def __init__(self):
            super().__init__()
            self.fn1 = self.square
            self.fn2 = self.matmul

        @as_funcnode('X')
        def square(self, x):
            return x.array ** 2

        @as_funcnode('Y', attributes=['b'])
        def matmul(self, x1, x2, b):
            if isinstance(x1, chainer.Variable):
                x1 = x1.array
            return x1@x2.array + b

        def __call__(self, x1, x2):
            h = self.fn1(x1)
            h2 = self.fn2(h, x2, b=0.1)
            return F.sigmoid(h2)
    return Model()


@pytest.fixture(scope='function')
def addon_converters():
    def custom_converter_x(params):
        return onnx_helper.make_node(
            'Custom_X', params.input_names, len(params.output_names)),

    def custom_converter_y(params):
        return onnx_helper.make_node(
            'Custom_Y', params.input_names, len(params.output_names),
            b=params.func.b),

    return {
        'X': custom_converter_x,
        'Y': custom_converter_y,
    }


def test_fake_as_funcnode_without_replace(tmpdir, model):
    path = str(tmpdir)

    x1 = input_generator.increasing(3, 4)
    x2 = input_generator.increasing(4, 5)

    with warnings.catch_warnings(record=True):
        export_testcase(model, (x1, x2), path)

    model_filepath = os.path.join(path, 'model.onnx')
    assert os.path.isfile(model_filepath)

    onnx_model = onnx.load(model_filepath)
    node_names = {n.name for n in onnx_model.graph.node}
    # no node is generated
    assert len(node_names) == 0


def test_fake_as_funcnode(tmpdir, model, addon_converters):
    path = str(tmpdir)

    x1 = input_generator.increasing(3, 4)
    x2 = input_generator.increasing(4, 5)
    fn2 = model.fn2

    def dummy_fn2(x1, x2, b):
        # wrapped `model.fn1` returns chainer.Variable, so need to care
        # if call `model.fn2` directly, cause recursive loop
        return fn2(x1.array, x2, b)

    model.fn1 = fake_as_funcnode(model.fn1, 'X')
    model.fn2 = fake_as_funcnode(dummy_fn2, 'Y', attributes=[(2, 'b')])

    with warnings.catch_warnings(record=True):
        export_testcase(
            model, (x1, x2), path, external_converters=addon_converters)

    model_filepath = os.path.join(path, 'model.onnx')
    assert os.path.isfile(model_filepath)

    onnx_model = onnx.load(model_filepath)
    node_names = {n.name for n in onnx_model.graph.node}
    assert node_names == {'X_0', 'Y_0', 'Sigmoid_0'}


def test_as_funcnode(tmpdir, model_dec, addon_converters):
    path = str(tmpdir)

    x1 = input_generator.increasing(3, 4)
    x2 = input_generator.increasing(4, 5)

    with warnings.catch_warnings(record=True):
        export_testcase(
            model_dec, (x1, x2), path, external_converters=addon_converters)

    model_filepath = os.path.join(path, 'model.onnx')
    assert os.path.isfile(model_filepath)

    onnx_model = onnx.load(model_filepath)
    node_names = {n.name for n in onnx_model.graph.node}
    assert node_names == {'X_0', 'Y_0', 'Sigmoid_0'}


@unittest.skip('Need to use customized FasterRCNNVGG16 model')
def test_faster_rcnn(tmpdir):
    path = str(tmpdir)

    from chainercv.links import FasterRCNNVGG16
    model = FasterRCNNVGG16(
        n_fg_class=12,
        pretrained_model=None)
    x = input_generator.increasing(1, 3, 244, 244)
    rpn_pl = model.rpn.proposal_layer

    import chainer.backend
    import numpy as np

    def dummy_rpn_pl(i, rpn, *args, **kwargs):
        xp = chainer.backend.get_array_module(rpn)
        roi = rpn_pl(rpn.array, *args, **kwargs)
        batch_index = i * xp.ones((len(roi),), dtype=np.int32)
        return roi, batch_index

    model.rpn.proposal_layer = fake_as_funcnode(
        dummy_rpn_pl, 'ProposalCreator')

    def custom_converter_pl(params):
        return onnx_helper.make_node(
            'ChainerRPNProposalCreator', params.input_names,
            len(params.output_names)),
    addon_converters = {'ProposalCreator': custom_converter_pl}

    with warnings.catch_warnings(record=True):
        export_testcase(model, x, path, external_converters=addon_converters)


@unittest.skip('Need to use customized FasterRCNNFPNResNet50 model')
def test_fpn(tmpdir):
    path = str(tmpdir)

    from chainercv.links import FasterRCNNFPNResNet50
    model = FasterRCNNFPNResNet50(
        n_fg_class=80,
        pretrained_model='coco')
    x = input_generator.increasing(1, 3, 1024, 1024)

    org_rpn_decode = model.rpn.decode
    org_head_dist = model.head.distribute

    def dummy_rpn_decode(hs, locs, confs, in_shape):
        anchors = model.rpn.anchors(h.shape[2:] for h in hs)
        rois, roi_indices = org_rpn_decode(locs, confs, anchors, in_shape)
        return rois, roi_indices

    def dummy_head_distribute(rois, roi_indices):
        # if call `model.head.distribute` directly, cause recursive loop
        rois, roi_indices = org_head_dist(rois.array, roi_indices.array)
        return tuple(rois) + tuple(roi_indices)

    model.rpn.decode = fake_as_funcnode(dummy_rpn_decode, 'FPN_RPN_Decode')
    model.head.distribute = fake_as_funcnode(
        dummy_head_distribute, 'FPN_Head_Distribute')

    def custom_converter_roi_ave_align_2d(params):
        return onnx_helper.make_node(
            'ChainerROIAverageAlign2D', params.input_names,
            len(params.output_names)),

    def custom_converter_rpn_decode(params):
        return onnx_helper.make_node(
            'ChainerRPNDecode', params.input_names,
            len(params.output_names)),

    def custom_converter_fpn_head_distribute(params):
        return onnx_helper.make_node(
            'ChainerFPNHeadDistribute', params.input_names,
            len(params.output_names)),

    addon_converters = {
        'ROIAverageAlign2D': custom_converter_roi_ave_align_2d,
        'FPN_RPN_Decode': custom_converter_rpn_decode,
        'FPN_Head_Distribute': custom_converter_fpn_head_distribute,
    }

    with warnings.catch_warnings(record=True):
        export_testcase(model, x, path, external_converters=addon_converters)


class TestShape(ONNXModelTest):

    def test_output(self):
        model = chainer.Sequential(
            FX.shape
        )
        x = input_generator.increasing(2, 3)
        self.expect(model, x)
