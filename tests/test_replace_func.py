import os
import warnings

import chainer
import chainer.functions as F
import onnx
import pytest

from onnx_chainer import export_testcase
from onnx_chainer import onnx_helper
from onnx_chainer.replace_func import as_funcnode
from onnx_chainer.replace_func import fake_as_funcnode
from onnx_chainer.testing import input_generator


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
