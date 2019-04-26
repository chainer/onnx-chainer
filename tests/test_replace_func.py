import os
import warnings

import chainer
import chainer.functions as F
from chainer import testing
import onnx
import pytest

from onnx_chainer import export_testcase
from onnx_chainer import onnx_helper
from onnx_chainer.replace_func import as_funcnode
from onnx_chainer.replace_func import fake_as_funcnode
from onnx_chainer.testing import input_generator
from tests.helper import ONNXModelTest


def test_fake_as_funcnode_without_replace(tmpdir):
    path = str(tmpdir)

    class Model(chainer.Chain):
        def _init__(self):
            super().__init__()

        def add(self, xs, value=0.01):
            return xs.array + value

        def __call__(self, xs):
            return F.sigmoid(self.add(xs))

    model = Model()
    x = input_generator.increasing(3, 4)

    with warnings.catch_warnings(record=True):
        export_testcase(model, x, path)

    model_filepath = os.path.join(path, 'model.onnx')
    assert os.path.isfile(model_filepath)

    onnx_model = onnx.load(model_filepath)
    node_names = {n.name for n in onnx_model.graph.node}
    # no node is generated
    assert len(node_names) == 0


@testing.parameterize(
    {'func_kind': 'list', 'in_shape': (2, 3, 4), 'op_type': 'Add'},
    {'func_kind': 'list_kwargs', 'in_shape': (2, 3, 4), 'op_type': 'Add'},
    {'func_kind': 'var_with_dec', 'in_shape': (3, 4),
     'op_type': 'AddConstant'},
    {'func_kind': 'var_kwargs', 'in_shape': (3, 4), 'op_type': 'AddConstant'},
    {'func_kind': 'var', 'in_shape': (3, 4), 'op_type': 'AddConstant'},
)
class TestReplaceFunc(ONNXModelTest):

    def get_model(self, target_func, input_converter=None):
        class Model(chainer.Chain):
            def __init__(self, target_func, input_converter=None):
                super().__init__()
                self.input_converter = input_converter
                self.fn = target_func

            def __call__(self, xs):
                if self.input_converter is not None:
                    args, kwargs = self.input_converter(xs)
                h = self.fn(*args, **kwargs)
                return F.sigmoid(h)

        return Model(target_func, input_converter)

    def test_output(self):
        attr = None
        is_dec = False
        if self.func_kind == 'list':
            def input_converter(xs):
                return ([xs[0], xs[1]],), {}

            def target_func(xs):
                return xs[0].array + xs[1].array

        elif self.func_kind == 'list_kwargs':
            def input_converter(xs):
                return (), {'xs': [xs[0], xs[1]]}

            def target_func(xs=None):
                assert xs is not None
                return xs[0].array + xs[1].array

        elif self.func_kind == 'var_with_dec':
            def input_converter(xs):
                return (xs,), {}

            @as_funcnode('AddConstant', attributes=['value'])
            def target_func(x, value=0.01):
                return x.array + value

            is_dec = True

        elif self.func_kind == 'var_kwargs':
            def input_converter(xs):
                return (), {'x': xs, 'value': 0.02}

            def target_func(x=None, value=0.01):
                assert x is not None
                return x.array + value

            attr = ['value']

        else:
            assert self.func_kind == 'var'

            def input_converter(xs):
                return (xs, 0.01), {}

            def target_func(x, value):
                return x.array + value

            attr = [(1, 'value')]

        model = self.get_model(target_func, input_converter)
        x = input_generator.increasing(*self.in_shape)

        if not is_dec:
            model.fn = fake_as_funcnode(
                model.fn, self.op_type, attributes=attr)

        name = 'replace_func_' + self.func_kind
        self.expect(model, x, name=name)


@pytest.mark.parametrize('return_type', ['list', 'dict'])
def test_replace_func_collection_return(tmpdir, return_type):
    path = str(tmpdir)

    class Model(chainer.Chain):
        def __init__(self, return_type):
            super().__init__()
            self.return_type = return_type

        def tiled_array(self, xs, n=5):
            if self.return_type == 'list':
                return [xs.array * i for i in range(1, 1+n)]
            else:
                self.return_type == 'dict'
                return {str(i): xs.array * i for i in range(1, 1+n)}

        def __call__(self, xs):
            return self.tiled_array(xs)

    model = Model(return_type)
    x = input_generator.increasing(1, 5)

    model.tiled_array = fake_as_funcnode(
        model.tiled_array, 'xTiledArray', attributes=['n'])

    def tiled_array_converter(params):
        return onnx_helper.make_node(
            'xTiledArray', params.input_names, len(params.output_names)),

    addon_converters = {'xTiledArray': tiled_array_converter}

    with warnings.catch_warnings(record=True):
        export_testcase(model, x, path, external_converters=addon_converters)

    model_filepath = os.path.join(path, 'model.onnx')
    assert os.path.isfile(model_filepath)

    onnx_model = onnx.load(model_filepath)
    node_names = [n.name for n in onnx_model.graph.node]
    assert len(node_names) == 1
    assert node_names[0] == 'xTiledArray_0'
    output_names = [n.name for n in onnx_model.graph.output]
    assert len(output_names) == 5
    for i, name in enumerate(output_names):
        assert name == 'xTiledArray_0_{:d}'.format(i)
