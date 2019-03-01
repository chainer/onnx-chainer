import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing

from onnx_chainer.testing import input_generator
from tests.helper import ONNXModelTest


@testing.parameterize(
    {
        'name': 'local_response_normalization',
        'input_argname': 'x',
        'args': {'k': 1, 'n': 3, 'alpha': 1e-4, 'beta': 0.75},
        'opset_version': 1
    },
    {
        'name': 'normalize',
        'input_argname': 'x',
        'args': {'axis': 1},
        'opset_version': 1
    }
)
class TestNormalizations(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args, input_argname):
                super(Model, self).__init__()
                self.ops = ops
                self.args = args
                self.input_argname = input_argname

            def __call__(self, x):
                self.args[self.input_argname] = x
                return self.ops(**self.args)

        ops = getattr(F, self.name)
        self.model = Model(ops, self.args, self.input_argname)
        self.x = input_generator.increasing(2, 5, 3, 3)

    def test_output(self):
        self.expect(self.model, self.x, name=self.name)


@testing.parameterize(
    {'kwargs': {}},
    {'kwargs': {'use_beta': False}, 'name': 'use_beta_false'},
    {'kwargs': {'use_gamma': False}, 'name': 'use_gamma_false'},
)
class TestBatchNormalization(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, **kwargs):
                super(Model, self).__init__()
                with self.init_scope():
                    self.bn = L.BatchNormalization(5, **kwargs)

            def __call__(self, x):
                return self.bn(x)

        self.model = Model(**self.kwargs)
        self.x = input_generator.increasing(2, 5)

    def test_output(self):
        name = 'batchnormalization'
        if hasattr(self, 'name'):
            name += '_' + self.name
        self.expect(self.model, self.x, name=name)
