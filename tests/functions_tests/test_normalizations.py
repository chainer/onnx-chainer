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
    {'kwargs': {'use_beta': False}, 'condition': 'use_beta_false'},
    {'kwargs': {'use_gamma': False}, 'condition': 'use_gamma_false'},
    {'train': True, 'kwargs': {}},
    {'train': True,
     'kwargs': {'use_beta': False}, 'condition': 'use_beta_false'},
    {'train': True,
     'kwargs': {'use_gamma': False}, 'condition': 'use_gamma_false'},
    {'train': True,
     'kwargs': {'initial_avg_mean': 0.5}, 'condition': 'init_avg_mean'},
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
        train = getattr(self, 'train', False)
        name = 'batch_normalization'
        if not train:
            name = 'fixed_' + name
        if hasattr(self, 'condition'):
            name += '_' + self.condition

        def test_input_names(onnx_model):
            input_names = set(v.name for v in onnx_model.graph.input)
            assert 'param_bn_avg_mean' in input_names
            assert 'param_bn_avg_var' in input_names

        self.expect(
            self.model, self.x, name=name, train=train,
            customized_model_test_func=test_input_names)
