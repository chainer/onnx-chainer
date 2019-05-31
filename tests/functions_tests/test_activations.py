import chainer
import chainer.functions as F
import chainer.links as L
from chainer import testing

from onnx_chainer.testing import input_generator
from tests.helper import ONNXModelTest


@testing.parameterize(
    {'name': 'clipped_relu'},
    {'name': 'elu'},
    {'name': 'hard_sigmoid'},
    {'name': 'leaky_relu'},
    {'name': 'log_softmax'},
    {'name': 'log_softmax',
     'args': {'axis': 0}},
    {'name': 'log_softmax',
     'args': {'axis': 2}},
    {'name': 'relu'},
    {'name': 'sigmoid'},
    {'name': 'softmax'},
    {'name': 'softmax',
     'args': {'axis': 0}},
    {'name': 'softmax',
     'args': {'axis': 2}},
    {'name': 'softplus'},
    {'name': 'tanh'},
)
class TestActivations(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self, ops, args):
                super(Model, self).__init__()
                self.ops = ops
                self.args = args

            def __call__(self, x):
                return self.ops(x, **self.args)

        ops = getattr(F, self.name)
        args = {}
        if hasattr(self, 'args'):
            args = self.args
        self.model = Model(ops, args)
        self.x = input_generator.increasing(2, 5, 3)

    def test_output(self):
        self.expect(self.model, self.x, self.name)


class TestPReLU(ONNXModelTest):

    def setUp(self):

        class Model(chainer.Chain):

            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.prelu = L.PReLU()

            def __call__(self, x):
                return self.prelu(x)

        self.model = Model()
        self.x = input_generator.increasing(2, 5)

    def test_output(self):
        self.expect(self.model, self.x)
