import chainer
from chainer import testing

from onnx_chainer.testing import input_generator
from tests.helper import ONNXModelTest


@testing.parameterize(
    {'op_name': 'Neg', 'ops': '-a'},
    {'op_name': 'Absolute', 'ops': 'abs(a)'},
    {'op_name': 'Clip', 'ops': 'chainer.functions.clip(a, 0.1, 0.2)'},
    {'op_name': 'Exp', 'ops': 'chainer.functions.exp(a)'},
    {'op_name': 'Sqrt', 'ops': 'chainer.functions.sqrt(a)'},
    {'op_name': 'PowVarConst',
     'ops': 'chainer.functions.math.basic_math.pow(a, 2)'},
    {'op_name': 'Sum', 'ops': 'chainer.functions.sum(a)'},
    {'op_name': 'Sum', 'ops': 'chainer.functions.sum(a, axis=1)',
     'condition': 'axis1'},
    {'op_name': 'Sum', 'ops': 'chainer.functions.sum(a, keepdims=True)',
     'condition': 'keepdims'},
    {'op_name': 'AddConstant', 'ops': 'a + 1'},
    {'op_name': 'Max', 'ops': 'chainer.functions.max(a)'},
    {'op_name': 'Max', 'ops': 'chainer.functions.max(a, axis=0)',
     'condition': 'axis0'},
    {'op_name': 'Max', 'ops': 'chainer.functions.max(a, keepdims=True)',
     'condition': 'keepdims'},
    {'op_name': 'Mean', 'ops': 'chainer.functions.mean(a)'},
    {'op_name': 'Mean', 'ops': 'chainer.functions.mean(a, axis=0)',
     'condition': 'axis0'},
    {'op_name': 'Mean', 'ops': 'chainer.functions.mean(a, keepdims=True)',
     'condition': 'keepdims'},
    {'op_name': 'Min', 'ops': 'chainer.functions.min(a)'},
    {'op_name': 'Min', 'ops': 'chainer.functions.min(a, axis=0)',
     'condition': 'axis0'},
    {'op_name': 'Min', 'ops': 'chainer.functions.min(a, keepdims=True)',
     'condition': 'keepdims'},
    {'op_name': 'Prod', 'ops': 'chainer.functions.prod(a)'},
    {'op_name': 'Prod', 'ops': 'chainer.functions.prod(a, axis=0)',
     'condition': 'axis0'},
    {'op_name': 'Prod', 'ops': 'chainer.functions.prod(a, keepdims=True)',
     'condition': 'keepdims'},
    {'op_name': 'LogSumExp', 'ops': 'chainer.functions.logsumexp(a)'},
    {'op_name': 'LogSumExp', 'ops': 'chainer.functions.logsumexp(a, axis=0)',
     'condition': 'axis0'},
    {'op_name': 'Square', 'ops': 'chainer.functions.square(a)'},
    {'op_name': 'BroadcastTo',
     'ops': 'chainer.functions.broadcast_to(a, (2,2,3))'},
)
class TestUnaryMathOperators(ONNXModelTest):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a):
                if not isinstance(a, chainer.Variable):
                    a = chainer.Variable(a)
                return eval(self.ops)

        self.model = Model(self.ops)
        self.a = chainer.Variable(input_generator.positive_increasing(2, 3))

    def test_output(self):
        name = self.op_name.lower()
        if hasattr(self, 'condition'):
            name += '_' + self.condition
        skip_opset_version = []
        if self.op_name == 'BroadcastTo':
            skip_opset_version.append(7)
        self.expect(self.model, self.a, name=name,
                    skip_opset_version=skip_opset_version,
                    expected_num_initializers=0)


@testing.parameterize(
    {'op_name': 'Add', 'ops': 'a + b'},
    {'op_name': 'Sub', 'ops': 'a - b'},
    {'op_name': 'Mul', 'ops': 'a * b'},
    {'op_name': 'Div', 'ops': 'a / b'},
    {'op_name': 'MatMul',
     'ops': 'chainer.functions.matmul(a, b, transb=True)'},
    {'op_name': 'Maximum', 'ops': 'chainer.functions.maximum(a, b)'},
    {'op_name': 'Minimum', 'ops': 'chainer.functions.minimum(a, b)'},
)
class TestBinaryMathOperators(ONNXModelTest):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a, b):
                if not isinstance(a, chainer.Variable):
                    a = chainer.Variable(a)
                if not isinstance(b, chainer.Variable):
                    b = chainer.Variable(b)
                return eval(self.ops)

        self.model = Model(self.ops)
        a = chainer.Variable(input_generator.increasing(2, 3))
        b = chainer.Variable(input_generator.nonzero_increasing(2, 3) * 0.3)
        self.x = (a, b)

    def test_output(self):
        name = self.op_name.lower()
        self.expect(self.model, self.x, name=name,
                    expected_num_initializers=0)


@testing.parameterize(
    {'op_name': 'LinearInterpolate',
     'ops': 'chainer.functions.linear_interpolate(a, b, c)'},
)
class TestTernaryMathOperators(ONNXModelTest):

    def setUp(self):
        class Model(chainer.Chain):

            def __init__(self, ops):
                super(Model, self).__init__()
                self.ops = ops

            def __call__(self, a, b, c):
                if not isinstance(a, chainer.Variable):
                    a = chainer.Variable(a)
                if not isinstance(b, chainer.Variable):
                    b = chainer.Variable(b)
                if not isinstance(c, chainer.Variable):
                    c = chainer.Variable(c)
                return eval(self.ops)

        self.model = Model(self.ops)
        a = chainer.Variable(input_generator.increasing(2, 3))
        b = chainer.Variable(input_generator.increasing(2, 3) * 0.3)
        c = chainer.Variable(input_generator.increasing(2, 3) * 0.7)
        self.x = (a, b, c)

    def test_output(self):
        name = self.op_name.lower()
        self.expect(self.model, self.x, name=name,
                    expected_num_initializers=0)
