import inspect
from inspect import Parameter

import chainer


class WrappedFunctionNode(chainer.FunctionNode):
    """Wrap the target function and operate as ``FunctionNode``

    Arguments:
        name (str): name of the function node
        func (func): the target function
        args (str): args for the function
        kwargs (str): kwargs for the function
        attributes (list): parameters to be set node's attributes
    """

    def __init__(self, name, func, args, kwargs, attributes=None):
        self.user_name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs

        if attributes is not None:
            self.set_attributes(attributes)

    def set_attributes(self, attributes):
        for p in attributes:
            if isinstance(p, tuple):
                assert isinstance(p[0], int)
                setattr(self, p[1], self.args[p[0]])
            elif isinstance(p, str):
                setattr(self, p, self.kwargs.get(p, None))

    def forward(self, xs):
        self.xs = xs
        results = self.func(*self.args, **self.kwargs)
        if isinstance(results, (tuple, list)):
            dummy_results = tuple(_unwrap_var(ret) for ret in results)
        elif isinstance(results, dict):
            dummy_results = tuple(_unwrap_var(ret) for ret in results.values())
        else:
            dummy_results = _unwrap_var(results)
            dummy_results = dummy_results,
        if not chainer.is_arrays_compatible(dummy_results):
            raise ValueError(
                'returned values from the function wrapped by \'as_funcnode\' '
                'must consist only array, function name: {}'.format(self.name))
        return dummy_results

    def backward(self, indexes, gys):
        xp = chainer.backend.get_array_module(self.xs[0])
        ret = tuple(chainer.Variable(xp.zeros_like(x)) for x in self.xs)
        return ret


def fake_as_funcnode(alt_func, name, attributes=None):
    """The target function fakes FunctionNode

    The target function is replaced to the alternative function to connect
    variable node by acting function node. ``alt_func`` must satisfy the
    following restrictions.

    1. Inputs includes one or more ``chainer.Variable`` to trace variables.
    2. Output consists nothing but ``ndarray`` or ``chainer.Variable``

    Even if ``alt_func`` returns ``ndarray``, the value forced to be converted
    to ``chainer.Variable``. A caller of the target function have to care
    both cases, returning ``ndarray`` and ``chainer.Variable``.

    When ``alt_func`` returns ``list`` of variable, the wrapped function will
    also returns multiple variables as ``tuple``. However ``dict`` cannot
    be return, the wrapped function breaks down the returned values as
    ``tuple`` of values, keys will be ignored.

    Example::

       >>> model.func = fake_as_funcnode(model.func, 'CustomNode')

    Then ``model.func`` will be operated as a function node named "CustomNode".
    See tests/test_replace_func.py more details.

    Args:
        alt_func (func): actual called function. There are some constrains, see
            the above documentation.
        name (str): function name. This name is used for what ONNX operator
            to be assigned.
        attributes (list): to set as function param. the list should be
            ``tuple`` as ``(index of args, name)`` or key name of ``kwargs``.

    Returns:
        func: wrapped function, called on exporting.
    """

    def _wrapper(*args, **kwargs):
        inputs = list(filter(_is_var, _flatten(args)))
        inputs.extend(list(filter(_is_var, _flatten(list(kwargs.values())))))
        if not inputs:
            raise ValueError(
                'arguments of the function wrapped by \'as_funcnode\' '
                'must include at least one chainer.Variable, function name: '
                '{}'.format(name))

        arg_spec = inspect.signature(alt_func)
        bound = arg_spec.bind(*args, **kwargs)
        bound.apply_defaults()
        # default values are set on `bound.arguments`, but cannot get them
        # from `bound.kwargs`
        for i, (k, v) in enumerate(bound.arguments.items()):
            if i < len(args):
                continue
            kwargs[k] = v

        wrapped = WrappedFunctionNode(
            name, alt_func, args, kwargs, attributes=attributes)
        ret = wrapped.apply(inputs)
        if len(ret) > 1:
            return ret
        return ret[0]

    chainer.utils.experimental('as_funcnode')
    return _wrapper


def as_funcnode(name, attributes=None):
    """The target function fakes FunctionNode

    The target function is overwrapped to connect variable node by acting
    function node. Expected to be used as decorator. More detail, see
    ``fake_as_funcnode`` documentation.

    Example::

       >>> class Model(chainer.Chain):
       >>>     @as_funcnode('CustomNode')
       >>>     def func(self, *args, **kwargs):
       >>>         pass

    Then ``model.func`` will be operated as a function node named "CustomNode".
    See tests/test_replace_func.py more details.

    Args:
        name (str): function name. This name is used for what ONNX operator
            to be assigned.
        attributes (list): to set as function param. the list should be
            ``tuple`` as ``(index of args, name)`` or key name of ``kwargs``.
    """
    def _wrapper(fn):
        return fake_as_funcnode(fn, name, attributes=attributes)

    return _wrapper


def _unwrap_var(var):
    return var.array if _is_var(var) else var


def _is_var(array):
    # alias for type checking
    return isinstance(array, chainer.Variable)


def _is_array(v):
    return not isinstance(v, (list, tuple))


def _flatten(xs):
    if _is_array(xs):
        return [xs]

    o = []
    for x in xs:
        if _is_array(x):
            o.append(x)
        else:
            o.extend(_flatten(x))
    return o
