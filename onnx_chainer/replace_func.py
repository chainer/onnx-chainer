import chainer


class WrappedFunctionNode(chainer.FunctionNode):

    def __init__(self, name, func, args, kwargs):
        self.user_name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def forward(self, xs):
        self.xs = xs
        results = self.func(*self.args, **self.kwargs)
        if isinstance(results, (tuple, list)):
            dummy_results = tuple(
                ret.array if _isvar(ret) else ret for ret in results)
        elif isinstance(results, dict):
            dummy_results = tuple(
                ret.array if _isvar(ret) else ret for ret in results.values())
        else:
            dummy_results = results.array if _isvar(results) else results
            dummy_results = dummy_results,
        return dummy_results

    def backward(self, indexes, gys):
        xp = chainer.backend.get_array_module(self.xs[0])
        ret = tuple(chainer.Variable(xp.zeros_like(x)) for x in self.xs)
        return ret


def fallback(alt_func, name):
    """Fallback the target function

    The target function is replaced to the alternative function to connect
    variable node. ``alt_func`` must satisfy the following restrictions.

    1. Inputs includes one or more ``chainer.Variable`` to trace variables.
    2. Output consists nothing but ``ndarray`` or ``chainer.Variable``

    Even if ``alt_func`` returns ``ndarray``, the value forced to be converted
    to ``chainer.Variable``. A caller of the target function have to care
    both cases, return ``ndarray`` and ``chainer.Variable``.

    Arguments:
        alt_func (func): actual called function. There are some constrains, see
            the above documentation.
        name (str): function name. This name is used for what ONNX operator
            to be assigned.

    Returns:
        func: wrapped function, called on exporting.
    """

    def _wrapper(*args, **kwargs):
        inputs = []
        for arg in args:
            if _isvar(arg):
                inputs.append(arg)
            elif isinstance(arg, (tuple, list)):
                inputs.extend([a for a in arg if _isvar(a)])
        for arg in kwargs.values():
            if _isvar(arg):
                inputs.append(arg)
            elif isinstance(arg, (tuple, list)):
                inputs.extend([a for a in arg if _isvar(a)])
        wrapped = WrappedFunctionNode(name, alt_func, args, kwargs)
        ret = wrapped.apply(inputs)
        if len(ret) > 1:
            return ret
        return ret[0]

    return _wrapper


def overwrap(name):
    """Overwrap the target function

    The target function is overwrapped to connect variable node.
    Expected to use as decorator. More detail, see ``fallback`` documentation.

    Arguments:
        name (str): function name. This name is used for what ONNX operator
            to be assigned.
    """
    def _wrapper(fn):
        return fallback(fn, name)

    return _wrapper


def _isvar(array):
    # alias for type checking
    return isinstance(array, chainer.Variable)
