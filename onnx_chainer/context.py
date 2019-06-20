import chainer

from onnx_chainer import onnx_helper


class Context(object):
    """Context of converter

    This context shares names during exporting.

    Attributes:
        name_list (dict): list of being exported as ONNX node name with pinned
            or not, keyed by instance ID. When the target variable is
            ``chainer.Variable`` or ``chainer.Parameter``, instance ID of
            ``ndarray`` held by the variable is also put as key, because some
            functions like ``F.where`` internally unwrap variable.

    """

    def __init__(self, model):
        self.name_list = dict()
        self.parameters = []
        for name, param in model.namedparams():
            onnx_name = onnx_helper.cleanse_param_name(name)
            self.set_name(param, onnx_name)

    def get_name(self, variable):
        str_id = id(variable)
        if str_id in self.name_list:
            return self.name_list[str_id][0]
        else:
            new_name = 'v{}'.format(len(self.name_list))
            self.set_name(variable, new_name)
            return new_name

    def set_name(self, variable, name, pinned=False):
        """Set ONNX node name

        Arguments:
            variable (var): target variable
            name (str): name to be exported as ONNX node name
            pinned (bool): if ``True``, the name will not be overwritten in
                subsequence process.
        """

        str_id = id(variable)
        assert str_id not in self.name_list or not self.name_list[str_id][1]
        self.name_list[str_id] = (name, pinned)
        if isinstance(variable, (chainer.Variable, chainer.Parameter)):
            array_id = id(variable.array)
            self.name_list[array_id] = (name, pinned)

    def is_pinned(self, variable):
        str_id = id(variable)
        if str_id not in self.name_list:
            return False
        return self.name_list[str_id][1]

    def add_param(self, array, name, use_original_name=False):
        """Add array to context parameter

        To be converted as ONNX tensor.

        Returns:
            (str) registered name.
        """
        param = chainer.Parameter(array)
        if use_original_name:
            onnx_name = name
        else:
            if not (name.startswith('/') or name.startswith('_')):
                name = '/' + name
            onnx_name = '{}_{}'.format(
                onnx_helper.get_func_name(),
                onnx_helper.cleanse_param_name(name))
        self.set_name(param, onnx_name)
        self.parameters.append(param)
        return onnx_name
