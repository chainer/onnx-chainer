import chainer

from onnx_chainer import onnx_helper


class Context(object):
    """context of converter

    This context shares names during exporting.

    Arguments:
        name_list (dict): list of being exported as ONNX node name keyed by
            instance ID. When the target variable is ``chainer.Variable`` or
            ``chainer.Parameter``, instance ID of data array held by the
            variable is also put as key, because some functions has data array
            instead of variable.

    """

    def __init__(self, model):
        self.name_list = dict()
        for name, param in model.namedparams():
            assert isinstance(param, chainer.Parameter)
            onnx_name = onnx_helper.cleanse_param_name(name)
            self.name_list[str(id(param))] = onnx_name
            self.name_list[str(id(param.array))] = onnx_name

    def get_name(self, variable):
        str_id = str(id(variable))
        if str_id in self.name_list:
            return self.name_list[str_id]
        else:
            new_name = 'v{}'.format(len(self.name_list))
            self.name_list[str_id] = new_name
            if isinstance(variable, (chainer.Variable, chainer.Parameter)):
                array_id = str(id(variable.array))
                self.name_list[array_id] = new_name
            return new_name

    def set_name(self, variable, name):
        str_id = str(id(variable))
        self.name_list[str_id] = name
        if isinstance(variable, (chainer.Variable, chainer.Parameter)):
            array_id = str(id(variable.array))
            self.name_list[array_id] = name
