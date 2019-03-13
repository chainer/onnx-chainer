from onnx_chainer import onnx_helper


class Context(object):

    def __init__(self, model):
        self.name_list = dict()
        for name, param in model.namedparams():
            onnx_name = onnx_helper.cleanse_param_name(name)
            self.name_list[str(id(param))] = onnx_name

    def get_name(self, variable):
        str_id = str(id(variable))
        if str_id in self.name_list:
            return self.name_list[str_id]
        else:
            new_name = 'v{}'.format(len(self.name_list))
            self.name_list[str_id] = new_name
            return new_name

    def set_name(self, variable, name):
        str_id = str(id(variable))
        self.name_list[str_id] = name
