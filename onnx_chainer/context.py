class Context(object):

    def __init__(self, model):
        self.namedparams = dict()
        for name, param in model.namedparams():
            self.namedparams[str(id(param))] = name
        self.name_list = dict()

    def get_name(self, variable):
        str_id = str(id(variable))
        if str_id in self.namedparams:
            return self.namedparams[str_id]
        elif str_id in self.name_list:
            return self.name_list[str_id]
        else:
            new_name = 'v{}'.format(len(self.name_list))
            self.name_list[str_id] = new_name
            return new_name
