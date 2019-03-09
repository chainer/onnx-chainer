class Context(object):

    def __init__(self, model):
        self.name_list = dict()
        for name, param in model.namedparams():
            replaced_name = 'param' + name.replace('/', '_')
            self.name_list[str(id(param))] = replaced_name

    def get_name(self, variable):
        str_id = str(id(variable))
        if str_id in self.name_list:
            return self.name_list[str_id]
        else:
            new_name = 'v{}'.format(len(self.name_list))
            self.name_list[str_id] = new_name
            return new_name
