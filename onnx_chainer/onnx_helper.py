import collections
import onnx


__func_name = None
__func_to_id = collections.defaultdict(int)


def set_func_name(func_name):
    global __func_name
    __func_name = func_name


def gensym():
    assert __func_name is not None
    __func_to_id[__func_name] += 1
    return 'tmp{}_{}'.format(__func_name, __func_to_id[__func_name])


def make_node(op_name, input_names, output_spec=1, **kwargs):
    if isinstance(output_spec, int):
        output_names = [gensym() for i in range(output_spec)]
    else:
        output_names = output_spec
    return onnx.helper.make_node(op_name, input_names, output_names, **kwargs)
