import collections
import onnx


__func_name = None
__func_to_id = collections.defaultdict(int)


def set_func_name(func_name):
    """Set the name of Chainer function being converted.

    Args:
      func_name (str): The name of Chainer function.
    """
    global __func_name
    __func_name = func_name


def gensym():
    """Returns a unique symbol.

    Returns:
      A unique string symbol.
    """
    assert __func_name is not None
    __func_to_id[__func_name] += 1
    return 'tmp{}_{}'.format(__func_name, __func_to_id[__func_name])


def make_node(op_name, input_names, num_outputs, **kwargs):
    """A thin wrapper of `onnx.helper.make_node`.

    Unlike `onnx.helper.make_node`, this function takes the number of
    output values instead of the names of them. Unique names will be
    assigned automatically.

    Args:
      op_name (str): The name of an ONNX op.
      input_names (list of str): The names of input values.
      num_outputs (int): The number of output values.
      **kwargs (dict): ONNX attributes of the node.

    Returns:
      An `onnx.NodeProto` object.
    """
    output_names = [gensym() for i in range(num_outputs)]
    return onnx.helper.make_node(op_name, input_names, output_names, **kwargs)


class GraphBuilder(object):
    """A helper class to build consecutive ONNX nodes."""

    def __init__(self):
        self._nodes = []

    def op(self, op_name, input_names, num_outputs=1, **kwargs):
        """Creates a new ONNX node and returns its outputs.

        Args:
          op_name (str): The name of an ONNX op.
          input_names (list of str): The names of input values.
          num_outputs (int): The number of output values.
          **kwargs (dict): ONNX attributes of the node.

        Returns:
          A str of the output name when `num_outputs` is 1.
          A tuple of str of the output names otherwise.
        """
        # Prevent a common mistake. `input_names="input"` creates a
        # node with 5 inputs.
        assert not isinstance(input_names, str)
        node = make_node(op_name, input_names, num_outputs, **kwargs)
        self._nodes.append(node)
        if num_outputs == 1:
            return node.output[0]
        else:
            return tuple(node.output)

    def const(self, array):
        """Creates a Constant node of ONNX.

        Args:
          array (numpy.ndarray): A numpy array.

        Returns:
          A str of the name of the constant value.
        """
        tensor = onnx.numpy_helper.from_array(array)
        return self.op('Constant', [], 1, value=tensor)

    def nodes(self):
        """Returns all nodes created so far.

        Returns:
          A list of `onnx.NodeProto` objects, suitable as the return
          value of converter functions.
        """
        return tuple(self._nodes)


def write_tensor_pb(filename, name, value):
    with open(filename, 'wb') as f:
        t = onnx.numpy_helper.from_array(value, name)
        f.write(t.SerializeToString())


def cleanse_param_name(name):
    """Converts Chainer parameter names to ONNX names.

    Note ONNX identifiers must be a valid C identifier.

    Args:
      name (str): A Chainer parameter name (e.g., /l/W).

    Returns
      A valid ONNX name (e.g., param_l_W).
    """
    return 'param' + name.replace('/', '_')
