from __future__ import print_function

import collections
import warnings

import chainer
import onnx
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from onnx_chainer.context import Context
from onnx_chainer.functions.converter import FunctionConverterParams
from onnx_chainer import mapping
from onnx_chainer import onnx_helper

try:
    from onnx import checker
    from onnx import helper
    from onnx import numpy_helper

    _available = True
except ImportError:
    _available = False

MINIMUM_OPSET_VERSION = 7


def _check_available():
    if not _available:
        raise ImportError(
            'ONNX is not installed on your environment. Exporting your model '
            'in ONNX format needs the onnx package.\n\n'
            '\t$ pip install onnx\n\n')


def convert_parameter(parameter, context):
    if isinstance(parameter, chainer.Parameter):
        array = parameter.array
    elif isinstance(parameter, chainer.Variable):
        array = parameter.array
    elif isinstance(parameter, chainer.get_array_types()):
        array = parameter
    else:
        raise ValueError(
            'The type of parameter is unknown. It should be either Parameter '
            'or Variable or ndarray, but the type was {}.'.format(
                type(parameter)))
    array = chainer.cuda.to_cpu(array)
    return numpy_helper.from_array(array, context.get_name(parameter))


def rename_tensors(model, force_rename_network_out=True):
    names = {v.name: v.name for v in model.graph.initializer}
    network_output_names = set() if force_rename_network_out else\
        {o.name for o in model.graph.output}
    op_counts = collections.defaultdict(int)

    for op in model.graph.node:
        for i, input_name in enumerate(op.input):
            if input_name not in names:
                names[input_name] = input_name
            op.input[i] = names[input_name]

        op_name = '{}_{}'.format(op.op_type, op_counts[op.op_type])
        op_counts[op.op_type] += 1
        if len(op.output) == 1:
            if op.output[0] not in network_output_names:
                names[op.output[0]] = op_name
                op.output[0] = names[op.output[0]]
        else:
            for i, output_name in enumerate(op.output):
                if output_name in network_output_names:
                    continue
                names[output_name] = '{}_{}'.format(op_name, i)
                op.output[i] = names[output_name]

    for v in tuple(model.graph.input) + tuple(model.graph.output):
        if v.name in names:
            v.name = names[v.name]


def rename_variable_name(
        context, variables, named_vars, new_names, prefix='Input'):
    # Update ``named_vars`` keys to ``new_names``
    if isinstance(variables, (list, tuple)):
        if not new_names:
            new_names = ['{}_{}'.format(prefix, i)
                         for i in range(len(named_vars))]
        if not isinstance(new_names, (list, tuple)) or\
                len(variables) != len(new_names):
            raise ValueError(
                'Replacing name list is not match with input (or output) '
                'variables')
        for i, var in enumerate(variables):
            del named_vars[context.get_name(var)]
            new_name = new_names[i]
            named_vars[new_name] = var
            context.set_name(var, new_name)
    elif isinstance(variables, dict):
        if not new_names:
            new_names = {k: '{}_{}'.format(prefix, i)
                         for i, k in enumerate(variables.keys())}
        if not isinstance(new_names, (list, tuple, dict)) or\
                len(variables) != len(new_names):
            raise ValueError(
                'Replacing name dict is not match with input (or output) '
                'variables')
        if isinstance(new_names, (list, tuple)):
            new_names = {k: v for k, v in zip(variables.keys(), new_names)}
        for k, v in variables.items():
            if k not in new_names:
                raise ValueError(
                    'Key of replacing name is not found in variables')
            del named_vars[context.get_name(v)]
            new_name = new_names[k]
            named_vars[new_name] = v
            context.set_name(v, new_name)
    elif isinstance(variables, chainer.Variable):
        if not new_names:
            new_names = prefix + '_0'
        if isinstance(new_names, list):
            if len(new_names) != 1:
                raise ValueError('Replacing name must be single')
            new_name = new_names[0]
        else:
            new_name = str(new_names)
        del named_vars[context.get_name(variables)]
        named_vars[new_name] = variables
        context.set_name(variables, new_name)


class ONNXExport(chainer.FunctionHook):

    def __init__(self, context, converters, opset_version=None):
        self.context = context
        self.converters = converters

        self.graph = []
        self.inputs = {}  # Input `Variable` objects keyed by string IDs
        self.additional_parameters = []
        self.specified_opset_version = opset_version

    def create_node(
            self, func_name, func, input_names, output_names, parameters):
        onnx_helper.set_func_name(func_name)
        converter = self.converters.get(func_name, None)
        if converter is None:
            raise ValueError('{} is not supported'.format(func_name))
        params = FunctionConverterParams(
            func, self.specified_opset_version, input_names, output_names,
            self.context, parameters)
        nodes = converter(params)
        nodes = list(reversed(nodes))
        assert len(nodes[0].output) == len(output_names)
        nodes[0].output[:] = output_names
        return nodes

    def backward_postprocess(self, function, in_data, out_grad):
        if isinstance(function, chainer.function.FunctionAdapter):
            function = function.function
        func_name = function.__class__.__name__
        input_names = []
        for i in function.inputs:
            # 'i' is a VariableNode, so check if it has a Variable/Parameter
            var = i.get_variable_or_none()
            if var is None:  # No reference to Variable/Parameter
                input_name = self.context.get_name(i)  # Use VariableNode as is
            else:  # It is a parameter inside a Link or network input
                input_name = self.context.get_name(var)
                self.inputs[input_name] = var
            input_names.append(input_name)

        # This is to get corresponding VariableNode id from the output
        # Variable of the network
        output_names = []
        for o in function.outputs:
            var = o().get_variable_or_none()
            if var is not None:  # If the output is kept
                output_name = self.context.get_name(var)
                if output_name in self.inputs:
                    del self.inputs[output_name]
            else:
                output_name = self.context.get_name(o())
            output_names.append(output_name)

        nodes = self.create_node(
            func_name, function, input_names, output_names,
            self.additional_parameters)
        self.graph.extend(nodes)


def export(model, args, filename=None, export_params=True,
           graph_name='Graph', save_text=False, opset_version=None,
           input_names=None, output_names=None, train=False,
           return_named_inout=False, external_converters=None,
           external_opset_imports=None):
    """Export function for chainer.Chain in ONNX format.

    This function performs a forward computation of the given
    :class:`~chainer.Chain`, ``model``, by passing the given arguments ``args``
    directly. It means, the output :class:`~chainer.Variable` object ``y`` to
    make the computational graph will be created by:

    ``y = model(*args)``

    ``external_converters`` and ``external_opset_import`` are for external
    custom operator. When some ~chainer.FunctionNode are expected to convert to
    own customized operator, set converter function with ~chainer.FunctionNode
    name.

    >>> import onnx
    >>> def custom_converter(param):
    >>>     return onnx.helper.make_node(
    >>>         'CustomizedRelu', param.input_names, param.output_names,
    >>>         domain='chainer'),
    >>>
    >>> external_converters = {'ReLU': custom_converter}
    >>> external_imports = {'chainer': 0}
    >>>
    >>> export(model, args,
    >>>        external_converters=external_converters,
    >>>        external_imports=external_imports)

    Returned model has ``CustomizedRelu`` node.

    Args:
        model (~chainer.Chain): The model object you want to export in ONNX
            format. It should have :meth:`__call__` method because the second
            argument ``args`` is directly given to the model by the ``[]``
            accessor.
        args (list or dict): The arguments which are given to the model
            directly.
        filename (str or file-like object): The filename used for saving the
            resulting ONNX model. If None, nothing is saved to the disk.
        export_params (bool): If True, this function exports all the parameters
            included in the given model at the same time. If False, the
            exported ONNX model doesn't include any parameter values.
        graph_name (str): A string to be used for the ``name`` field of the
            graph in the exported ONNX model.
        save_text (bool): If True, the text format of the output ONNX model is
            also saved with ``.txt`` extention.
        opset_version (int): The operator set version of ONNX. If not specified
            or ``None`` is given, the latest opset version of the onnx module
            is used. If an integer is given, it will be ensured that all the
            operator version in the exported ONNX file is less than this value.
        input_names (str, list or dict): Customize input names of the graph.
            Number of ``input_names`` must be same as number of ``args``.
            When set dict type, keys must be same as ``args``'s keys.
        output_names (str, list or dict): Customize output name of the graph.
            Number of ``output_names`` must be same as actual outputs from
            ``model``. When set dict type, keys must be same as the key of
            ``model`` output.
        train (bool): If True, output computational graph with train mode.
        return_named_inout (bool): If set True, return ONNX model with named
            inputs, and named outputs.
        external_converters (dict): Add-on converter. Convert functions
            keyed by ~chainer.FunctionNode name.
        external_opset_imports (dict): Import external opset. opset version
            number keyed by domain name.

    Returns:
        ~onnx.ModelProto or tuple:
            When ``return_named_inout`` is ``False``, return ModelProto as an
            ONNX model. Otherwise return the tuple of ModelProto, named inputs
            and outputs, both inputs and outputs are list of ~chainer.Variable.

    """

    _check_available()

    with chainer.using_config('train', train),\
            chainer.using_config('in_recomputing', True),\
            chainer.using_config('enable_backprop', True):
        return _export(
            model, args, filename, export_params, graph_name, save_text,
            opset_version, input_names, output_names, return_named_inout,
            external_converters, external_opset_imports)


def _export(model, args, filename, export_params, graph_name, save_text,
            opset_version, input_names, output_names, return_named_inout,
            external_converters, external_opset_imports):
    if opset_version is None:
        opset_version = int(onnx.defs.onnx_opset_version())
    elif opset_version < MINIMUM_OPSET_VERSION:
        warnings.warn(
            'ONNX-Chainer has been tested only with opset_version >= {m}. '
            'This is because ONNXRuntime supports only opset_version >= {m}. '
            'The ONNX file exported with your requested opset_version ({o}) '
            'may cause some problems because the converters used for the '
            'opset_version have not been tested.'.format(
                m=MINIMUM_OPSET_VERSION,
                o=opset_version)
        )

    # Forward computation
    context = Context(model)
    network_inputs = {}
    if isinstance(args, tuple):
        args = list(args)
    if isinstance(args, list):
        for i, arg in enumerate(args):
            if isinstance(arg, chainer.get_array_types()):
                args[i] = chainer.Variable(arg)
            network_inputs[context.get_name(args[i])] = args[i]
        flat_args = args
        outputs = model(*args)
    elif isinstance(args, dict):
        for key, arg in args.items():
            if isinstance(arg, chainer.get_array_types()):
                args[key] = chainer.Variable(arg)
            network_inputs[context.get_name(args[key])] = args[key]
        flat_args = list(args.values())
        outputs = model(**args)
    elif isinstance(args, chainer.get_array_types()):
        args = chainer.Variable(args)
        network_inputs[context.get_name(args)] = args
        flat_args = [args]
        outputs = model(args)
    elif isinstance(args, chainer.Variable):
        network_inputs[context.get_name(args)] = args
        flat_args = [args]
        outputs = model(args)
    else:
        raise ValueError(
            'The \'args\' argument should be a list, tuple, dict, '
            'numpy array, or Chainer Variable. But a {} object was '
            'given.'.format(type(args)))
    rename_variable_name(context, args, network_inputs, input_names)

    initializers = []
    input_tensors = []
    param_names = set()
    for param in model.params():
        name = context.get_name(param)
        param_names.add(name)
        tensor = convert_parameter(param, context)
        initializers.append(tensor)
        input_tensors.append(helper.make_tensor_value_info(
            name, tensor.data_type, tensor.dims))

    for name, var in network_inputs.items():
        input_tensors.append(helper.make_tensor_value_info(
            name, NP_TYPE_TO_TENSOR_TYPE[var.dtype], var.shape))

    if external_converters:
        chainer.utils.experimental('external_converters')
        converters = dict(mapping.converters, **external_converters)
    else:
        converters = mapping.converters

    if isinstance(outputs, (list, tuple)):
        flat_outputs = outputs
    elif isinstance(outputs, dict):
        flat_outputs = list(outputs.values())
    elif isinstance(outputs, chainer.Variable):
        flat_outputs = [outputs]
    else:
        raise RuntimeError(
            'Unexpected output type from the model: {}'.format(type(outputs)))
    network_outputs = {context.get_name(var): var for var in flat_outputs}
    if output_names:
        rename_variable_name(
            context, outputs, network_outputs, output_names)
    # Backward computation to construct graph
    with ONNXExport(context, converters, opset_version) as o:
        chainer.grad(flat_outputs, list(model.params()) + flat_args)

    implicit_input_names = set(o.inputs.keys()) - param_names -\
        set(network_inputs.keys())
    for name in implicit_input_names:
        tensor = convert_parameter(o.inputs[name], context)
        initializers.append(tensor)
        input_tensors.append(helper.make_tensor_value_info(
            name, tensor.data_type, tensor.dims))

    # If additional parameters are created during conversion
    if o.additional_parameters:
        for param in o.additional_parameters:
            tensor = convert_parameter(param, context)
            initializers.append(tensor)
            input_tensors.append(helper.make_tensor_value_info(
                context.get_name(param), tensor.data_type, tensor.dims))

    # The graph must be topologically sorted
    graph = reversed(o.graph)

    # Convert output tensors
    output_tensors = []
    for name, var in network_outputs.items():
        output_tensors.append(helper.make_tensor_value_info(
            name, NP_TYPE_TO_TENSOR_TYPE[var.dtype], var.shape))

    if not export_params:
        initializers = []

    onnx_graph = helper.make_graph(
        graph, graph_name, input_tensors, output_tensors,
        initializer=initializers)

    opset_imports = [helper.make_operatorsetid('', opset_version)]
    if external_opset_imports:
        chainer.utils.experimental('external_opset_imports')
        for domain, version in external_opset_imports.items():
            opset_imports.append(helper.make_operatorsetid(domain, version))
    model = helper.make_model(
        onnx_graph,
        producer_name='Chainer',
        producer_version=chainer.__version__,
        opset_imports=opset_imports
    )

    model.ir_version = onnx.IR_VERSION

    rename_tensors(model, force_rename_network_out=not output_names)
    try:
        checker.check_model(model)
    except onnx.checker.ValidationError as e:
        if external_converters is None:
            raise e
        else:
            warnings.warn(
                'Unregistered operator error is occurred but ignored because '
                'exporting with `external_converters`, please take care about '
                'ONNX format check is insufficient. Error message:\n{}'.format(
                    str(e)))

    if filename is not None and isinstance(filename, str):
        with open(filename, 'wb') as fp:
            fp.write(model.SerializeToString())
        if save_text:
            with open(filename + '.txt', 'w') as fp:
                print(model, file=fp)
    elif hasattr(filename, 'write'):
        filename.write(model.SerializeToString())

    if return_named_inout:
        chainer.utils.experimental('return_named_inout')
        return model, network_inputs, network_outputs
    return model
