from __future__ import print_function

import collections
import warnings

import chainer
import onnx
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from onnx_chainer.context import Context
from onnx_chainer import functions
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


def create_node(
        func_name, opset_version, func, input_names,
        output_names, context, parameters):
    for opver in sorted(mapping.operators[func_name], reverse=True):
        if opver <= opset_version:
            break
    opset_version = opver

    converter_name = 'convert_{}'.format(func_name)
    if hasattr(functions, converter_name):
        onnx_helper.set_func_name(func_name)
        converter = getattr(functions, converter_name)
        nodes = converter(
            func, opset_version, input_names, len(output_names),
            context, parameters)
        nodes = list(reversed(nodes))
        assert len(nodes[0].output) == len(output_names)
        nodes[0].output[:] = output_names
    else:
        raise ValueError('{} is not supported.'.format(func_name))
    return nodes


def rename_tensors(model):
    names = {v.name: v.name for v in model.graph.initializer}
    op_counts = collections.defaultdict(int)

    for op in model.graph.node:
        op_name = '{}_{}'.format(op.op_type, op_counts[op.op_type])
        op_counts[op.op_type] += 1

        for i in range(len(op.input)):
            if op.input[i] not in names:
                names[op.input[i]] = 'Input_{}'.format(op_counts['Input'])
                op_counts['Input'] += 1
            op.input[i] = names[op.input[i]]

        for i in range(len(op.output)):
            if len(op.output) <= 1:
                names[op.output[i]] = op_name
            else:
                names[op.output[i]] = '{}_{}'.format(op_name, i)
            op.output[i] = names[op.output[i]]

    for v in tuple(model.graph.input) + tuple(model.graph.output):
        if v.name in names:
            v.name = names[v.name]


class ONNXExport(chainer.FunctionHook):

    def __init__(self, context, opset_version=None):
        self.context = context

        self.graph = []
        self.inputs = {}  # Input `Variable` objects keyed by string IDs
        # Renamed string IDs keyed by their original string IDs
        self.renamed_outputs = {}
        self.additional_parameters = []
        self.specified_opset_version = opset_version

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
                    # ONNX checker does not accept one value is both input and
                    # output by output SSA checking. To avoid it, add Identity
                    # operator to separate output value.
                    id_node = onnx_helper.make_node(
                        'Identity', [output_name], 1)
                    self.renamed_outputs[output_name] = id_node.output[0]
                    self.graph.append(id_node)
                    del self.inputs[output_name]
            else:
                output_name = self.context.get_name(o())
            output_names.append(output_name)

        opset_versions = mapping.operators[func_name]
        if isinstance(opset_versions, int):
            opset_version = opset_versions
        elif self.specified_opset_version is None:
            # If no opset version is specified,
            # use the latest version for the operator
            opset_version = opset_versions[-1]
        else:
            # If a version is specified, use the last version <= specified one
            for opset_version in sorted(opset_versions, reverse=True):
                if opset_version <= self.specified_opset_version:
                    break

        if opset_version > self.specified_opset_version:
            raise RuntimeError('ONNX-chainer cannot convert `{}` of Chainer '
                               'with ONNX opset_version {}'.format(
                                   func_name, self.specified_opset_version))

        nodes = create_node(
            func_name, opset_version, function, input_names,
            output_names, self.context, self.additional_parameters)
        self.graph.extend(nodes)


def export(model, args, filename=None, export_params=True,
           graph_name='Graph', save_text=False, opset_version=None,
           train=False, return_flat_inout=False):
    """Export function for chainer.Chain in ONNX format.

    This function performs a forward computation of the given
    :class:`~chainer.Chain`, ``model``, by passing the given arguments ``args``
    directly. It means, the output :class:`~chainer.Variable` object ``y`` to
    make the computational graph will be created by:

    y = model(*args)

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
        train (bool): If True, output computational graph with train mode.
        return_flat_inout (bool): If set True, return ONNX model with flat
            inputs, and flat outputs.

    Returns:
        ~onnx.ModelProto or tuple:
            When ``return_flat_input`` is ``False``, return ModelProto as an
            ONNX model. Otherwise return the tuple of ModelProto, flat inputs
            and outputs, both inputs and outputs are list of ~chainer.Variable.

    """

    _check_available()

    with chainer.using_config('train', train),\
            chainer.using_config('in_recomputing', True),\
            chainer.using_config('enable_backprop', True):
        return _export(
            model, args, filename, export_params, graph_name, save_text,
            opset_version, return_flat_inout)


def _export(model, args, filename, export_params, graph_name, save_text,
            opset_version, return_flat_inout):
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
    network_inputs = []
    if isinstance(args, tuple):
        args = list(args)
    if isinstance(args, list):
        for i, arg in enumerate(args):
            if isinstance(arg, chainer.get_array_types()):
                args[i] = chainer.Variable(arg)
            network_inputs.append(args[i])
        flat_args = args
        outputs = model(*args)
    elif isinstance(args, dict):
        for key, arg in args.items():
            if isinstance(arg, chainer.get_array_types()):
                args[key] = chainer.Variable(arg)
            network_inputs.append(args[key])
        flat_args = list(args.values())
        outputs = model(**args)
    elif isinstance(args, chainer.get_array_types()):
        args = chainer.Variable(args)
        network_inputs.append(args)
        flat_args = [args]
        outputs = model(args)
    elif isinstance(args, chainer.Variable):
        network_inputs.append(args)
        flat_args = [args]
        outputs = model(args)
    else:
        raise ValueError(
            'The \'args\' argument should be a list, tuple, dict, '
            'numpy array, or Chainer Variable. But a {} object was '
            'given.'.format(type(args)))

    context = Context(model)

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

    network_input_names = set()
    for i in network_inputs:
        name = context.get_name(i)
        network_input_names.add(name)
        input_tensors.append(helper.make_tensor_value_info(
            name, NP_TYPE_TO_TENSOR_TYPE[i.dtype], i.shape))

    with ONNXExport(context, opset_version) as o:
        if isinstance(outputs, (list, tuple)):
            flat_outputs = outputs
        elif isinstance(outputs, dict):
            flat_outputs = list(outputs.values())
        elif isinstance(outputs, chainer.Variable):
            flat_outputs = [outputs]
        else:
            raise RuntimeError(
                'Unexpected output type from the model: {}'.format(
                    type(outputs)))
        chainer.grad(flat_outputs, list(model.params()) + flat_args)

    implicit_input_names = set(o.inputs.keys()) - param_names -\
        network_input_names
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
    if isinstance(outputs, dict):
        outputs = list(outputs.values())
    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

    for output in outputs:
        output_id = context.get_name(output)
        if output_id in o.renamed_outputs:
            output_id = o.renamed_outputs[output_id]
        output_tensors.append(helper.make_tensor_value_info(
            output_id, NP_TYPE_TO_TENSOR_TYPE[output.dtype], output.shape))

    if not export_params:
        initializers = []

    onnx_graph = helper.make_graph(
        graph, graph_name, input_tensors, output_tensors,
        initializer=initializers)

    model = helper.make_model(
        onnx_graph,
        producer_name='Chainer',
        producer_version=chainer.__version__,
        opset_imports=[helper.make_opsetid('', opset_version)]
    )

    model.ir_version = onnx.IR_VERSION

    rename_tensors(model)
    checker.check_model(model)

    if filename is not None and isinstance(filename, str):
        with open(filename, 'wb') as fp:
            fp.write(model.SerializeToString())
        if save_text:
            with open(filename + '.txt', 'w') as fp:
                print(model, file=fp)
    elif hasattr(filename, 'write'):
        filename.write(model.SerializeToString())

    if return_flat_inout:
        return model, flat_args, flat_outputs
    return model
