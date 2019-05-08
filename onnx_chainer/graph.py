import collections
from collections import OrderedDict
import heapq

import chainer

from onnx_chainer.functions.converter import FunctionConverterParams
from onnx_chainer import onnx_helper


class Node(object):
    def __init__(self, node, no=None):
        self.no = no
        self.node = node
        self.in_edges = []
        self.out_edges = []

    def __hash__(self):
        return id(self.node).__hash__()

    def __eq__(self, r):
        return self.node is r.node

    def add_in_edge(self, from_node):
        self.in_edges.append(from_node)

    def add_out_edge(self, to_node):
        self.out_edges.append(to_node)


class Graph(object):

    def __init__(
            self, context, converters, opset_version, network_outputs):
        self.context = context
        self.converters = converters

        self.graph = []
        # Converter nodes keyed by "number:func_name"
        self.converted_nodes = OrderedDict()
        self.func_name_counts = collections.defaultdict(int)
        self.inputs = {}  # Input `Variable` objects keyed by string IDs
        self.additional_parameters = []
        self.specified_opset_version = opset_version
        self.network_outputs = network_outputs

        self.chainer_graph = self._build_computational_graph(
            network_outputs.values())

    def _build_computational_graph(self, outputs):
        cands = []
        seen_edges = set()
        nodes = set()
        push_count = [0]

        def add_cand(cand):
            heapq.heappush(cands, (-cand.rank, push_count[0], cand))
            push_count[0] += 1

        converted_nodes = {}
        created_order = [0]

        def create_node(node):
            node_id = id(node)
            if node_id in converted_nodes:
                return converted_nodes[node_id]
            else:
                converted_nodes[node_id] = Node(node, created_order[0])
                created_order[0] += 1
                return converted_nodes[node_id]

        for o in outputs:
            if isinstance(o, chainer.Variable):
                o = o.node
            add_cand(o)
            o_node = create_node(o)
            nodes.add(o_node)

        while cands:
            _, _, cand = heapq.heappop(cands)
            if isinstance(cand, chainer.variable.VariableNode):
                creator = cand.creator_node
                if creator is not None and (creator, cand) not in seen_edges:
                    add_cand(creator)
                    seen_edges.add((creator, cand))
                    creator_node = create_node(creator)
                    cand_node = create_node(cand)
                    cand_node.add_in_edge(creator_node)
                    creator_node.add_out_edge(cand_node)
                    nodes.add(creator_node)
            elif isinstance(cand, chainer.FunctionNode):
                for input_ in cand.inputs:
                    if input_ is not cand and (input_, cand) not in seen_edges:
                        add_cand(input_)
                        seen_edges.add((input_, cand))
                        input_node = create_node(input_)
                        cand_node = create_node(cand)
                        cand_node.add_in_edge(input_node)
                        input_node.add_out_edge(cand_node)
                        nodes.add(input_node)

        # topological sorting
        visited = set()
        sorted_nodes = []

        def visit(node):
            if node not in visited:
                visited.add(node)
                if node.out_edges is not None:
                    for n in node.out_edges:
                        visit(n)
                sorted_nodes.insert(0, node)
        for node in sorted(nodes, key=lambda n: n.no):
            visit(node)
        for no, node in enumerate(sorted_nodes):
            node.no = no

        return sorted(nodes, key=lambda n: n.no)

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

    def convert_to_onnx_node(self, function):
        if isinstance(function, chainer.function.FunctionAdapter):
            function = function.function
        func_name = getattr(
            function, 'custom_function_node_name', function.__class__.__name__)
        temp_node_name = '{}:{}'.format(
            self.func_name_counts[func_name], func_name)
        self.func_name_counts[func_name] += 1

        input_names = []
        for input_var in function.inputs:
            # 'input_var' is a VariableNode,
            # so check if it has a Variable/Parameter
            var = input_var.get_variable_or_none()
            if var is None:  # No reference to Variable/Parameter
                # Use VariableNode as is
                input_name = self.context.get_name(input_var)
            else:  # It is a parameter inside a Link or network input
                input_name = self.context.get_name(var)
                self.inputs[input_name] = var
            input_names.append(input_name)

        # This is to get corresponding VariableNode id from the output
        # Variable of the network
        output_names = []
        for output_ref in function.outputs:
            if output_ref() is None:
                output_name = self.context.get_name(output_ref)
            else:
                var = output_ref().get_variable_or_none()
                if var is not None:  # If the output is kept
                    output_name = self.context.get_name(var)
                    if output_name in self.inputs:
                        del self.inputs[output_name]
                else:
                    output_name = self.context.get_name(output_ref())
            output_names.append(output_name)

        nodes = self.create_node(
            func_name, function, input_names, output_names,
            self.additional_parameters)
        self.converted_nodes[temp_node_name] = nodes

    def after_rename(self):
        """Rename output names.

        When renaming an output name, another node can reference the same value
        as input, so the input name must be renamed at once. So this renaming
        process should be run after all functions are converted

        If input/output names are given externally, these given names take
        priority over named by this process.
        """
        func_name_counts = collections.defaultdict(int)
        names = {}
        for temp_func_name, nodes in self.converted_nodes.items():
            func_name = temp_func_name[temp_func_name.index(':')+1:]
            base_node_name = '{}_{}'.format(
                func_name, func_name_counts[func_name])
            func_name_counts[func_name] += 1
            for num, node in enumerate(reversed(nodes)):
                if len(nodes) > 1 and num != len(nodes)-1:
                    node_name = '{}_tmp_{}'.format(base_node_name, num)
                else:
                    node_name = base_node_name
                node.name = node_name

                for i, input_name in enumerate(node.input):
                    if input_name not in names:
                        names[input_name] = input_name
                    node.input[i] = names[input_name]

                for i, output_name in enumerate(node.output):
                    var = None
                    if output_name in self.network_outputs:
                        var = self.network_outputs[output_name]
                        if self.context.is_pinned(var):
                            continue
                    if len(node.output) == 1:
                        names[output_name] = node_name
                    else:
                        names[output_name] = '{}_{}'.format(node_name, i)
                    node.output[i] = names[output_name]
                    if var is not None:
                        del self.network_outputs[output_name]
                        self.network_outputs[names[output_name]] = var
                self.graph.append(node)

    def to_onnx_graph(self):
        for node in self.chainer_graph:
            if isinstance(node.node, chainer.variable.VariableNode):
                continue
            self.convert_to_onnx_node(node.node)
        self.after_rename()
