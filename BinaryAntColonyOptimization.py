import copy
import os
import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydot
import tensorflow as tf
from tensorflow.python.util import nest

import swpathnet_func_eyetracker as sw


class BinaryAntColonyOptimization:

    def __init__(self, model, initial_parameter=0.5, evaporate_rate=0.9):

        self.model = model
        self.initial_parameter = initial_parameter
        self.evaporate_rate = evaporate_rate

        # make Graph
        dot = self.model_to_dot(model,
                                show_shapes=False,
                                show_layer_names=True,
                                rankdir='TB',
                                expand_nested=False,
                                dpi=96,
                                subgraph=False)
        network = nx.nx_pydot.from_pydot(dot)
        self.G = nx.DiGraph(network)

        # self.show_graph()

        # Delete not weighted layers in Graph
        pathnet = sw.sw_pathnet(self.model, 2, True,
                                is_reuse_initweight=True)
        dic_weighted = pathnet.gen_li_weighted_with_name(model)
        li_not_weighted = []
        for node in self.G.nodes(data=True):
            if not dic_weighted[node[1]['label']]:
                # print(node[1]['label'])
                li_not_weighted.append(node[0])
        for node in li_not_weighted:
            new_edges = []
            for p in self.G.predecessors(node):
                for s in self.G.successors(node):
                    new_edges.append((p, s))
            # print(new_edges)
            self.G.remove_node(node)
            self.G.add_edges_from(new_edges)

        # add Starts
        li_next_nodes = []
        self.starts = []
        for node in self.G.nodes(data=True):
            if len(list(self.G.predecessors(node[0]))) == 0:
                li_next_nodes.append(node[0])
        for i, n in enumerate(li_next_nodes):
            self.G.add_edge(str(i), n)
            self.G.nodes[str(i)]['label'] = 'start{}'.format(i)
            self.starts.append(str(i))

        # self.show_graph()

        # save join_nodes(join layers)
        self.join_nodes = {}
        for sorted_node in nx.topological_sort(self.G):
            print(self.G.nodes[sorted_node]['label'])
            if self.G.in_degree(sorted_node) > 1:
                # 2を掛けてるのは一つのレイヤーに２つのノードがあるため
                self.join_nodes[self.G.nodes[sorted_node]['label']] = self.G.in_degree(sorted_node)

        # set existing nodes as trainable node
        nx.set_node_attributes(self.G, name='trainable', values=1)

        # make BACO Graph
        name_num = 1000
        nodes = list(self.G.nodes())
        for node in nodes:

            if not (node in self.starts):
                edges = []
                name_num += 1
                name_str = str(name_num)
                for pre in self.G.predecessors(node):
                    edges.append((pre, name_str, initial_parameter))

                for suc in self.G.successors(node):
                    edges.append((name_str, suc, initial_parameter))

                self.G.add_weighted_edges_from(edges)
                self.G.nodes[name_str]['label'] = self.G.nodes[node]['label']
                self.G.nodes[name_str]['trainable'] = 0

        # set weight attribute to all edges
        nx.set_edge_attributes(self.G, name='weight', values=initial_parameter)

        # print(self.G.edges)
        # print('aaa')
        # self.show_no_label_graph()

    # generate path
    def gen_path(self):

        path_with_weights = {list(self.join_nodes.keys())[0]: [0] * 2, list(self.join_nodes.keys())[1]: [0] * 2}
        path = {}
        route_edges = []
        copy_join_nodes = self.join_nodes.copy()
        for start in self.starts:

            next = start
            while len(list(self.G.successors(next))) != 0:
                edges = self.G.out_edges(next)

                # if next node is joined node
                if self.G.nodes[list(edges)[0][1]]['label'] in copy_join_nodes:

                    node_label = self.G.nodes[list(edges)[0][1]]['label']

                    for x in edges:
                        # save weights for join_nodes
                        path_with_weights[node_label][self.G.nodes[x[1]]['trainable']] += copy.deepcopy(
                            self.G.edges[x]['weight'])

                    copy_join_nodes[node_label] -= 1

                    if copy_join_nodes[node_label] == 0:
                        weights = path_with_weights[node_label]

                        # 複数のノードの重みを使用した確率の計算
                        probability = []
                        for weight in weights:
                            probability.append(weight / sum(weights))

                        trainable = np.random.choice([0, 1], p=probability)
                        path[node_label] = trainable

                        for edge in edges:
                            if self.G.nodes[edge[1]]['trainable'] == trainable:
                                route_edges.append((next, edge[1]))
                                next = edge[1]

                    else:
                        break

                else:
                    weights_sum = 0
                    for x in edges:
                        weights_sum += (self.G.edges[x]['weight'])

                    probability = []

                    for x in edges:
                        probability.append(self.G.edges[x]['weight'] / weights_sum)

                    tmp_next = np.random.choice([edge[1] for edge in edges], p=probability)
                    route_edges.append((next, tmp_next))
                    next = tmp_next
                    path[self.G.nodes[next]['label']] = self.G.nodes[next]['trainable']

        # self.show_graph()
        return path, route_edges

    # フェロモンの更新
    def update_pheromone(self, li_edges, li_acc):

        # evaporation
        weights = nx.get_edge_attributes(self.G, 'weight')
        for k, v in weights.items():
            weights[k] = v * self.evaporate_rate

        nx.set_edge_attributes(self.G, name='weight', values=weights)

        print(dict(self.G.edges))

        for edges, acc in zip(li_edges, li_acc):

            pheromone = 1 / (acc * acc)

            for edge in edges:
                self.G.edges[edge]['weight'] += pheromone

    def show_graph(self):
        self.set_seed()
        # show graph
        pos = nx.nx_pydot.pydot_layout(self.G, prog='dot')
        labels = nx.get_node_attributes(self.G, 'label')
        nx.draw(self.G, pos, with_labels=True, labels=labels)
        plt.show()

    def show_no_label_graph(self):
        self.set_seed()
        # show graph
        pos = nx.nx_pydot.graphviz_layout(self.G, prog='dot')
        nx.draw(self.G, pos)
        plt.show()

    def show_weights_graph(self):
        self.set_seed()
        # show graph
        pos = nx.nx_pydot.pydot_layout(self.G, prog='dot')
        labels = nx.get_node_attributes(self.G, 'label')
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw(self.G, pos)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        plt.show()

    def check_pydot(self):
        """Returns True if PyDot and Graphviz are available."""
        if pydot is None:
            return False
        try:
            # Attempt to create an image of a blank graph
            # to check the pydot/graphviz installation.
            pydot.Dot.create(pydot.Dot())
            return True
        except OSError:
            return False

    def is_wrapped_model(self, layer):
        from tensorflow.python.keras.engine import network
        from tensorflow.python.keras.layers import wrappers
        return (isinstance(layer, wrappers.Wrapper) and
                isinstance(layer.layer, network.Network))

    def add_edge(self, dot, src, dst):
        if not dot.get_edge(src, dst):
            dot.add_edge(pydot.Edge(src, dst))

    def model_to_dot(self, model,
                     show_shapes=False,
                     show_layer_names=True,
                     rankdir='TB',
                     expand_nested=False,
                     dpi=96,
                     subgraph=False):
        """Convert a Keras model to dot format.

        Arguments:
          model: A Keras model instance.
          show_shapes: whether to display shape information.
          show_layer_names: whether to display layer names.
          rankdir: `rankdir` argument passed to PyDot,
              a string specifying the format of the plot:
              'TB' creates a vertical plot;
              'LR' creates a horizontal plot.
          expand_nested: whether to expand nested models into clusters.
          dpi: Dots per inch.
          subgraph: whether to return a `pydot.Cluster` instance.

        Returns:
          A `pydot.Dot` instance representing the Keras model or
          a `pydot.Cluster` instance representing nested model if
          `subgraph=True`.

        Raises:
          ImportError: if graphviz or pydot are not available.
        """
        from tensorflow.python.keras.layers import wrappers
        from tensorflow.python.keras.engine import sequential
        from tensorflow.python.keras.engine import network

        if not self.check_pydot():
            if 'IPython.core.magics.namespace' in sys.modules:
                # We don't raise an exception here in order to avoid crashing notebook
                # tests where graphviz is not available.
                print('Failed to import pydot. You must install pydot'
                      ' and graphviz for `pydotprint` to work.')
                return
            else:
                raise ImportError('Failed to import pydot. You must install pydot'
                                  ' and graphviz for `pydotprint` to work.')

        if subgraph:
            dot = pydot.Cluster(style='dashed', graph_name=model.name)
            dot.set('label', model.name)
            dot.set('labeljust', 'l')
        else:
            dot = pydot.Dot()
            dot.set('rankdir', rankdir)
            dot.set('concentrate', True)
            dot.set('dpi', dpi)
            dot.set_node_defaults(shape='record')

        sub_n_first_node = {}
        sub_n_last_node = {}
        sub_w_first_node = {}
        sub_w_last_node = {}

        if not model._is_graph_network:
            node = pydot.Node(str(id(model)), label=model.name)
            dot.add_node(node)
            return dot
        elif isinstance(model, sequential.Sequential):
            if not model.built:
                model.build()
        layers = model._layers

        # Create graph nodes.
        for i, layer in enumerate(layers):
            layer_id = str(id(layer))

            # Append a wrapped layer's label to node's label, if it exists.
            layer_name = layer.name
            class_name = layer.__class__.__name__

            if isinstance(layer, wrappers.Wrapper):
                if expand_nested and isinstance(layer.layer, network.Network):
                    submodel_wrapper = self.model_to_dot(layer.layer, show_shapes,
                                                         show_layer_names, rankdir,
                                                         expand_nested,
                                                         subgraph=True)
                    # sub_w : submodel_wrapper
                    sub_w_nodes = submodel_wrapper.get_nodes()
                    sub_w_first_node[layer.layer.name] = sub_w_nodes[0]
                    sub_w_last_node[layer.layer.name] = sub_w_nodes[-1]
                    dot.add_subgraph(submodel_wrapper)
                else:
                    layer_name = '{}({})'.format(layer_name, layer.layer.name)
                    child_class_name = layer.layer.__class__.__name__
                    class_name = '{}({})'.format(class_name, child_class_name)

            if expand_nested and isinstance(layer, network.Network):
                submodel_not_wrapper = self.model_to_dot(layer, show_shapes,
                                                         show_layer_names, rankdir,
                                                         expand_nested,
                                                         subgraph=True)
                # sub_n : submodel_not_wrapper
                sub_n_nodes = submodel_not_wrapper.get_nodes()
                sub_n_first_node[layer.name] = sub_n_nodes[0]
                sub_n_last_node[layer.name] = sub_n_nodes[-1]
                dot.add_subgraph(submodel_not_wrapper)

            # Create node's label.
            if show_layer_names:
                label = '{}'.format(layer_name)
            else:
                label = class_name

            # Rebuild the label as a table including input/output shapes.
            if show_shapes:

                def format_shape(shape):
                    return str(shape).replace(str(None), '?')

                try:
                    outputlabels = format_shape(layer.output_shape)
                except AttributeError:
                    outputlabels = '?'
                if hasattr(layer, 'input_shape'):
                    inputlabels = format_shape(layer.input_shape)
                elif hasattr(layer, 'input_shapes'):
                    inputlabels = ', '.join(
                        [format_shape(ishape) for ishape in layer.input_shapes])
                else:
                    inputlabels = '?'
                label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label,
                                                               inputlabels,
                                                               outputlabels)

            if not expand_nested or not isinstance(layer, network.Network):
                node = pydot.Node(layer_id, label=label)
                dot.add_node(node)

        # Connect nodes with edges.
        for layer in layers:
            layer_id = str(id(layer))
            for i, node in enumerate(layer._inbound_nodes):
                node_key = layer.name + '_ib-' + str(i)
                if node_key in model._network_nodes:
                    for inbound_layer in nest.flatten(node.inbound_layers):
                        inbound_layer_id = str(id(inbound_layer))
                        if not expand_nested:
                            assert dot.get_node(inbound_layer_id)
                            assert dot.get_node(layer_id)
                            self.add_edge(dot, inbound_layer_id, layer_id)
                        else:
                            # if inbound_layer is not Model or wrapped Model
                            if (not isinstance(inbound_layer, network.Network) and
                                    not self.is_wrapped_model(inbound_layer)):
                                # if current layer is not Model or wrapped Model
                                if (not isinstance(layer, network.Network) and
                                        not self.is_wrapped_model(layer)):
                                    assert dot.get_node(inbound_layer_id)
                                    assert dot.get_node(layer_id)
                                    self.add_edge(dot, inbound_layer_id, layer_id)
                                # if current layer is Model
                                elif isinstance(layer, network.Network):
                                    self.add_edge(dot, inbound_layer_id,
                                                  sub_n_first_node[layer.name].get_name())
                                # if current layer is wrapped Model
                                elif self.is_wrapped_model(layer):
                                    self.add_edge(dot, inbound_layer_id, layer_id)
                                    name = sub_w_first_node[layer.layer.name].get_name()
                                    self.add_edge(dot, layer_id, name)
                            # if inbound_layer is Model
                            elif isinstance(inbound_layer, network.Network):
                                name = sub_n_last_node[inbound_layer.name].get_name()
                                if isinstance(layer, network.Network):
                                    output_name = sub_n_first_node[layer.name].get_name()
                                    self.add_edge(dot, name, output_name)
                                else:
                                    self.add_edge(dot, name, layer_id)
                            # if inbound_layer is wrapped Model
                            elif self.is_wrapped_model(inbound_layer):
                                inbound_layer_name = inbound_layer.layer.name
                                self.add_edge(dot,
                                              sub_w_last_node[inbound_layer_name].get_name(),
                                              layer_id)

        return dot

    def set_seed(self, seed=200):
        tf.random.set_seed(seed)

        # optional
        # for numpy.random
        np.random.seed(seed)
        # for built-in random
        random.seed(seed)
        # for hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
