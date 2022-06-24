"""
Network Pandas implementation
(c) 2022 d373c7
"""

import pandas as pd
import numpy as np
from numba import jit

from .network import NetworkDefinition, NetworkNodeDefinition, NetworkEdgeDefinition

from ..features.common import Feature
from ..features.tensor import TensorDefinition

from typing import List


class NetworkNodeDefinitionPandas(NetworkNodeDefinition[pd.DataFrame]):
    def __init__(self, name: str, id_feature: Feature, td: TensorDefinition, df: pd.DataFrame):
        super(NetworkNodeDefinitionPandas, self).__init__(name, id_feature)
        self._td = td
        self._df = df

    @property
    def tensor_definition(self) -> TensorDefinition:
        return self._td

    @property
    def node_list(self) -> pd.DataFrame:
        return self._df


class NetworkEdgeDefinitionPandas(NetworkEdgeDefinition[pd.DataFrame]):
    def __init__(self, name: str, id_feature: Feature, from_node: NetworkNodeDefinitionPandas, from_node_id: Feature,
                 to_node: NetworkNodeDefinitionPandas, to_node_id: Feature, td: TensorDefinition, df: pd.DataFrame):
        super(NetworkEdgeDefinitionPandas, self).__init__(
            name, id_feature, from_node, from_node_id, to_node, to_node_id)
        self._td = td
        self._df = df

    @property
    def tensor_definition(self) -> TensorDefinition:
        return self._td

    @property
    def edge_list(self) -> pd.DataFrame:
        return self._df


class NetworkDefinitionPandas(NetworkDefinition):
    def __init__(self, name: str):
        super(NetworkDefinitionPandas, self).__init__(name)
        self._nodes: List[NetworkNodeDefinitionPandas] = []
        self._edges: List[NetworkEdgeDefinitionPandas] = []

    def add_node_definition(self, node: NetworkNodeDefinitionPandas):
        self._nodes.append(node)

    def add_edge_definition(self, edge: NetworkEdgeDefinitionPandas):
        self._edges.append(edge)

    @property
    def node_definition_list(self) -> List[NetworkNodeDefinitionPandas]:
        return self._nodes

    @property
    def edge_definition_list(self) -> List[NetworkEdgeDefinitionPandas]:
        return self._edges

    @property
    def _node_id_list(self) -> List[np.ndarray]:
        nl = [n.node_list[n.id_feature.name] for n in self._nodes]
        return [n.to_numpy() for n in nl]

    @property
    def _edge_id_list(self) -> List[np.ndarray]:
        el = [e.edge_list[[e.from_node_id.name, e.to_node_id.name]] for e in self._edges]
        return [e.to_numpy() for e in el]

    def replace_by_index(self, edge: NetworkEdgeDefinitionPandas) -> np.ndarray:
        e_index = self._edges.index(edge)
        e = self._edges[e_index]
        el = e.edge_list[[e.from_node_id.name, e.to_node_id.name]].to_numpy()
        nn = [n.name for n in self._nodes]
        n_f_i, n_t_i = nn.index(e.from_node.name), nn.index(e.to_node.name)
        n_f = self._nodes[n_f_i].node_list[self._nodes[n_f_i].id_feature.name].unique()
        n_t = self._nodes[n_t_i].node_list[self._nodes[n_t_i].id_feature.name].unique()
        out = _replace_by_index(el, n_f, n_t, n_f_i == n_t_i)
        return out

    def adjacency_matrix(self) -> np.ndarray:
        nl = self._node_id_list
        el = self._edge_id_list
        nn = [n.name for n in self._nodes]
        ni = [(nn.index(e.from_node.name), nn.index(e.to_node.name)) for e in self._edges]
        el = [_replace_by_index(e, nl[f_ind], nl[t_ind], f_ind == t_ind) for e, (f_ind, t_ind) in zip(el, ni)]
        t_len = sum([len(n) for n in nl])
        am = np.zeros((t_len, t_len), dtype=np.uint8)
        for e in el:
            am[e[:, 0], e[:, 1]] = 1
        return am


# Some numba jit-ed functions
@jit(nopython=True, cache=True)
def _replace_by_index(edges: np.ndarray, f_nodes: np.ndarray, t_nodes: np.ndarray, from_equal_to: bool) -> np.ndarray:
    # Allocate output structure
    out = np.zeros(edges.shape, dtype=np.uint32)
    # Make an index to sort to node arrays.
    f_sorter = np.argsort(f_nodes)
    f_nodes_sorted = f_nodes[f_sorter]
    if from_equal_to:
        t_sorter = f_sorter
        t_nodes_sorted = f_nodes_sorted
    else:
        t_sorter = np.argsort(t_nodes)
        t_nodes_sorted = t_nodes[t_sorter]
    # Iterate over the edges, set the out to be the indexes of the nodes.
    for i in range(edges.shape[0]):
        out[i, 0] = f_sorter[np.searchsorted(f_nodes_sorted, edges[i, 0])]
        out[i, 1] = t_sorter[np.searchsorted(t_nodes_sorted, edges[i, 1])]
    return out
