"""
Some Helpers for Graph Geometric library
(c) 2022 d373c7
"""
import logging
import torch
import d373c7.network as nw

from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


class GeometricData:
    def __int__(self):
        pass

    @classmethod
    def get_hetero_data(cls, network: nw.NetworkDefinitionPandas) -> HeteroData:
        g = HeteroData()
        # Add Nodes
        for n in network.node_definition_list:
            pn = [f for f in n.tensor_definition.feature_names]
            pn.remove(n.id_feature.name)
            g[n.name].x = torch.as_tensor(n.node_list[pn].to_numpy(), dtype=torch.float32)
        # Add Edges
        for e in network.edge_definition_list:
            g[e.from_node.name, e.name, e.to_node.name].edge_index = \
                torch.as_tensor(e.edge_list[[e.from_node_id.name, e.to_node_id.name]].to_numpy().transpose())

        return g

