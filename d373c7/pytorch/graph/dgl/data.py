"""
Some Helpers for Pytorch DGL library
(c) 2022 d373c7
"""
import logging
import dgl
import torch

import d373c7.network as nw

logger = logging.getLogger(__name__)


class DLGData:
    def __int__(self):
        pass

    @classmethod
    def get_hetero_graph(cls, network: nw.NetworkDefinitionPandas) -> dgl.DGLGraph:
        nl = [e.edge_list[[e.from_node_id.name, e.to_node_id.name]].to_numpy() for e in network.edge_definition_list]
        g = dgl.heterograph({
            (e.from_node.name, e.name, e.to_node.name): (torch.as_tensor(n[:, 0]), torch.as_tensor(n[:, 1]))
            for e, n in zip(network.edge_definition_list, nl)
        })
        # Add Node Properties
        for n in network.node_definition_list:
            pn = [f for f in n.tensor_definition.feature_names]
            pn.remove(n.id_feature.name)
            for p in pn:
                g.nodes[n.name].data[p] = torch.as_tensor(n.node_list[p].to_numpy())
        return g
