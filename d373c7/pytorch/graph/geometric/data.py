"""
Some Helpers for Graph Geometric library
(c) 2022 d373c7
"""
import logging
import torch
import d373c7.network as nw

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from typing import List

logger = logging.getLogger(__name__)


class GeometricHeteroData:
    def __init__(self, network: nw.NetworkDefinitionPandas, labels: List[torch.Tensor],
                 validation_size: int, test_size: int):
        self._network = network
        self._labels = labels
        self._val_size = validation_size
        self._test_size = test_size
        self._hetero_data_undirected = self._create_hetero_data(network, labels)
        T.ToUndirected()(self._hetero_data_undirected)

    @property
    def network(self) -> nw.NetworkDefinitionPandas:
        return self._network

    @property
    def labels(self) -> List[torch.Tensor]:
        return self._labels

    @property
    def test_size(self) -> int:
        return self._test_size

    @property
    def validation_size(self) -> int:
        return self._val_size

    @staticmethod
    def _create_hetero_data(network: nw.NetworkDefinitionPandas, labels: List[torch.Tensor]) -> HeteroData:
        g_dict = {}
        # Add Nodes
        for n in network.node_definition_list:
            pn = [f for f in n.tensor_definition.feature_names]
            pn.remove(n.id_feature.name)
            g_dict[n.name] = {'x': torch.as_tensor(n.node_list[pn].to_numpy(), dtype=torch.float32)}
        # Add Edges
        for e in network.edge_definition_list:
            pe = [f for f in e.tensor_definition.feature_names]
            try:
                pe.remove(e.id_feature.name)
            except ValueError:
                # If we have duplicate names for instance the id and to node, the remove will throw a ValueError
                pass
            try:
                pe.remove(e.from_node_id.name)
            except ValueError:
                # If we have duplicate names for instance the id and to node, the remove will throw a ValueError
                pass
            try:
                pe.remove(e.to_node_id.name)
            except ValueError:
                # If we have duplicate names for instance the id and to node, the remove will throw a ValueError
                pass
            # Set indexes
            if len(pe) == 0:
                g_dict[(e.from_node.name, e.name, e.to_node.name)] = {
                    'edge_index':
                        torch.as_tensor(e.edge_list[[e.from_node_id.name, e.to_node_id.name]].to_numpy().transpose())
                }
            else:
                g_dict[(e.from_node.name, e.name, e.to_node.name)] = {
                    'edge_index':
                        torch.as_tensor(e.edge_list[[e.from_node_id.name, e.to_node_id.name]].to_numpy().transpose()),
                    'edge_attr':
                        torch.as_tensor(e.edge_list[pe].to_numpy(), dtype=torch.float32)
                }
        g = HeteroData(g_dict)
        return g

    @property
    def hetero_undirected(self) -> HeteroData:
        return self._hetero_data_undirected

    @property
    def validation_mask(self) -> torch.Tensor:
        out = torch.zeros((self.labels[0].shape[0], 1), dtype=torch.bool)
        out[self.labels[0].shape[0]-self.test_size-self.validation_size:self.labels[0].shape[0]-self.test_size] = 1.0
        return out

    @property
    def test_mask(self) -> torch.Tensor:
        out = torch.zeros((self.labels[0].shape[0], 1), dtype=torch.bool)
        out[self.labels[0].shape[0]-self.test_size: self.labels[0].shape[0]] = 1.0
        return out

    @property
    def train_mask(self) -> torch.Tensor:
        out = torch.zeros((self.labels[0].shape[0], 1), dtype=torch.bool)
        out[0:self.labels[0].shape[0]-self.test_size-self.validation_size] = 1.0
        return out

    def random_under_sampled_train_mask(self, sample_ratio: float) -> torch.Tensor:
        train_mask = torch.zeros((self.labels[0].shape[0], 1), dtype=torch.bool)
        train_mask[0:self.labels[0].shape[0]-self.test_size-self.validation_size] = 1.0
        out = torch.zeros((self.labels[0].shape[0], 1), dtype=torch.bool)
        mask_fraud = torch.logical_and(train_mask, torch.eq(self.labels[0], 1.0))
        mask_non_fraud = torch.logical_and(train_mask, torch.eq(self.labels[0], 0.0))
        # Select all fraud record within the train mask
        out[mask_fraud] = 1.0
        # Select the same amount of random non-fraud records
        perm = torch.randperm(torch.count_nonzero(mask_non_fraud).item())
        out[
            torch.argwhere(mask_non_fraud)
            [perm[0:int((torch.count_nonzero(mask_fraud).item()-1) * sample_ratio)]]
        ] = 1.0
        return out
