"""
Definition of Network Features
(c) 2022 d373c7
"""
import logging
from typing import List
from dataclasses import dataclass
from .common import enforce_types, LearningCategory
from .common import Feature

logger = logging.getLogger(__name__)


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureNetworkNodeProperty(Feature):
    """
    Representation of a network node

    Args:
        node_name (str): Name of the node
        node_id (Feature): Identifier of feature of the node. Must be unique
        node_properties (List[Feature]): A list of features to use a properties.
    """

    @property
    def learning_category(self) -> LearningCategory:
        pass

    @property
    def inference_ready(self) -> bool:
        pass


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureNetworkEdgeProperty(Feature):
    """
    Representation of a network edge property
    """

    @property
    def learning_category(self) -> LearningCategory:
        pass

    @property
    def inference_ready(self) -> bool:
        pass


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureNetworkNode:
    """
    Representation of a network node

    Args:
        node_name (str): Name of the node
        node_id (Feature): Identifier of feature of the node. Must be unique
        node_properties (List[Feature]): A list of features to use a properties.
    """
    node_name: str
    node_id: Feature
    node_properties: List[Feature]


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureNetworkEdge:
    """
    Representation of a network node

    Args:
        edge_name (str): Name of the edge
        from_node (NetworkNode): Identifier of feature of the node. Must be unique
        to_node (NetworkNode): A list of features to use a properties.
    """
    edge_name: str
    from_node: FeatureNetworkNode
    to_node: FeatureNetworkNode
    properties: List[Feature]


@enforce_types
@dataclass(unsafe_hash=True, order=True)
class FeatureNetwork:
    """
    Representation of a network.

    Args:
        network_name (str) : The name of the network
        network_nodes (List[NetworkNode]): The node definitions in this network
        network_edges (List[NetworkEdge]): The edge definitions in this network
    """
    network_name: str
    network_nodes: List[FeatureNetworkNode]
    network_edges: List[FeatureNetworkEdge]
