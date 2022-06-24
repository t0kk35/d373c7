"""
Network classes
(c) 2022 d373c7
"""
from abc import ABC, abstractmethod

from common import enforce_types

from typing import TypeVar, Generic, List

from ..features.common import Feature

NP = TypeVar('NP')


@enforce_types
class NetworkNodeDefinition(Generic[NP], ABC):
    def __init__(self, name: str, id_feature: Feature):
        self._name = name
        self._id_feature = id_feature

    @property
    def name(self) -> str:
        return self._name

    @property
    def id_feature(self) -> Feature:
        return self._id_feature

    @property
    @abstractmethod
    def node_list(self) -> NP:
        pass


@enforce_types
class NetworkEdgeDefinition(Generic[NP], ABC):
    def __init__(self, name: str, id_feature: Feature, from_node: NetworkNodeDefinition, from_node_id: Feature,
                 to_node: NetworkNodeDefinition, to_node_id: Feature):
        self._name = name
        self._id_feature = id_feature
        self._from_node = from_node
        self._from_node_id = from_node_id
        self._to_node = to_node
        self._to_node_id = to_node_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def id_feature(self) -> Feature:
        return self._id_feature

    @property
    def from_node(self) -> NetworkNodeDefinition:
        return self._from_node

    @property
    def from_node_id(self) -> Feature:
        return self._from_node_id

    @property
    def to_node(self) -> NetworkNodeDefinition:
        return self._to_node

    @property
    def to_node_id(self) -> Feature:
        return self._to_node_id

    @property
    @abstractmethod
    def edge_list(self) -> NP:
        pass


class NetworkDefinition(ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def add_node_definition(self, node: NetworkNodeDefinition):
        pass

    @abstractmethod
    def add_edge_definition(self, edge: NetworkEdgeDefinition):
        pass

    @property
    @abstractmethod
    def node_definition_list(self) -> List[NetworkNodeDefinition]:
        pass

    @property
    @abstractmethod
    def edge_definition_list(self) -> List[NetworkEdgeDefinition]:
        pass
