"""
Network Tests
(c) 2022 d373c7
"""
import unittest

import numpy as np

import d373c7.network as nw
import d373c7.features as ft
import d373c7.engines as en

FILE_DIR = './files/'


class TestNetworkNode(unittest.TestCase):
    def test_creation_node(self):
        n_file = FILE_DIR + 'base_nodes.csv'
        name = 'node'
        n_id = ft.FeatureSource('node-id', ft.FEATURE_TYPE_INT_32)
        td = ft.TensorDefinition('Nodes', [n_id])
        with en.EnginePandasNumpy() as e:
            df_n = e.from_csv(td, n_file, inference=False)
        nn = nw.NetworkNodeDefinitionPandas(name, n_id, td, df_n)
        self.assertEqual(nn.name, name, f'Names should have been equal. Got {nn.name}')
        self.assertEqual(nn.id_feature, n_id, f'Did not get id feature, Got {nn.id_feature}')
        self.assertEqual(nn.tensor_definition, td, f'TensorDefinition not correctly set')
        self.assertTrue(nn.node_list.equals(df_n), f'DataFrame not correctly set')


class TestNetworkEdge(unittest.TestCase):
    def test_creation_edge(self):
        n_file = FILE_DIR + 'base_nodes.csv'
        e_file = FILE_DIR + 'base_edges.csv'
        node_name = 'node'
        n_id = ft.FeatureSource('node-id', ft.FEATURE_TYPE_INT_32)
        td_n = ft.TensorDefinition('Nodes', [n_id])
        edge_name = 'edge'
        e_id = ft.FeatureSource('edge-id', ft.FEATURE_TYPE_INT_32)
        e_from_node_id = ft.FeatureSource('from-node-id', ft.FEATURE_TYPE_INT_32)
        e_to_node_id = ft.FeatureSource('to-node-id', ft.FEATURE_TYPE_INT_32)
        td_e = ft.TensorDefinition('Edges', [e_id, e_from_node_id, e_to_node_id])
        with en.EnginePandasNumpy() as e:
            df_n = e.from_csv(td_n, n_file, inference=False)
            df_e = e.from_csv(td_e, e_file, inference=False)
        n = nw.NetworkNodeDefinitionPandas(node_name, n_id, td_n, df_n)
        e = nw.NetworkEdgeDefinitionPandas(edge_name, e_id, n, e_from_node_id, n, e_to_node_id, td_e, df_e)
        self.assertEqual(e.name, edge_name, f'Names should have been equal. Got {e.name}')
        self.assertEqual(e.id_feature, e_id, f'Did not get id feature, Got {e.id_feature}')
        self.assertEqual(e.from_node, n, f'Did not get correct from node {n}')
        self.assertEqual(e.from_node_id, e_from_node_id, f'From_node_id not correctly set {e_from_node_id}')
        self.assertEqual(e.to_node, n, f'Did not get correct from node {n}')
        self.assertEqual(e.to_node_id, e_to_node_id, f'From_node_id not correctly set {e_to_node_id}')
        self.assertEqual(e.tensor_definition, td_e, f'TensorDefinition not correctly set')
        self.assertTrue(e.edge_list.equals(df_e), f'DataFrame not correctly set')


class TestNetwork(unittest.TestCase):
    def test_create_network(self):
        n_file = FILE_DIR + 'base_nodes.csv'
        e_file = FILE_DIR + 'base_edges.csv'
        node_name = 'node'
        n_id = ft.FeatureSource('node-id', ft.FEATURE_TYPE_INT_32)
        td_n = ft.TensorDefinition('Nodes', [n_id])
        edge_name = 'edge'
        e_id = ft.FeatureSource('edge-id', ft.FEATURE_TYPE_INT_32)
        e_from_node_id = ft.FeatureSource('from-node-id', ft.FEATURE_TYPE_INT_32)
        e_to_node_id = ft.FeatureSource('to-node-id', ft.FEATURE_TYPE_INT_32)
        td_e = ft.TensorDefinition('Edges', [e_id, e_from_node_id, e_to_node_id])
        with en.EnginePandasNumpy() as e:
            df_n = e.from_csv(td_n, n_file, inference=False)
            df_e = e.from_csv(td_e, e_file, inference=False)
        nn = nw.NetworkNodeDefinitionPandas(node_name, n_id, td_n, df_n)
        e = nw.NetworkEdgeDefinitionPandas(edge_name, e_id, nn, e_from_node_id, nn, e_to_node_id, td_e, df_e)
        n_name = 'network'
        n = nw.NetworkDefinitionPandas(n_name)
        self.assertEqual(n.name, n_name, f'Name not set {n.name}')
        n.add_node_definition(nn)
        n.add_edge_definition(e)
        self.assertListEqual(n.node_definition_list, [nn], f'Nodelist not correct {n.node_definition_list}')
        self.assertListEqual(n.edge_definition_list, [e], f'EdgeList not correct {n.edge_definition_list}')
        # Make this a different test
        am = n.adjacency_matrix()
        self.assertEqual(type(am), np.ndarray, f'Expecting a numpy array. Got {type(am)}')
        self.assertEqual(len(am.shape), 2, f'Should have been a 2 dimensional array. Got {len(am.shape)}')
        self.assertEqual(am.shape[0], am.shape[1], f'Was expecting a square matrix. {am.shape[0]} {am.shape[1]}')
        self.assertTrue(np.all(np.equal(np.unique(am), np.array([0, 1]))), f'Contains more than 0 & 1 {np.unique(am)}')
        self.assertEqual(np.count_nonzero(am), len(df_e), f'Was expecting {len(df_e)} ones, got {np.count_nonzero(am)}')
        # Add some more tests

    def test_replace_by_index(self):
        n_file = FILE_DIR + 'base_nodes.csv'
        e_file = FILE_DIR + 'base_edges.csv'
        n_id = ft.FeatureSource('node-id', ft.FEATURE_TYPE_INT_32)
        td_n = ft.TensorDefinition('Nodes', [n_id])
        e_id = ft.FeatureSource('edge-id', ft.FEATURE_TYPE_INT_32)
        e_from_node_id = ft.FeatureSource('from-node-id', ft.FEATURE_TYPE_INT_32)
        e_to_node_id = ft.FeatureSource('to-node-id', ft.FEATURE_TYPE_INT_32)
        td_e = ft.TensorDefinition('Edges', [e_id, e_from_node_id, e_to_node_id])
        with en.EnginePandasNumpy() as e:
            df_n = e.from_csv(td_n, n_file, inference=False)
            df_e = e.from_csv(td_e, e_file, inference=False)
        nn = nw.NetworkNodeDefinitionPandas('nodes', n_id, td_n, df_n)
        e = nw.NetworkEdgeDefinitionPandas('edges', e_id, nn, e_from_node_id, nn, e_to_node_id, td_e, df_e)
        n = nw.NetworkDefinitionPandas('Network')
        n.add_node_definition(nn)
        n.add_edge_definition(e)
        idx = n.replace_by_index(e)
        self.assertEqual(type(idx), np.ndarray, f'Expecting a numpy array. Got {type(idx)}')


def main():
    unittest.main()


if __name__ == '__main__':
    main()
