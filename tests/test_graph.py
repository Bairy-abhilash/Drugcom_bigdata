"""
Graph Construction Tests
========================

Unit tests for graph construction and heterograph building.
"""

import unittest
import torch
import dgl
import numpy as np
from typing import Dict, List

import sys
sys.path.append('.')

from src.graph.graph_constructor import GraphConstructor
from src.graph.heterograph_builder import HeteroGraphBuilder


class TestGraphConstructor(unittest.TestCase):
    """Test cases for GraphConstructor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.constructor = GraphConstructor()
        
        # Sample data
        self.sample_drugs = {
            'drug_1': {'smiles': 'CCO', 'targets': ['target_1', 'target_2']},
            'drug_2': {'smiles': 'CC(C)O', 'targets': ['target_2', 'target_3']},
            'drug_3': {'smiles': 'c1ccccc1', 'targets': ['target_1']}
        }
        
        self.sample_targets = {
            'target_1': {'diseases': ['disease_1']},
            'target_2': {'diseases': ['disease_1', 'disease_2']},
            'target_3': {'diseases': ['disease_2']}
        }
        
        self.sample_synergy = [
            ('drug_1', 'drug_2', 0.8),
            ('drug_1', 'drug_3', 0.3),
            ('drug_2', 'drug_3', 0.6)
        ]
    
    def test_initialization(self):
        """Test GraphConstructor initialization."""
        self.assertIsNotNone(self.constructor)
        self.assertIsInstance(self.constructor, GraphConstructor)
    
    def test_build_homogeneous_graph(self):
        """Test building a homogeneous graph."""
        num_nodes = 10
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        
        graph = self.constructor.build_homogeneous_graph(
            num_nodes=num_nodes,
            edges=edges
        )
        
        self.assertIsInstance(graph, dgl.DGLGraph)
        self.assertEqual(graph.num_nodes(), num_nodes)
        self.assertEqual(graph.num_edges(), len(edges))
    
    def test_add_node_features(self):
        """Test adding node features to graph."""
        num_nodes = 5
        feature_dim = 128
        
        graph = dgl.graph(([], []), num_nodes=num_nodes)
        features = torch.randn(num_nodes, feature_dim)
        
        graph = self.constructor.add_node_features(graph, features)
        
        self.assertTrue('feat' in graph.ndata)
        self.assertEqual(graph.ndata['feat'].shape, (num_nodes, feature_dim))
    
    def test_add_edge_weights(self):
        """Test adding edge weights."""
        edges = [(0, 1), (1, 2), (2, 3)]
        weights = [0.5, 0.7, 0.9]
        
        graph = dgl.graph(edges)
        graph = self.constructor.add_edge_weights(graph, weights)
        
        self.assertTrue('weight' in graph.edata)
        self.assertEqual(graph.edata['weight'].shape[0], len(edges))
    
    def test_drug_similarity_edges(self):
        """Test creation of drug similarity edges."""
        drug_features = torch.randn(5, 128)
        similarity_threshold = 0.7
        
        edges = self.constructor.create_drug_similarity_edges(
            drug_features,
            threshold=similarity_threshold
        )
        
        self.assertIsInstance(edges, list)
        for src, dst in edges:
            self.assertIsInstance(src, int)
            self.assertIsInstance(dst, int)
            self.assertNotEqual(src, dst)


class TestHeteroGraphBuilder(unittest.TestCase):
    """Test cases for HeteroGraphBuilder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = HeteroGraphBuilder()
        
        # Sample heterogeneous graph data
        self.sample_data = {
            'num_nodes': {
                'drug': 10,
                'target': 15,
                'disease': 5
            },
            'edges': {
                ('drug', 'targets', 'target'): [(0, 1), (0, 2), (1, 3)],
                ('target', 'associated_with', 'disease'): [(1, 0), (2, 1)],
                ('drug', 'similar_to', 'drug'): [(0, 1), (2, 3)]
            }
        }
    
    def test_initialization(self):
        """Test HeteroGraphBuilder initialization."""
        self.assertIsNotNone(self.builder)
        self.assertIsInstance(self.builder, HeteroGraphBuilder)
    
    def test_build_heterograph(self):
        """Test building a heterogeneous graph."""
        graph = self.builder.build(
            self.sample_data['num_nodes'],
            self.sample_data['edges']
        )
        
        self.assertIsInstance(graph, dgl.DGLGraph)
        self.assertTrue(len(graph.ntypes) > 0)
        self.assertTrue(len(graph.etypes) > 0)
    
    def test_node_types(self):
        """Test that all node types are present."""
        graph = self.builder.build(
            self.sample_data['num_nodes'],
            self.sample_data['edges']
        )
        
        expected_ntypes = ['drug', 'target', 'disease']
        for ntype in expected_ntypes:
            self.assertIn(ntype, graph.ntypes)
    
    def test_edge_types(self):
        """Test that all edge types are present."""
        graph = self.builder.build(
            self.sample_data['num_nodes'],
            self.sample_data['edges']
        )
        
        expected_etypes = ['targets', 'associated_with', 'similar_to']
        for etype in expected_etypes:
            self.assertIn(etype, graph.etypes)
    
    def test_node_counts(self):
        """Test that node counts match specification."""
        graph = self.builder.build(
            self.sample_data['num_nodes'],
            self.sample_data['edges']
        )
        
        for ntype, count in self.sample_data['num_nodes'].items():
            self.assertEqual(graph.num_nodes(ntype), count)
    
    def test_add_heterograph_features(self):
        """Test adding features to heterograph nodes."""
        graph = self.builder.build(
            self.sample_data['num_nodes'],
            self.sample_data['edges']
        )
        
        # Add features for each node type
        features = {
            'drug': torch.randn(10, 512),
            'target': torch.randn(15, 256),
            'disease': torch.randn(5, 128)
        }
        
        graph = self.builder.add_node_features(graph, features)
        
        for ntype, feat in features.items():
            self.assertTrue('feat' in graph.nodes[ntype].data)
            self.assertEqual(
                graph.nodes[ntype].data['feat'].shape,
                feat.shape
            )
    
    def test_metapath_extraction(self):
        """Test extracting metapaths from heterograph."""
        graph = self.builder.build(
            self.sample_data['num_nodes'],
            self.sample_data['edges']
        )
        
        metapath = [('drug', 'targets', 'target'), 
                    ('target', 'associated_with', 'disease')]
        
        result = self.builder.extract_metapath(graph, metapath)
        
        self.assertIsNotNone(result)
    
    def test_graph_statistics(self):
        """Test computation of graph statistics."""
        graph = self.builder.build(
            self.sample_data['num_nodes'],
            self.sample_data['edges']
        )
        
        stats = self.builder.get_statistics(graph)
        
        self.assertIn('num_nodes', stats)
        self.assertIn('num_edges', stats)
        self.assertIn('node_types', stats)
        self.assertIn('edge_types', stats)
    
    def test_bidirectional_edges(self):
        """Test adding bidirectional edges."""
        num_nodes = {'A': 5, 'B': 5}
        edges = {
            ('A', 'to', 'B'): [(0, 1), (1, 2)]
        }
        
        graph = self.builder.build(num_nodes, edges, bidirectional=True)
        
        # Should have reverse edges
        self.assertIn('rev_to', graph.etypes)
    
    def test_self_loops(self):
        """Test handling of self-loops."""
        num_nodes = {'A': 5}
        edges = {
            ('A', 'self', 'A'): [(0, 0), (1, 1), (2, 2)]
        }
        
        graph = self.builder.build(num_nodes, edges)
        
        self.assertEqual(graph.num_edges('self'), 3)
    
    def test_empty_graph(self):
        """Test creation of empty heterograph."""
        num_nodes = {'A': 5, 'B': 3}
        edges = {}
        
        graph = self.builder.build(num_nodes, edges)
        
        self.assertEqual(graph.num_nodes('A'), 5)
        self.assertEqual(graph.num_nodes('B'), 3)
        self.assertEqual(graph.num_edges(), 0)


class TestGraphUtilities(unittest.TestCase):
    """Test utility functions for graph operations."""
    
    def test_graph_to_device(self):
        """Test moving graph to device."""
        graph = dgl.graph(([0, 1, 2], [1, 2, 3]))
        graph.ndata['feat'] = torch.randn(4, 10)
        
        device = torch.device('cpu')
        graph = graph.to(device)
        
        self.assertEqual(graph.ndata['feat'].device.type, 'cpu')
    
    def test_subgraph_extraction(self):
        """Test extracting subgraph."""
        graph = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]))
        nodes = [0, 1, 2]
        
        subgraph = dgl.node_subgraph(graph, nodes)
        
        self.assertEqual(subgraph.num_nodes(), len(nodes))
    
    def test_batch_graphs(self):
        """Test batching multiple graphs."""
        graphs = [
            dgl.graph(([0, 1], [1, 2])),
            dgl.graph(([0, 1, 2], [1, 2, 0])),
        ]
        
        batched = dgl.batch(graphs)
        
        self.assertEqual(batched.batch_size, len(graphs))


if __name__ == '__main__':
    unittest.main()