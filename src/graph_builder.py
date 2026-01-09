"""
Graph Builder Module
Constructs similarity graphs using NetworkX for community detection
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set
from collections import defaultdict


class SimilarityGraphBuilder:
    """
    Builds and analyzes similarity graphs for collusion detection
    - Nodes represent students
    - Edges represent similarity above threshold
    - Communities represent potential colluding groups
    """
    
    def __init__(self, similarity_threshold: float = 0.65, knn_k: int = None):
        """
        Initialize graph builder
        
        Args:
            similarity_threshold: Minimum similarity to create an edge
        """
        self.similarity_threshold = similarity_threshold
        self.knn_k = knn_k
        self.graph = None
    
    def build_graph(self, similarity_matrix: pd.DataFrame, jaccard_matrix: pd.DataFrame = None, jaccard_edge_threshold: float = None) -> nx.Graph:
        """
        Build similarity graph from pairwise similarity matrix
        
        Args:
            similarity_matrix: Student similarity matrix
            
        Returns:
            NetworkX graph with students as nodes and similarities as edge weights
        """
        G = nx.Graph()
        
        # Add all students as nodes
        student_ids = similarity_matrix.index.values
        G.add_nodes_from(student_ids)
        
        # Add edges for similarities above threshold
        for i, student1 in enumerate(student_ids):
            for j, student2 in enumerate(student_ids[i+1:], start=i+1):
                similarity = similarity_matrix.iloc[i, j]
                
                if similarity >= self.similarity_threshold:
                    if jaccard_matrix is not None and jaccard_edge_threshold is not None:
                        jac = jaccard_matrix.iloc[i, j]
                        if jac < jaccard_edge_threshold:
                            continue
                    G.add_edge(student1, student2, weight=similarity)

        # Optional k-NN pruning: keep top-k strongest edges per node
        if self.knn_k and self.knn_k > 0 and G.number_of_edges() > 0:
            keep_edges = set()
            for node in G.nodes:
                neighbors = []
                for nbr in G.neighbors(node):
                    weight = G.get_edge_data(node, nbr).get('weight', 0)
                    neighbors.append((nbr, weight))
                neighbors.sort(key=lambda x: x[1], reverse=True)
                for nbr, _w in neighbors[:self.knn_k]:
                    # store as sorted tuple for undirected consistency
                    keep_edges.add(tuple(sorted((node, nbr))))

            # Remove edges not in keep set
            to_remove = []
            for u, v in G.edges():
                key = tuple(sorted((u, v)))
                if key not in keep_edges:
                    to_remove.append((u, v))
            G.remove_edges_from(to_remove)
        
        self.graph = G
        
        print(f"📊 Graph constructed:")
        print(f"   Nodes (students): {G.number_of_nodes()}")
        print(f"   Edges (similar pairs): {G.number_of_edges()}")
        print(f"   Similarity threshold: {self.similarity_threshold}")
        if self.knn_k:
            print(f"   k-NN pruning: top-{self.knn_k} edges per node")
        
        return G
    
    def get_connected_components(self, use_community_detection: bool = True, dynamic_prune: bool = True) -> List[Set[str]]:
        """
        Get connected components (groups of connected students)
        Optionally applies community detection to split large components and
        dynamic edge pruning to remove weak cross-group links.

        Args:
            use_community_detection: If True, use modularity-based communities within components
            dynamic_prune: If True, prune weak edges inside large components before splitting

        Returns:
            List of connected components or communities (sets of student IDs)
        """
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")

        components = [set(c) for c in nx.connected_components(self.graph)]

        # Filter out single-node components
        components = [c for c in components if len(c) > 1]

        refined_components: List[Set[str]] = []

        for comp in components:
            if len(comp) <= 5:
                refined_components.append(comp)
                continue

            subgraph = self.graph.subgraph(comp).copy()

            # Dynamic pruning of weak edges based on local distribution
            if dynamic_prune and subgraph.number_of_edges() > 0:
                edge_weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
                # Raise threshold to the 75th percentile of internal weights
                internal_threshold = max(self.similarity_threshold, float(np.percentile(edge_weights, 75)))
                to_remove = [(u, v) for u, v, data in subgraph.edges(data=True) if data.get('weight', 0) < internal_threshold]
                subgraph.remove_edges_from(to_remove)

            # If pruning disconnected the subgraph into smaller parts, use those
            sub_components = [set(c) for c in nx.connected_components(subgraph)]
            sub_components = [c for c in sub_components if len(c) > 1]

            if use_community_detection and subgraph.number_of_edges() > 0:
                try:
                    # Modularity-based communities (greedy)
                    communities = list(nx.community.greedy_modularity_communities(subgraph, weight='weight'))
                    # Convert to sets and filter
                    community_sets = [set(comm) for comm in communities if len(comm) >= 2]
                    if community_sets:
                        refined_components.extend(community_sets)
                        continue
                except Exception:
                    # Fallback to sub-components
                    pass

            # Fallback: use sub-components if community detection didn't split
            if sub_components:
                refined_components.extend(sub_components)
            else:
                refined_components.append(comp)

        print(f"\n🔗 Connected components:")
        print(f"   Components found: {len(refined_components)}")
        for i, comp in enumerate(refined_components, 1):
            print(f"   Component {i}: {len(comp)} students")

        return refined_components
    
    def get_subgraph_statistics(self, nodes: Set[str]) -> Dict:
        """
        Compute statistics for a subgraph (e.g., a suspected colluding group)
        
        Args:
            nodes: Set of student IDs in the subgraph
            
        Returns:
            Dictionary with subgraph statistics
        """
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")
        
        subgraph = self.graph.subgraph(nodes)
        
        # Get edge weights
        edge_weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
        
        stats = {
            'size': len(nodes),
            'num_edges': subgraph.number_of_edges(),
            'density': nx.density(subgraph),
            'avg_similarity': np.mean(edge_weights) if edge_weights else 0,
            'min_similarity': np.min(edge_weights) if edge_weights else 0,
            'max_similarity': np.max(edge_weights) if edge_weights else 0
        }
        
        return stats
    
    def get_graph_summary(self) -> Dict:
        """
        Get comprehensive graph summary statistics
        
        Returns:
            Dictionary with graph-level statistics
        """
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")
        
        # Get edge weights
        edge_weights = [data['weight'] for _, _, data in self.graph.edges(data=True)]
        
        # Get degree distribution
        degrees = [d for _, d in self.graph.degree()]
        
        summary = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_degree': np.mean(degrees) if degrees else 0,
            'max_degree': np.max(degrees) if degrees else 0,
            'num_connected_components': nx.number_connected_components(self.graph),
            'avg_edge_weight': np.mean(edge_weights) if edge_weights else 0
        }
        
        return summary
