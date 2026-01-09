"""
Visualization Module
Creates graphs and visual reports for collusion detection results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Set, Dict
import os


class CollusionVisualizer:
    """
    Visualizes similarity graphs and detection results
    """
    
    def __init__(self, output_dir: str = "outputs/graphs"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_similarity_graph(self,
                               graph: nx.Graph,
                               suspicious_groups: List[Set[str]],
                               title: str = "Student Similarity Graph",
                               save_path: str = None):
        """
        Plot the similarity graph with suspicious groups highlighted
        
        Args:
            graph: NetworkX graph
            suspicious_groups: List of suspicious student groups
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(14, 10))
        
        # Flatten suspicious students
        suspicious_students = set()
        for group in suspicious_groups:
            suspicious_students.update(group)
        
        # Node colors
        node_colors = ['#FF4444' if node in suspicious_students else '#4444FF' 
                      for node in graph.nodes()]
        
        # Edge colors based on weight
        edges = graph.edges()
        weights = [graph[u][v]['weight'] for u, v in edges]
        
        # Layout - use larger k for better separation of clusters
        pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
        
        # Calculate node degrees for sizing
        degrees = dict(graph.degree())
        node_sizes = [degrees[node] * 100 + 300 for node in graph.nodes()]
        
        # Draw network
        # 1. Non-suspicious nodes
        normal_nodes = [n for n in graph.nodes() if n not in suspicious_students]
        nx.draw_networkx_nodes(graph, pos, nodelist=normal_nodes, 
                             node_color='#4444FF', node_size=[degrees[n] * 100 + 300 for n in normal_nodes], 
                             alpha=0.6, label='Normal')
                             
        # 2. Suspicious nodes (highlighted)
        suspicious_nodes_list = [n for n in graph.nodes() if n in suspicious_students]
        nx.draw_networkx_nodes(graph, pos, nodelist=suspicious_nodes_list, 
                             node_color='#FF4444', node_size=[degrees[n] * 100 + 300 for n in suspicious_nodes_list], 
                             alpha=0.9, label='Suspicious')
        
        # 3. Edges with varying width based on weight
        weights = [graph[u][v]['weight'] ** 4 * 2 for u, v in edges] # Exponential formatting for clearer visibility
        nx.draw_networkx_edges(graph, pos, width=weights, alpha=0.4, 
                              edge_color='gray')
        
        # 4. Labels
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF4444', label='Suspicious Students'),
            Patch(facecolor='#4444FF', label='Normal Students')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Graph saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'similarity_graph.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_similarity_heatmap(self,
                                 similarity_matrix: pd.DataFrame,
                                 title: str = "Student Similarity Heatmap",
                                 save_path: str = None):
        """
        Plot heatmap of similarity matrix
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            title: Plot title
            save_path: Path to save the plot
        """
        import seaborn as sns
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(similarity_matrix, cmap='YlOrRd', 
                   vmin=0, vmax=1, square=True,
                   cbar_kws={'label': 'Similarity Score'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Student ID', fontsize=10)
        plt.ylabel('Student ID', fontsize=10)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Heatmap saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'similarity_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_group_statistics(self,
                               analysis_results: List[Dict],
                               save_path: str = None):
        """
        Plot comprehensive group statistics with clearer separation
        - Scatter plot: Similarity vs Wrong Answer Overlap
        - Box plots: Distribution comparison
        
        Args:
            analysis_results: List of group analysis results
            save_path: Path to save the plot
        """
        import seaborn as sns
        
        if not analysis_results:
            print("⚠️  No groups to visualize")
            return
        
        # Convert to DataFrame for easier plotting
        df_stats = pd.DataFrame(analysis_results)
        df_stats['Type'] = df_stats['is_suspicious'].apply(lambda x: 'Suspicious' if x else 'Normal')
        
        # Set style
        sns.set_style("whitegrid")
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # 1. Scatter Plot: Similarity vs Wrong Answer Overlap
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot Normal groups
        normal = df_stats[df_stats['Type'] == 'Normal']
        suspicious = df_stats[df_stats['Type'] == 'Suspicious']
        
        ax1.scatter(normal['avg_similarity'], normal['wrong_overlap_ratio'], 
                   c='#4444FF', alpha=0.6, label='Normal Groups', s=100)
        ax1.scatter(suspicious['avg_similarity'], suspicious['wrong_overlap_ratio'], 
                   c='#FF4444', alpha=0.9, label='Suspicious Groups', s=150, marker='D')
        
        # Add thresholds
        ax1.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, label='Sim. Threshold')
        ax1.axhline(y=0.4, color='gray', linestyle=':', alpha=0.5, label='Overlap Threshold')
        
        ax1.set_title('Collusion Detection Boundary: Similarity vs Wrong Answer Overlap', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Average Similarity Score', fontsize=12)
        ax1.set_ylabel('Wrong Answer Overlap Ratio', fontsize=12)
        ax1.legend(loc='upper left', frameon=True)
        
        # 2. Box Plot: Similarity Comparison
        ax2 = fig.add_subplot(gs[1, 0])
        sns.boxplot(x='Type', y='avg_similarity', hue='Type', data=df_stats, ax=ax2, palette={'Suspicious': '#FF4444', 'Normal': '#4444FF'}, legend=False)
        ax2.set_title('Distribution of Similarity Scores', fontweight='bold')
        ax2.set_xlabel('')
        ax2.set_ylabel('Average Similarity')
        
        # 3. Box Plot: Wrong Answer Overlap Comparison
        ax3 = fig.add_subplot(gs[1, 1])
        sns.boxplot(x='Type', y='wrong_overlap_ratio', hue='Type', data=df_stats, ax=ax3, palette={'Suspicious': '#FF4444', 'Normal': '#4444FF'}, legend=False)
        ax3.set_title('Distribution of Wrong Answer Overlap', fontweight='bold')
        ax3.set_xlabel('')
        ax3.set_ylabel('Overlap Ratio')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Statistics plot saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'group_statistics.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()


if __name__ == "__main__":
    print("Visualization Module - Use via main.py or app.py")
