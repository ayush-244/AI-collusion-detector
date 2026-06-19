"""
Visualization Module
Creates graphs and visual reports for collusion detection results
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

    def plot_interactive_network(self,
                                 graph: nx.Graph,
                                 suspicious_groups: List[Set[str]],
                                 student_features: pd.DataFrame = None) -> "go.Figure":
        """
        Create an interactive Plotly visualization of the similarity network.
        Nodes represent students; red is suspicious, blue is normal.
        """
        import plotly.graph_objects as go
        
        # Flatten suspicious students
        suspicious_students = set()
        for group in suspicious_groups:
            suspicious_students.update(group)
            
        # Get layout (spring layout is standard and clean)
        pos = nx.spring_layout(graph, k=0.6, iterations=50, seed=42)
        
        degrees = dict(graph.degree())
        
        # Index student features by student_id if available
        features_dict = {}
        if student_features is not None and not student_features.empty:
            for _, row in student_features.iterrows():
                features_dict[row['student_id']] = row
                
        edge_x = []
        edge_y = []
        
        # We will create centers for hover details on edges
        edge_centers_x = []
        edge_centers_y = []
        edge_hover_text = []
        
        for u, v in graph.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Midpoint for hover details
            weight = graph[u][v]['weight']
            edge_centers_x.append((x0 + x1) / 2)
            edge_centers_y.append((y0 + y1) / 2)
            edge_hover_text.append(f"Match: {u} & {v}<br>Similarity Score: {weight:.3f}")

        # Edge line trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#CBD5E1'),
            hoverinfo='none',
            mode='lines',
            name='Connections'
        )
        
        # Invisible markers at edge midpoints for hover interaction
        edge_hover_trace = go.Scatter(
            x=edge_centers_x, y=edge_centers_y,
            mode='markers',
            marker=dict(size=4, color='rgba(0,0,0,0)'),
            text=edge_hover_text,
            hoverinfo='text',
            name='Connection Details',
            showlegend=False
        )

        # Node properties
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            is_susp = node in suspicious_students
            status_str = "Suspicious" if is_susp else "Normal"
            color = "#EF4444" if is_susp else "#3B82F6"
            size = degrees[node] * 4 + 18
            
            # Rich hover HTML card
            h_text = f"<b>Student ID:</b> {node}<br><b>Status:</b> {status_str}<br><b>Connections:</b> {degrees[node]}"
            if node in features_dict:
                row = features_dict[node]
                if 'accuracy_rate' in row:
                    acc = row['accuracy_rate']
                    h_text += f"<br><b>Accuracy:</b> {acc*100:.1f}%" if acc <= 1.0 else f"<br><b>Accuracy:</b> {acc:.2f} (std)"
                if 'avg_time_per_question' in row:
                    t_spent = row['avg_time_per_question']
                    h_text += f"<br><b>Avg Time:</b> {t_spent:.1f}s"
                if 'answer_change_rate' in row:
                    chg = row['answer_change_rate']
                    h_text += f"<br><b>Answer Changes:</b> {chg*100:.1f}%" if chg <= 1.0 else f"<br><b>Answer Changes:</b> {chg:.2f} (std)"
            
            node_text.append(h_text)
            node_color.append(color)
            node_size.append(size)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[str(n) for n in graph.nodes()],
            textposition="top center",
            hoverinfo='text',
            hovertext=node_text,
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line=dict(width=1.5, color='#FFFFFF')
            ),
            name='Students'
        )
        
        # Legend proxies
        normal_legend = go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color='#3B82F6'),
            name='Normal Students'
        )
        susp_legend = go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color='#EF4444'),
            name='Suspicious Students'
        )
        
        fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace, normal_legend, susp_legend],
                        layout=go.Layout(
                            title=dict(text='<b>Student Similarity Network Map</b>', font=dict(size=16)),
                            showlegend=True,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='rgba(255, 255, 255, 0.95)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            legend=dict(x=0.85, y=0.95, bgcolor='rgba(255,255,255,0.7)', bordercolor='#E2E8F0', borderwidth=1)
                        ))
                        
        return fig

    def plot_interactive_heatmap(self, similarity_matrix: pd.DataFrame) -> "go.Figure":
        """
        Create an interactive Plotly heatmap of pairwise similarities
        """
        import plotly.graph_objects as go
        
        student_ids = list(similarity_matrix.index)
        matrix_values = similarity_matrix.values
        
        hover_text = []
        for i in range(len(student_ids)):
            row_text = []
            for j in range(len(student_ids)):
                s1 = student_ids[i]
                s2 = student_ids[j]
                val = matrix_values[i, j]
                row_text.append(f"Student A: {s1}<br>Student B: {s2}<br>Similarity: {val:.3f}")
            hover_text.append(row_text)
            
        fig = go.Figure(data=go.Heatmap(
            z=matrix_values,
            x=student_ids,
            y=student_ids,
            text=hover_text,
            hoverinfo='text',
            colorscale='YlOrRd',
            zmin=0,
            zmax=1,
            colorbar=dict(title='Similarity')
        ))
        
        fig.update_layout(
            title=dict(text='<b>Pairwise Student Similarity Heatmap</b>', font=dict(size=16)),
            xaxis=dict(title='Student ID', tickangle=45),
            yaxis=dict(title='Student ID'),
            margin=dict(b=40, l=40, r=20, t=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    def plot_interactive_scatter(self, df_pairs: pd.DataFrame) -> "go.Figure":
        """
        Create an interactive Plotly scatter plot for behavior vs wrong answer similarity
        """
        import plotly.express as px
        import plotly.graph_objects as go
        
        color_map = {'Suspicious': '#EF4444', 'Normal': '#94A3B8'}
        
        fig = px.scatter(
            df_pairs,
            x='Behavior Similarity',
            y='Shared Wrong Answers',
            color='Group',
            color_discrete_map=color_map,
            hover_data={
                'Behavior Similarity': ':.3f',
                'Shared Wrong Answers': ':.3f',
                'Overall Match Score': ':.3f',
                'Group': True
            },
            title="<b>Cheating Analysis (Wrong Answers vs Behavioral Speed)</b>"
        )
        
        # High-suspicion zone boundary rectangle
        fig.add_shape(
            type="rect",
            x0=0.6, y0=0.4, x1=1.0, y1=1.0,
            line=dict(color="#EF4444", width=2, dash="dash"),
            fillcolor="rgba(239, 68, 68, 0.05)",
            layer="below"
        )
        
        fig.add_annotation(
            x=0.8, y=0.9,
            text="High Suspicion Zone",
            showarrow=False,
            font=dict(color="#EF4444", size=12, family="Arial, sans-serif"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#EF4444",
            borderwidth=1,
            borderpad=4
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.75, line=dict(width=0.5, color='#FFFFFF')))
        
        fig.update_layout(
            xaxis_title="Behavioral Metric Similarity (Speed, Timing)",
            yaxis_title="Shared Wrong Answers (Jaccard Index)",
            plot_bgcolor='rgba(248, 250, 252, 0.95)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(b=40, l=40, r=20, t=50),
            legend=dict(title="Student Pairs", bgcolor='rgba(255,255,255,0.7)', bordercolor='#E2E8F0', borderwidth=1)
        )
        
        return fig


if __name__ == "__main__":
    print("Visualization Module - Use via main.py or app.py")
