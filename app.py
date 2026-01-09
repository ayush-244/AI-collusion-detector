"""
Streamlit Dashboard for AI-Based Peer Behavior Similarity Detector
Interactive web interface for collusion detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import yaml
from io import StringIO
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import ExamDataGenerator
from preprocessing import ExamDataPreprocessor
from feature_engineering import FeatureEngineer
from similarity_metrics import SimilarityCalculator
from graph_builder import SimilarityGraphBuilder
from collusion_detector import CollusionDetector
from visualization import CollusionVisualizer


# Page configuration
st.set_page_config(
    page_title="AI Collusion Detector",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .suspicious-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def load_config():
    """Load configuration"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def main():
    # Header
    st.markdown('<div class="main-header">🎓 AI-Based Peer Behavior Similarity Detector</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Detect collusion in online exams** by analyzing behavioral similarities from exam interaction logs.
    Upload your exam data or generate synthetic data to see the system in action.
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Generate Synthetic Data", "Upload CSV File"]
    )
    
    config = load_config()
    
    # Configuration parameters
    st.sidebar.subheader("Detection Parameters")
    
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.5,
        max_value=0.95,
        value=config['similarity']['min_similarity_threshold'],
        step=0.05,
        help="Minimum similarity to create graph edges"
    )
    
    suspicious_threshold = st.sidebar.slider(
        "Suspicious Threshold",
        min_value=0.6,
        max_value=0.95,
        value=config['detection']['suspicious_similarity_threshold'],
        step=0.05,
        help="Minimum similarity to flag as suspicious"
    )
    
    min_group_size = st.sidebar.number_input(
        "Minimum Group Size",
        min_value=2,
        max_value=10,
        value=config['detection']['min_group_size'],
        help="Minimum students in a group to investigate"
    )
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'ground_truth' not in st.session_state:
        st.session_state.ground_truth = None
    
    # Main content
    if data_source == "Generate Synthetic Data":
        st.subheader("📊 Synthetic Data Generation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_students = st.number_input("Normal Students", 50, 1000, 100)
        with col2:
            num_questions = st.number_input("Questions", 20, 200, 50)
        with col3:
            num_groups = st.number_input("Colluding Groups", 1, 20, 3)
        
        if st.button("🎲 Generate Data", type="primary"):
            with st.spinner("Generating synthetic exam data..."):
                # Generate a random seed for this run
                import random
                current_seed = random.randint(1, 100000)
                
                generator = ExamDataGenerator(
                    num_normal_students=num_students,
                    num_questions=num_questions,
                    exam_duration_minutes=120,
                    seed=current_seed
                )
                
                # Create random colluding groups
                import random
                
                # Available student IDs (1 to num_students)
                all_ids = list(range(1, num_students + 1))
                random.shuffle(all_ids)
                
                colluding_groups = []
                current_idx = 0
                
                # Define group sizes (e.g., 4, 5, 3)
                group_sizes = [4, 5, 3]
                
                for i in range(num_groups):
                    size = group_sizes[i % len(group_sizes)]
                    if current_idx + size <= len(all_ids):
                        group_ids = sorted(all_ids[current_idx : current_idx + size])
                        colluding_groups.append(group_ids)
                        current_idx += size
                
                df, ground_truth = generator.generate_complete_dataset(colluding_groups)
                
                # Save to session_state
                st.session_state.df = df
                st.session_state.ground_truth = ground_truth
                
                st.success(f"✅ Generated {len(df)} records for {df['student_id'].nunique()} students")
        
        # Display ground truth if available in session state
        if st.session_state.ground_truth is not None:
             st.info("🎯 **Ground Truth Colluding Groups:**\n\n" + 
                       "\n".join([f"- Group {i+1}: {', '.join(g)}" for i, g in enumerate(st.session_state.ground_truth)]))
    
    else:
        st.subheader("📁 Upload Exam Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV must contain: student_id, question_id, time_spent, selected_option, is_correct, answer_changed, timestamp"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.ground_truth = None
            st.success(f"✅ Loaded {len(df)} records")
    
    # Process data if available in session state
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("---")
        
        # Show data preview
        with st.expander("👀 View Data Preview"):
            st.dataframe(df.head(20), use_container_width=True)
        
        if st.button("🚀 Run Collusion Detection", type="primary"):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Preprocessing
            status_text.text("🔧 Preprocessing data...")
            progress_bar.progress(15)
            
            preprocessor = ExamDataPreprocessor()
            df_clean = preprocessor.preprocess(df, verbose=False)
            
            # Step 2: Feature Engineering
            status_text.text("🔨 Extracting features...")
            progress_bar.progress(30)
            
            engineer = FeatureEngineer()
            feature_matrix = engineer.create_feature_matrix(df_clean, normalize=True)
            wrong_answer_matrix = engineer.extract_wrong_answer_vectors(df_clean)
            
            # Step 3: Similarity Computation
            status_text.text("📐 Computing similarities...")
            progress_bar.progress(50)
            
            calculator = SimilarityCalculator(
                cosine_weight=config['similarity']['cosine_weight'],
                jaccard_weight=config['similarity']['jaccard_weight']
            )
            
            similarities = calculator.compute_all_similarities(
                feature_matrix, 
                wrong_answer_matrix,
                verbose=False
            )
            
            # Step 4: Graph Construction
            status_text.text("🔗 Building similarity graph...")
            progress_bar.progress(65)
            
            graph_builder = SimilarityGraphBuilder(similarity_threshold=similarity_threshold)
            graph = graph_builder.build_graph(similarities['composite'])
            components = graph_builder.get_connected_components()
            
            # Step 5: Collusion Detection
            status_text.text("🚨 Detecting collusion...")
            progress_bar.progress(80)
            
            detector = CollusionDetector(
                suspicious_similarity_threshold=suspicious_threshold,
                min_group_size=min_group_size,
                wrong_answer_overlap_threshold=config['detection']['wrong_answer_overlap_threshold']
            )
            
            analysis_results = detector.detect_suspicious_groups(
                components,
                similarities['composite'],
                wrong_answer_matrix
            )
            
            # Step 6: Visualization
            status_text.text("📊 Generating visualizations...")
            progress_bar.progress(95)
            
            visualizer = CollusionVisualizer(output_dir="outputs/graphs")
            
            suspicious_groups = [
                set(result['students']) 
                for result in analysis_results 
                if result['is_suspicious']
            ]
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display Results
            st.markdown("---")
            st.header("📊 Detection Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Students", df_clean['student_id'].nunique())
            with col2:
                st.metric("Graph Edges", graph.number_of_edges())
            with col3:
                st.metric("Groups Found", len(components))
            with col4:
                st.metric("🚨 Suspicious Groups", len(suspicious_groups))
            
            # Suspicious Groups
            st.subheader("🚨 Suspicious Groups")
            
            report_df = detector.generate_report(analysis_results)
            suspicious_df = report_df[report_df['is_suspicious']]
            
            if len(suspicious_df) > 0:
                for _, row in suspicious_df.iterrows():
                    with st.expander(f"⚠️ Group {row['group_id']} - {row['group_size']} students (Similarity: {row['avg_similarity']:.3f})"):
                        st.write(f"**Students:** {row['students']}")
                        st.write(f"**Average Similarity:** {row['avg_similarity']:.3f}")
                        st.write(f"**Max Similarity:** {row['max_similarity']:.3f}")
                        st.write(f"**Wrong Answer Overlap:** {row['wrong_overlap_ratio']:.3f}")
                        st.write(f"**Reasons:** {row['reasons']}")
            else:
                st.info("No suspicious groups detected with current thresholds.")
            
            # All Groups Table
            st.subheader("📋 All Detected Groups")
            st.dataframe(report_df, use_container_width=True)
            
            # Visualizations
            st.subheader("📈 Visualizations")
            
            tab1, tab2, tab3 = st.tabs(["Similarity Graph", "Heatmap", "Statistics"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                import networkx as nx
                
                suspicious_students = set()
                for group in suspicious_groups:
                    suspicious_students.update(group)
                
                # Check for empty graph
                if graph.number_of_nodes() > 0:
                    # Layout
                    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
                    
                    # Compute degrees for sizing
                    degrees = dict(graph.degree())
                    
                    # Split nodes
                    normal_nodes = [n for n in graph.nodes() if n not in suspicious_students]
                    susp_nodes_list = [n for n in graph.nodes() if n in suspicious_students]
                    
                    # Draw Normal
                    nx.draw_networkx_nodes(graph, pos, nodelist=normal_nodes,
                                          node_color='#4444FF', 
                                          node_size=[degrees[n]*100 + 300 for n in normal_nodes],
                                          alpha=0.6, ax=ax, label='Normal')
                    
                    # Draw Suspicious
                    nx.draw_networkx_nodes(graph, pos, nodelist=susp_nodes_list,
                                          node_color='#FF4444', 
                                          node_size=[degrees[n]*100 + 300 for n in susp_nodes_list],
                                          alpha=0.9, ax=ax, label='Suspicious')
                    
                    # Draw Edges (Thicker = More Similar)
                    weights = [graph[u][v]['weight'] for u, v in graph.edges()]
                    # Scale weights for display
                    widths = [(w - 0.5) * 5 for w in weights] 
                    
                    nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.4, edge_color='gray', ax=ax)
                    nx.draw_networkx_labels(graph, pos, font_size=8, font_weight='bold', ax=ax)
                    
                    ax.legend(loc='upper right')
                
                ax.set_title("Student Similarity Graph", fontsize=14, fontweight='bold')
                ax.axis('off')
                
                st.pyplot(fig)
            
            with tab2:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                import seaborn as sns
                
                if not similarities['composite'].empty:
                    # Show subset for readability
                    max_students = 40
                    if len(similarities['composite']) > max_students:
                        subset = similarities['composite'].iloc[:max_students, :max_students]
                    else:
                        subset = similarities['composite']
                    
                    sns.heatmap(subset, cmap='YlOrRd', vmin=0, vmax=1, 
                               square=True, ax=ax, cbar_kws={'label': 'Similarity'})
                
                ax.set_title("Similarity Heatmap", fontsize=14, fontweight='bold')
                
                st.pyplot(fig)
            
            with tab3:
                import seaborn as sns
                
                st.markdown("### 📊 Understanding the Detection")
                
                st.info("""
                **How to read these charts:**
                *   **Normal Students (Grey)**: Most students have low similarity. They are clustered at the bottom-left.
                *   **Suspicious Students (Red)**: These students are "outliers". They have suspiciously high overlap in both **wrong answers** and **behavior**.
                """)
                
                # 1. Prepare Pairwise Data (Flattening the matrices)
                with st.spinner("Analyzing student pairs..."):
                    cosine_matrix = calculator.compute_cosine_similarity(feature_matrix)
                    jaccard_matrix = calculator.compute_jaccard_similarity(wrong_answer_matrix)
                    
                    # Extract pairs (upper triangle)
                    mask = np.triu(np.ones_like(cosine_matrix, dtype=bool), k=1)
                    rows, cols = np.where(mask)
                    student_ids = cosine_matrix.index
                    
                    # Flatten data with USER-FRIENDLY names
                    pair_data = {
                        'Behavior Similarity': cosine_matrix.values[rows, cols],
                        'Shared Wrong Answers': jaccard_matrix.values[rows, cols],
                        'Overall Match Score': similarities['composite'].values[rows, cols],
                        'Group': ['Normal'] * len(rows)
                    }
                    
                    df_pairs = pd.DataFrame(pair_data)
                    
                    # Label Suspicious Pairs
                    suspicious_pairs_set = set()
                    for group_data in analysis_results:
                        if group_data['is_suspicious']:
                            students = group_data['students']
                            import itertools
                            for s1, s2 in itertools.combinations(students, 2):
                                suspicious_pairs_set.add(frozenset([s1, s2]))
                    
                    # Apply labels
                    pair_labels = []
                    for r, c in zip(rows, cols):
                        s1, s2 = student_ids[r], student_ids[c]
                        if frozenset([s1, s2]) in suspicious_pairs_set:
                            pair_labels.append('Suspicious')
                        else:
                            pair_labels.append('Normal')
                    
                    df_pairs['Group'] = pair_labels
                    df_pairs = df_pairs.sort_values('Group', ascending=True) 
                
                # 2. Plotting
                fig = plt.figure(figsize=(14, 10))
                gs = fig.add_gridspec(2, 2)
                
                sns.set_style("whitegrid")
                
                # A. Scatter Plot
                ax1 = fig.add_subplot(gs[0, :])
                sns.scatterplot(
                    data=df_pairs,
                    x='Behavior Similarity',
                    y='Shared Wrong Answers',
                    hue='Group',
                    palette={'Suspicious': '#FF4444', 'Normal': '#DDDDDD'},
                    style='Group',
                    markers={'Suspicious': 'D', 'Normal': 'o'},
                    size='Group',
                    sizes={'Suspicious': 120, 'Normal': 30},
                    alpha=0.6,
                    ax=ax1
                )
                ax1.set_title("Are they copying? (Wrong Answers vs Behavior)", fontsize=14, fontweight='bold')
                ax1.set_xlabel("Behavior Similarity (Timing, Speed)", fontsize=11)
                ax1.set_ylabel("Shared Wrong Answers (0 = None, 1 = All)", fontsize=11)
                ax1.legend(loc='upper left', title="Student Pairs")
                
                # Add "Cheating Zone" box
                from matplotlib.patches import Rectangle
                rect = Rectangle((0.6, 0.4), 0.4, 0.6, linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
                ax1.add_patch(rect)
                ax1.text(0.98, 0.98, "High Suspicion Zone", ha='right', va='top', transform=ax1.transAxes, color='#CC0000', fontweight='bold')
                
                # B. Simple Comparison
                ax2 = fig.add_subplot(gs[1, :])
                
                # Compare average scores
                avg_stats = df_pairs.groupby('Group')[['Overall Match Score', 'Shared Wrong Answers']].mean().reset_index()
                
                # Melt for easy plotting
                melted = avg_stats.melt(id_vars='Group', var_name='Metric', value_name='Score')
                
                sns.barplot(data=melted, x='Metric', y='Score', hue='Group', 
                           palette={'Suspicious': '#FF4444', 'Normal': '#4444FF'}, ax=ax2)
                
                ax2.set_title("How much more similar are the cheaters?", fontweight='bold')
                ax2.set_ylabel("Average Score (0-1)")
                ax2.set_xlabel("")
                ax2.set_ylim(0, 1.1)
                
                # Add value labels
                for container in ax2.containers:
                    ax2.bar_label(container, fmt='%.2f')

                plt.tight_layout()
                st.pyplot(fig)

            
            # Download Reports
            st.subheader("📥 Download Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Groups Report",
                    data=csv,
                    file_name="suspicious_groups_report.csv",
                    mime="text/csv"
                )
            
            with col2:
                top_pairs = calculator.get_top_similar_pairs(
                    similarities['composite'],
                    threshold=similarity_threshold,
                    top_n=50
                )
                csv_pairs = top_pairs.to_csv(index=False)
                st.download_button(
                    label="📄 Download Similar Pairs",
                    data=csv_pairs,
                    file_name="similar_pairs_report.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()
