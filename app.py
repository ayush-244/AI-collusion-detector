"""
Streamlit Dashboard for AI-Based Peer Behavior Similarity Detector
Interactive, production-ready web interface for collusion detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Reconfigure stdout/stderr to UTF-8 to prevent UnicodeEncodeError on Windows terminals
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import yaml
import random
from io import StringIO
import matplotlib
matplotlib.use('Agg')
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

# Load configuration
def load_config():
    """Load configuration"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        # Fallback default configuration
        return {
            'similarity': {
                'cosine_weight': 0.4,
                'jaccard_weight': 0.6,
                'min_similarity_threshold': 0.70,
                'knn_k': 3,
                'edge_wrong_overlap_threshold': 0.60,
                'use_community_detection': True,
                'dynamic_prune': True
            },
            'detection': {
                'suspicious_similarity_threshold': 0.78,
                'min_group_size': 3,
                'wrong_answer_overlap_threshold': 0.60
            },
            'output': {
                'data_dir': 'data',
                'reports_dir': 'outputs/reports',
                'graphs_dir': 'outputs/graphs'
            }
        }

# Custom styling
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Styles */
    .stApp {
        background-color: #F8FAFC;
    }
    
    html, body, [class*="css"], .stText, p, li, span, label {
        font-family: 'Inter', sans-serif !important;
        color: #334155;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }

    /* Main Title Styling */
    .main-title-container {
        text-align: center;
        padding: 2.5rem 1.5rem 1.5rem 1.5rem;
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-radius: 20px;
        margin-bottom: 2.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.85rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-subtitle {
        font-size: 1.15rem;
        color: #475569;
        margin-top: 0.75rem;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* Custom Metric Cards */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -2px rgba(0, 0, 0, 0.02);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
        border-color: #CBD5E1;
    }
    .metric-card-suspicious {
        border-left: 5px solid #EF4444;
    }
    .metric-card-normal {
        border-left: 5px solid #3B82F6;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.25rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.25rem;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 6px;
        border-radius: 12px;
        border: none;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre;
        background-color: transparent;
        border-radius: 8px;
        color: #475569;
        font-weight: 600;
        border: none;
        padding: 0px 16px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1E3A8A !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
    }
</style>
""", unsafe_allow_html=True)


def run_detection_pipeline(df: pd.DataFrame, 
                           similarity_threshold: float,
                           suspicious_threshold: float,
                           min_group_size: int,
                           config: dict) -> dict:
    """Run the pipeline and return results in a dictionary"""
    # 1. Preprocessing
    preprocessor = ExamDataPreprocessor()
    df_clean = preprocessor.preprocess(df, verbose=False)
    
    # 2. Feature Engineering
    engineer = FeatureEngineer()
    feature_matrix = engineer.create_feature_matrix(df_clean, normalize=True)
    wrong_answer_matrix = engineer.extract_wrong_answer_vectors(df_clean)
    
    # 3. Similarity Computation
    calculator = SimilarityCalculator(
        cosine_weight=config['similarity']['cosine_weight'],
        jaccard_weight=config['similarity']['jaccard_weight']
    )
    similarities = calculator.compute_all_similarities(
        feature_matrix, 
        wrong_answer_matrix,
        verbose=False
    )
    
    # 4. Graph Construction
    graph_builder = SimilarityGraphBuilder(
        similarity_threshold=similarity_threshold,
        knn_k=config['similarity'].get('knn_k', 3)
    )
    graph = graph_builder.build_graph(
        similarities['composite'],
        jaccard_matrix=similarities['jaccard'],
        jaccard_edge_threshold=config['similarity'].get('edge_wrong_overlap_threshold', 0.60)
    )
    components = graph_builder.get_connected_components(
        use_community_detection=config['similarity'].get('use_community_detection', True),
        dynamic_prune=config['similarity'].get('dynamic_prune', True)
    )
    
    # 5. Collusion Detection
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
    
    # Report DataFrame
    report_df = detector.generate_report(analysis_results)
    
    # Similar pairs for download
    top_pairs = calculator.get_top_similar_pairs(
        similarities['composite'],
        threshold=similarity_threshold
    )
    
    return {
        'df_clean': df_clean,
        'feature_matrix': feature_matrix,
        'wrong_answer_matrix': wrong_answer_matrix,
        'similarities': similarities,
        'graph': graph,
        'components': components,
        'analysis_results': analysis_results,
        'report_df': report_df,
        'top_pairs': top_pairs,
        'calculator': calculator
    }


def render_wrong_answer_grid(group_students, df_clean):
    """Render a beautiful side-by-side comparison of options chosen by group members"""
    # Extract correct answers map from correct submissions
    correct_rows = df_clean[df_clean['is_correct'] == 1]
    correct_map = correct_rows.drop_duplicates('question_id').set_index('question_id')['selected_option'].to_dict()
    
    # If some questions have no correct submissions, fill default Option 1
    all_questions = sorted(df_clean['question_id'].unique())
    for q in all_questions:
        if q not in correct_map:
            # Look up mode or choose default
            q_rows = df_clean[df_clean['question_id'] == q]
            if len(q_rows) > 0:
                correct_map[q] = q_rows['selected_option'].mode()[0] if not q_rows['selected_option'].mode().empty else 1
            else:
                correct_map[q] = 1

    # Filter group data
    group_df = df_clean[df_clean['student_id'].isin(group_students)]
    
    # Pivot to get selected options and correctness status
    options_pivot = group_df.pivot(index='question_id', columns='student_id', values='selected_option')
    is_correct_pivot = group_df.pivot(index='question_id', columns='student_id', values='is_correct')
    
    # Re-index pivots to guarantee all questions exist
    options_pivot = options_pivot.reindex(all_questions).fillna(0).astype(int)
    is_correct_pivot = is_correct_pivot.reindex(all_questions).fillna(0).astype(int)
    
    # HTML Table Construction
    html = """
    <div style='overflow-x: auto; margin-top: 15px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);'>
    <table style='width: 100%; border-collapse: collapse; text-align: left; font-family: "Inter", sans-serif; font-size: 13.5px;'>
    <thead>
        <tr style='background-color: #F1F5F9; border-bottom: 2px solid #E2E8F0; color: #475569;'>
            <th style='padding: 12px 16px; border-right: 1px solid #E2E8F0;'>Question ID</th>
            <th style='padding: 12px 16px; border-right: 1px solid #E2E8F0;'>Correct Ans</th>
    """
    for student in group_students:
        html += f"<th style='padding: 12px 16px; border-right: 1px solid #E2E8F0; text-align: center;'>{student}</th>"
    html += """
            <th style='padding: 12px 16px; text-align: center;'>Indicators / Verdict</th>
        </tr>
    </thead>
    <tbody>
    """
    
    for q_id in all_questions:
        correct_ans = correct_map.get(q_id, "?")
        student_opts = [options_pivot.loc[q_id, s] for s in group_students]
        is_corrects = [is_correct_pivot.loc[q_id, s] for s in group_students]
        
        all_wrong = all(c == 0 for c in is_corrects)
        all_same = len(set(student_opts)) == 1
        any_wrong = any(c == 0 for c in is_corrects)
        
        # Color codes
        row_bg = "white"
        status_text = "Clean (Correct)"
        status_color = "#10B981"
        
        if all_wrong and all_same:
            row_bg = "#FEF2F2"  # Identical incorrect answers (strong collusion indicator)
            status_text = "⚠️ Coordinated Wrong Choice"
            status_color = "#EF4444"
        elif all_wrong:
            row_bg = "#FFFBEB"  # Both wrong, but selected different incorrect choices
            status_text = "Divergent Wrong Choices"
            status_color = "#F59E0B"
        elif any_wrong and all_same:
            # Should not happen because if one is correct and all same, then all must be correct.
            pass
        elif any_wrong:
            row_bg = "#F8FAFC"
            status_text = "Mixed Accuracy"
            status_color = "#64748B"
            
        html += f"<tr style='background-color: {row_bg}; border-bottom: 1px solid #E2E8F0; transition: background-color 0.2s;'>"
        html += f"<td style='padding: 10px 16px; font-weight: 700; border-right: 1px solid #E2E8F0; color: #1E293B;'>Q{q_id:02d}</td>"
        html += f"<td style='padding: 10px 16px; border-right: 1px solid #E2E8F0; font-weight: 600; color: #10B981;'>Option {correct_ans}</td>"
        
        for idx, student in enumerate(group_students):
            opt = student_opts[idx]
            is_c = is_corrects[idx]
            
            # Icon decoration
            cell_bg = "transparent"
            cell_color = "#EF4444"
            icon = "❌"
            if is_c == 1:
                cell_color = "#10B981"
                icon = "✅"
                
            cell_style = f"padding: 10px 16px; border-right: 1px solid #E2E8F0; text-align: center; color: {cell_color}; font-weight: 600; background-color: {cell_bg};"
            html += f"<td style='{cell_style}'>{icon} Opt {opt}</td>"
            
        html += f"<td style='padding: 10px 16px; font-weight: 700; text-align: center; color: {status_color};'>{status_text}</td>"
        html += "</tr>"
        
    html += "</tbody></table></div>"
    return html


def plot_group_timing_timeline(group_students, df_clean):
    """Generate interactive Plotly timeline showing pacing of student answers side-by-side"""
    import plotly.graph_objects as go
    fig = go.Figure()
    
    for student in group_students:
        student_data = df_clean[df_clean['student_id'] == student].sort_values('question_id')
        fig.add_trace(go.Scatter(
            x=student_data['question_id'],
            y=student_data['time_spent'],
            mode='lines+markers',
            name=student,
            line=dict(width=2.5),
            marker=dict(size=6),
            hovertemplate="Question %{x}<br>Time spent: %{y:.1f}s"
        ))
        
    fig.update_layout(
        title="<b>Exam Timing & Pacing Alignment</b>",
        xaxis_title="Question ID",
        yaxis_title="Time Spent per Question (seconds)",
        hovermode="x unified",
        plot_bgcolor='rgba(248, 250, 252, 0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(b=40, l=40, r=20, t=50),
        xaxis=dict(gridcolor='#E2E8F0'),
        yaxis=dict(gridcolor='#E2E8F0'),
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E2E8F0',
            borderwidth=1
        )
    )
    return fig


def plot_pairwise_metric_comparison(group_students, df_clean):
    """Plot side-by-side behavioral metrics of students in the group"""
    import plotly.express as px
    
    # Calculate group member stats
    group_df = df_clean[df_clean['student_id'].isin(group_students)]
    student_stats = group_df.groupby('student_id').agg({
        'time_spent': 'mean',
        'is_correct': 'mean',
        'answer_changed': 'mean'
    }).reset_index()
    
    student_stats = student_stats.rename(columns={
        'time_spent': 'Avg Time (s)',
        'is_correct': 'Accuracy Rate',
        'answer_changed': 'Answer Change Rate'
    })
    
    # Melt for grouped plotting
    melted = student_stats.melt(id_vars='student_id', var_name='Metric', value_name='Value')
    
    fig = px.bar(
        melted,
        x='Metric',
        y='Value',
        color='student_id',
        barmode='group',
        text_auto='.2f',
        title="<b>Behavioral Profile Metrics Comparison</b>"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(248, 250, 252, 0.95)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="",
        yaxis_title="Score / Value",
        margin=dict(b=40, l=40, r=20, t=50),
    )
    return fig


def render_dashboard_metrics(total_students, graph_edges, total_groups, suspicious_groups):
    """Render beautiful HSL/Outfit metric cards grid"""
    st.markdown(f"""
    <div class="dashboard-grid">
        <div class="metric-card metric-card-normal">
            <div class="metric-value">{total_students}</div>
            <div class="metric-label">Total Students</div>
        </div>
        <div class="metric-card metric-card-normal">
            <div class="metric-value">{graph_edges}</div>
            <div class="metric-label">Graph Edges</div>
        </div>
        <div class="metric-card metric-card-normal">
            <div class="metric-value">{total_groups}</div>
            <div class="metric-label">Connected Groups</div>
        </div>
        <div class="metric-card metric-card-suspicious">
            <div class="metric-value" style="color: #EF4444;">{suspicious_groups}</div>
            <div class="metric-label">🚨 Suspicious Groups</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def download_template_csv():
    """Returns sample CSV template"""
    template_cols = ["student_id", "question_id", "time_spent", "selected_option", "is_correct", "answer_changed", "timestamp"]
    sample_data = [
        ["S0001", 1, 45.2, 3, 1, 0, "2024-01-15 09:00:00"],
        ["S0001", 2, 72.1, 1, 0, 1, "2024-01-15 09:01:15"],
        ["S0002", 1, 41.5, 3, 1, 0, "2024-01-15 09:00:00"],
        ["S0002", 2, 70.0, 1, 0, 1, "2024-01-15 09:01:10"]
    ]
    df = pd.DataFrame(sample_data, columns=template_cols)
    return df.to_csv(index=False)


def main():
    # Header banner
    st.markdown("""
    <div class="main-title-container">
        <div class="main-title">🎓 AI-Based Peer Behavior similarity Detector</div>
        <div class="main-subtitle">Secure Online Testing Analytics & Coordinated Collusion Identifier</div>
    </div>
    """, unsafe_allow_html=True)
    
    config = load_config()
    
    # Sidebar Setup
    st.sidebar.markdown("""
    <div style='text-align: center; padding-bottom: 10px;'>
        <span style='font-size: 32px;'>⚙️</span>
        <h3 style='margin: 5px 0 15px 0; color: #1E3A8A;'>System Control</h3>
    </div>
    """, unsafe_allow_html=True)
    
    data_source = st.sidebar.radio(
        "Data Source Selection",
        ["🎲 Generate Synthetic Dataset", "📁 Upload Custom CSV File"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Adjust Detection Sensitivity")
    
    similarity_threshold = st.sidebar.slider(
        "Edge Similarity Threshold",
        min_value=0.50,
        max_value=0.95,
        value=config['similarity']['min_similarity_threshold'],
        step=0.01,
        help="Minimum similarity required to link two students in the graph."
    )
    
    suspicious_threshold = st.sidebar.slider(
        "Group Flagging Threshold",
        min_value=0.60,
        max_value=0.95,
        value=config['detection']['suspicious_similarity_threshold'],
        step=0.01,
        help="Average similarity score required to flag a group as suspicious."
    )
    
    min_group_size = st.sidebar.number_input(
        "Minimum Group size to Flag",
        min_value=2,
        max_value=15,
        value=config['detection']['min_group_size'],
        help="Coordinated groups with at least this size will be investigated."
    )
    
    # Initialize session state for caching run data
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'ground_truth' not in st.session_state:
        st.session_state.ground_truth = None
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    if 'last_run_params' not in st.session_state:
        st.session_state.last_run_params = {}
        
    # Main workflow logic
    if data_source == "🎲 Generate Synthetic Dataset":
        st.subheader("📊 Synthetic Dataset Generator")
        
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                num_students = st.number_input("Normal Students Profile Count", 40, 800, 100)
            with col2:
                num_questions = st.number_input("Exam Questions Count", 10, 150, 50)
            with col3:
                num_groups = st.number_input("Collusion Groups to Inject", 0, 10, 3)
                
            if st.button("🎲 Generate and load Dataset", use_container_width=True):
                with st.spinner("Generating synthetic interaction logs..."):
                    current_seed = random.randint(1, 100000)
                    generator = ExamDataGenerator(
                        num_normal_students=num_students,
                        num_questions=num_questions,
                        exam_duration_minutes=120,
                        seed=current_seed
                    )
                    
                    # Available student IDs
                    all_ids = list(range(1, num_students + 1))
                    random.shuffle(all_ids)
                    
                    colluding_groups = []
                    current_idx = 0
                    group_sizes = [4, 5, 3, 6, 2]
                    
                    for i in range(num_groups):
                        size = group_sizes[i % len(group_sizes)]
                        if current_idx + size <= len(all_ids):
                            group_ids = sorted(all_ids[current_idx : current_idx + size])
                            colluding_groups.append(group_ids)
                            current_idx += size
                            
                    df, ground_truth = generator.generate_complete_dataset(colluding_groups)
                    
                    st.session_state.df = df
                    st.session_state.ground_truth = ground_truth
                    # Reset previous run results
                    st.session_state.pipeline_results = None
                    st.success(f"✅ Generated {len(df)} records for {df['student_id'].nunique()} students.")
                    
        if st.session_state.ground_truth is not None:
            st.info("🎯 **Ground Truth (Injected Colluders):**\n\n" + 
                    "\n".join([f"- Group {i+1}: {', '.join(g)}" for i, g in enumerate(st.session_state.ground_truth)]))
                    
    else:
        st.subheader("📁 Upload Custom CSV File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file containing exam interaction logs",
            type=['csv'],
            help="Required columns: student_id, question_id, time_spent, selected_option, is_correct, answer_changed, timestamp"
        )
        
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                # Validation check
                validator = ExamDataPreprocessor()
                is_valid, errors = validator.validate_data(uploaded_df)
                
                if not is_valid:
                    st.error("🚨 **CSV File Validation Failed:**")
                    for err in errors:
                        st.write(f"- {err}")
                else:
                    st.session_state.df = uploaded_df
                    st.session_state.ground_truth = None
                    st.session_state.pipeline_results = None
                    st.success(f"✅ Successfully loaded {len(uploaded_df)} records.")
            except Exception as e:
                st.error(f"🚨 Error reading CSV: {str(e)}")
                
        # Provide template download
        template_csv = download_template_csv()
        st.download_button(
            label="📥 Download Reference CSV Schema Template",
            data=template_csv,
            file_name="collusion_detector_template.csv",
            mime="text/csv"
        )
        
    # Run collusion detection button
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Check if parameters changed, forcing re-run
        current_params = {
            'sim_t': similarity_threshold,
            'susp_t': suspicious_threshold,
            'min_sz': min_group_size,
            'df_len': len(df)
        }
        
        st.markdown("---")
        if st.button("🚀 Analyze Peer Behavior & Detect Collusion", type="primary", use_container_width=True):
            with st.spinner("Processing logs, computing similarities and building networks..."):
                results = run_detection_pipeline(
                    df,
                    similarity_threshold,
                    suspicious_threshold,
                    min_group_size,
                    config
                )
                st.session_state.pipeline_results = results
                st.session_state.last_run_params = current_params
                st.success("✅ Analysis Complete!")
                
        # If pipeline has run successfully, show dashboard tabs
        if st.session_state.pipeline_results is not None:
            res = st.session_state.pipeline_results
            
            # Verify if parameters match, warning if they don't
            if current_params != st.session_state.last_run_params:
                st.warning("⚠️ **Parameters changed in the sidebar.** Click 'Analyze Peer Behavior & Detect Collusion' to update results.")
                
            report_df = res['report_df']
            suspicious_df = report_df[report_df['is_suspicious']]
            
            # Layout the tabs
            tab_overview, tab_network, tab_deepdive, tab_heatmap, tab_explorer = st.tabs([
                "📊 Overview Dashboard", 
                "🔗 Collusion Network Map", 
                "🔎 Deep-Dive Evidence", 
                "📈 Heatmaps & Distribution", 
                "📁 Raw Data Explorer"
            ])
            
            # TAB 1: OVERVIEW
            with tab_overview:
                # Top metrics grid
                render_dashboard_metrics(
                    total_students=res['df_clean']['student_id'].nunique(),
                    graph_edges=res['graph'].number_of_edges(),
                    total_groups=len(res['components']),
                    suspicious_groups=len(suspicious_df)
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🚨 Detected Suspicious Student Groups")
                    if len(suspicious_df) > 0:
                        for _, row in suspicious_df.iterrows():
                            st.error(f"**Group {row['group_id']}** - {row['group_size']} Students<br>"
                                     f"• **Avg Similarity:** {row['avg_similarity']:.3f}<br>"
                                     f"• **Shared Wrong Answers:** {row['wrong_overlap_ratio']*100:.1f}%<br>"
                                     f"• **Suspected Members:** `{row['students']}`", 
                                     icon="⚠️")
                    else:
                        st.success("No suspicious groups detected with the current sensitivity thresholds.", icon="✅")
                        
                with col2:
                    st.subheader("📚 Explanation of Indicators")
                    st.markdown("""
                    The collusion detection engine cross-analyzes exam logs using three key vectors:
                    - **Behavioral Similarity (Cosine Similarity)**: Measures speed consistency, statistical timing variance, answer adjustment rates, and Accuracy profile correlations.
                    - **Wrong Answer Overlap (Jaccard Similarity)**: Evaluates if students share similar wrong answer options. High overlap is a strong indicator of answer sharing.
                    - **Community Clustering**: Detects tightly-coupled networks of similarity using greedy modularity maximization algorithms.
                    """)
                    
            # TAB 2: NETWORK MAP
            with tab_network:
                st.subheader("🔗 Network Visualization")
                st.write("Zoom, pan, and hover over individual nodes to inspect student accuracy profiles and similarities.")
                
                # Render interactive Plotly network
                visualizer = CollusionVisualizer()
                # Aggregate student details for interactive hovers
                student_stats = res['df_clean'].groupby('student_id').agg(
                    accuracy_rate=('is_correct', 'mean'),
                    avg_time_per_question=('time_spent', 'mean'),
                    answer_change_rate=('answer_changed', 'mean')
                ).reset_index()
                
                # Highlighting suspicious groups
                susp_groups_sets = [set(r['students']) for r in res['analysis_results'] if r['is_suspicious']]
                
                fig_net = visualizer.plot_interactive_network(
                    res['graph'], 
                    susp_groups_sets, 
                    student_stats
                )
                st.plotly_chart(fig_net, use_container_width=True)
                
                # Show tabular connection index
                st.subheader("📋 Detected Coordinated Networks Summary")
                st.dataframe(report_df, use_container_width=True)
                
            # TAB 3: DEEP DIVE EVIDENCE
            with tab_deepdive:
                st.subheader("🔎 Individual Student Collusion Deep-Dive")
                st.write("Select a suspected group of students below to inspect detailed timeline alignments and answer patterns.")
                
                if len(suspicious_df) > 0:
                    group_options = {f"Group {row['group_id']} ({row['group_size']} students, Avg Sim: {row['avg_similarity']:.3f})": row['students'].split(', ') 
                                     for _, row in suspicious_df.iterrows()}
                    selected_group_label = st.selectbox("Select Suspected Group to Investigate", list(group_options.keys()))
                    selected_students = group_options[selected_group_label]
                    
                    st.markdown("---")
                    
                    # Columns for metrics and timeline chart
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        # Timeline Pace Correlation
                        fig_timeline = plot_group_timing_timeline(selected_students, res['df_clean'])
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    with col_chart2:
                        # Bar comparison
                        fig_bar = plot_pairwise_metric_comparison(selected_students, res['df_clean'])
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                    # Side by side selected options HTML table
                    st.subheader("❌ Shared Wrong Option Choice Patterns Matrix")
                    st.write("Compare the options chosen by each student side-by-side. Rows highlighted in **Red** indicate the exact same wrong choice (direct collusion signature).")
                    wrong_ans_html = render_wrong_answer_grid(selected_students, res['df_clean'])
                    st.markdown(wrong_ans_html, unsafe_allow_html=True)
                    
                else:
                    st.info("No suspicious groups detected to run deep-dive analysis on.")
                    
            # TAB 4: HEATMAPS & STATS
            with tab_heatmap:
                col_heat1, col_heat2 = st.columns(2)
                
                visualizer = CollusionVisualizer()
                
                with col_heat1:
                    st.subheader("🔥 Similarity Heatmap")
                    fig_heat = visualizer.plot_interactive_heatmap(res['similarities']['composite'])
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                with col_heat2:
                    st.subheader("🎯 Suspicion Boundary Matrix")
                    # Pair flat-table preparation
                    cosine_mat = res['calculator'].compute_cosine_similarity(res['feature_matrix'])
                    jaccard_mat = res['calculator'].compute_jaccard_similarity(res['wrong_answer_matrix'])
                    
                    mask = np.triu(np.ones_like(cosine_mat, dtype=bool), k=1)
                    rows, cols = np.where(mask)
                    student_ids = cosine_mat.index
                    
                    suspicious_pairs_set = set()
                    for group_data in res['analysis_results']:
                        if group_data['is_suspicious']:
                            students = group_data['students']
                            import itertools
                            for s1, s2 in itertools.combinations(students, 2):
                                suspicious_pairs_set.add(frozenset([s1, s2]))
                    
                    pair_labels = []
                    for r, c in zip(rows, cols):
                        s1, s2 = student_ids[r], student_ids[c]
                        if frozenset([s1, s2]) in suspicious_pairs_set:
                            pair_labels.append('Suspicious')
                        else:
                            pair_labels.append('Normal')
                            
                    df_pairs = pd.DataFrame({
                        'Behavior Similarity': cosine_mat.values[rows, cols],
                        'Shared Wrong Answers': jaccard_mat.values[rows, cols],
                        'Overall Match Score': res['similarities']['composite'].values[rows, cols],
                        'Group': pair_labels
                    }).sort_values('Group', ascending=True)
                    
                    fig_scatter = visualizer.plot_interactive_scatter(df_pairs)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
            # TAB 5: DATA EXPLORER & REPORTS
            with tab_explorer:
                st.subheader("📁 Report Downloads")
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    csv_groups = report_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download Coordinated Groups Report (CSV)",
                        data=csv_groups,
                        file_name="suspicious_groups_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col_d2:
                    csv_pairs = res['top_pairs'].to_csv(index=False)
                    st.download_button(
                        label="📄 Download Similar Pairs Matrix (CSV)",
                        data=csv_pairs,
                        file_name="similar_pairs_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                st.markdown("---")
                st.subheader("👀 View Clean Preprocessed Data Preview")
                st.dataframe(res['df_clean'], use_container_width=True)


if __name__ == "__main__":
    main()
