"""
Main CLI Entry Point
End-to-end pipeline for AI-based peer behavior similarity detection
"""

import os
import sys
import yaml
import pandas as pd
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import ExamDataGenerator
from preprocessing import ExamDataPreprocessor
from feature_engineering import FeatureEngineer
from similarity_metrics import SimilarityCalculator
from graph_builder import SimilarityGraphBuilder
from collusion_detector import CollusionDetector
from visualization import CollusionVisualizer


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_pipeline(data_path: str = None, config_path: str = 'config.yaml', overrides: dict = None):
    """
    Run the complete collusion detection pipeline
    
    Args:
        data_path: Path to exam data CSV (if None, generates synthetic data)
        config_path: Path to configuration file
    """
    print("=" * 70)
    print("🎓 AI-BASED PEER BEHAVIOR SIMILARITY DETECTOR")
    print("=" * 70)
    
    # Load configuration
    config = load_config(config_path)
    overrides = overrides or {}
    # Apply any overrides provided (CLI)
    for k, v in overrides.items():
        section, key = k.split('.') if '.' in k else (None, None)
        if section and key and section in config:
            config[section][key] = v
    
    # Create output directories
    os.makedirs(config['output']['data_dir'], exist_ok=True)
    os.makedirs(config['output']['reports_dir'], exist_ok=True)
    os.makedirs(config['output']['graphs_dir'], exist_ok=True)
    
    # Step 1: Load or generate data
    print("\n📊 STEP 1: Data Loading")
    print("-" * 70)
    
    if data_path is None:
        print("Generating synthetic exam data...")
        generator = ExamDataGenerator(
            num_normal_students=config['data_generation']['num_normal_students'],
            num_questions=config['data_generation']['num_questions'],
            exam_duration_minutes=config['data_generation']['exam_duration_minutes'],
            seed=config['data_generation'].get('seed', 42)
        )
        
        # Create colluding groups
        colluding_groups = []
        start_id = 1
        for size in config['data_generation']['students_per_group']:
            group = list(range(start_id, start_id + size))
            colluding_groups.append(group)
            start_id += size
        
        df, ground_truth = generator.generate_complete_dataset(colluding_groups)
        
        # Save dataset
        data_save_path = os.path.join(config['output']['data_dir'], 'exam_data.csv')
        generator.save_dataset(df, data_save_path)
        
        print(f"\n🎯 Ground Truth Colluding Groups:")
        for i, group in enumerate(ground_truth, 1):
            print(f"   Group {i}: {group}")
    else:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        ground_truth = None
        print(f"✅ Loaded {len(df)} records")
    
    # Step 2: Preprocessing
    print("\n🔧 STEP 2: Data Preprocessing")
    print("-" * 70)
    
    preprocessor = ExamDataPreprocessor()
    df_clean = preprocessor.preprocess(df)
    
    # Step 3: Feature Engineering
    print("\n🔨 STEP 3: Feature Engineering")
    print("-" * 70)
    
    engineer = FeatureEngineer()
    feature_matrix = engineer.create_feature_matrix(df_clean, normalize=True)
    wrong_answer_matrix = engineer.extract_wrong_answer_vectors(df_clean)
    
    print(f"   Feature matrix shape: {feature_matrix.shape}")
    print(f"   Wrong answer matrix shape: {wrong_answer_matrix.shape}")
    
    # Step 4: Similarity Computation
    print("\n📐 STEP 4: Similarity Computation")
    print("-" * 70)
    
    calculator = SimilarityCalculator(
        cosine_weight=config['similarity']['cosine_weight'],
        jaccard_weight=config['similarity']['jaccard_weight']
    )
    
    similarities = calculator.compute_all_similarities(
        feature_matrix, 
        wrong_answer_matrix
    )
    
    # Get top similar pairs
    top_pairs = calculator.get_top_similar_pairs(
        similarities['composite'],
        threshold=config['similarity']['min_similarity_threshold'],
        top_n=20
    )
    
    print(f"\n   Top similar pairs (threshold={config['similarity']['min_similarity_threshold']}):")
    print(top_pairs.head(10).to_string(index=False))
    
    # Save pairs report
    pairs_report_path = os.path.join(config['output']['reports_dir'], 'similar_pairs.csv')
    top_pairs.to_csv(pairs_report_path, index=False)
    print(f"\n✅ Similar pairs saved to {pairs_report_path}")
    
    # Step 5: Graph Construction
    print("\n🔗 STEP 5: Graph Construction")
    print("-" * 70)
    
    graph_builder = SimilarityGraphBuilder(
        similarity_threshold=config['similarity']['min_similarity_threshold'],
        knn_k=config['similarity'].get('knn_k')
    )
    
    graph = graph_builder.build_graph(
        similarities['composite'],
        jaccard_matrix=similarities['jaccard'],
        jaccard_edge_threshold=config['similarity'].get('edge_wrong_overlap_threshold', None)
    )
    components = graph_builder.get_connected_components(
        use_community_detection=bool(config['similarity'].get('use_community_detection', True)),
        dynamic_prune=bool(config['similarity'].get('dynamic_prune', True))
    )
    
    # Step 6: Collusion Detection
    print("\n🚨 STEP 6: Collusion Detection")
    print("-" * 70)
    
    detector = CollusionDetector(
        suspicious_similarity_threshold=config['detection']['suspicious_similarity_threshold'],
        min_group_size=config['detection']['min_group_size'],
        wrong_answer_overlap_threshold=config['detection']['wrong_answer_overlap_threshold']
    )
    
    analysis_results = detector.detect_suspicious_groups(
        components,
        similarities['composite'],
        wrong_answer_matrix
    )
    
    # Generate report
    report_df = detector.generate_report(analysis_results)
    
    print("\n📋 Suspicious Groups Report:")
    print(report_df[report_df['is_suspicious']].to_string(index=False))
    
    # Save report
    groups_report_path = os.path.join(config['output']['reports_dir'], 'suspicious_groups.csv')
    report_df.to_csv(groups_report_path, index=False)
    print(f"\n✅ Groups report saved to {groups_report_path}")
    
    # Step 7: Visualization
    print("\n📊 STEP 7: Visualization")
    print("-" * 70)
    
    visualizer = CollusionVisualizer(output_dir=config['output']['graphs_dir'])
    
    # Get suspicious groups
    suspicious_groups = [
        set(result['students']) 
        for result in analysis_results 
        if result['is_suspicious']
    ]
    
    # Plot similarity graph
    graph_path = os.path.join(config['output']['graphs_dir'], 'similarity_graph.png')
    visualizer.plot_similarity_graph(graph, suspicious_groups, save_path=graph_path)
    
    # Plot heatmap
    heatmap_path = os.path.join(config['output']['graphs_dir'], 'similarity_heatmap.png')
    visualizer.plot_similarity_heatmap(similarities['composite'], save_path=heatmap_path)
    
    # Plot group statistics
    stats_path = os.path.join(config['output']['graphs_dir'], 'group_statistics.png')
    visualizer.plot_group_statistics(analysis_results, save_path=stats_path)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n📁 Outputs:")
    print(f"   Reports: {config['output']['reports_dir']}")
    print(f"   Graphs: {config['output']['graphs_dir']}")
    print(f"   Data: {config['output']['data_dir']}")
    
    print(f"\n📊 Summary:")
    print(f"   Total students analyzed: {df_clean['student_id'].nunique()}")
    print(f"   Suspicious groups detected: {len(suspicious_groups)}")
    print(f"   Total students flagged: {sum(len(g) for g in suspicious_groups)}")
    
    if ground_truth:
        print(f"\n🎯 Ground Truth vs Detection:")
        print(f"   Actual colluding groups: {len(ground_truth)}")
        print(f"   Detected suspicious groups: {len(suspicious_groups)}")

        # Evaluate detected groups against ground truth using Jaccard overlap
        def jaccard(a: set, b: set) -> float:
            return len(a & b) / len(a | b) if (a or b) else 0.0

        gt_sets = [set(g) for g in ground_truth]
        detected_sets = suspicious_groups

        print("\n🔍 Group overlap (Jaccard):")
        matched = 0
        for i, gt in enumerate(gt_sets, 1):
            scores = [jaccard(gt, det) for det in detected_sets]
            best = max(scores) if scores else 0.0
            best_idx = scores.index(best) + 1 if scores else -1
            print(f"   GT Group {i}: best match Detected {best_idx} with Jaccard={best:.3f}")
            if best >= 0.5:
                matched += 1

        precision = matched / len(detected_sets) if detected_sets else 0
        recall = matched / len(gt_sets) if gt_sets else 0
        print(f"\n✅ Detection quality: precision={precision:.2f}, recall={recall:.2f}")

        # Optional: simple parameter sweep to improve metrics
        print("\n🧪 Parameter sweep for better accuracy (quick search)...")
        best = {'precision': 0, 'recall': 0, 'f1': 0, 'cfg': None}
        sim_thresholds = [0.65, 0.68, 0.70]
        jaccard_edges = [0.50, 0.60]
        knn_vals = [None, 3]
        susp_thresh = [0.72, 0.75, 0.78]
        
        for st in sim_thresholds:
            for jt in jaccard_edges:
                for kk in knn_vals:
                    gb = SimilarityGraphBuilder(similarity_threshold=st, knn_k=kk)
                    g2 = gb.build_graph(
                        similarities['composite'],
                        jaccard_matrix=similarities['jaccard'],
                        jaccard_edge_threshold=jt
                    )
                    comps2 = gb.get_connected_components()
                    det = CollusionDetector(
                        suspicious_similarity_threshold=max(susp_thresh),
                        min_group_size=config['detection']['min_group_size'],
                        wrong_answer_overlap_threshold=config['detection']['wrong_answer_overlap_threshold']
                    )
                    res = det.detect_suspicious_groups(comps2, similarities['composite'], wrong_answer_matrix)
                    det_sets = [set(r['students']) for r in res if r['is_suspicious']]
                    matched = 0
                    for gt in gt_sets:
                        scores = [jaccard(gt, ds) for ds in det_sets]
                        if scores and max(scores) >= 0.5:
                            matched += 1
                    precision2 = matched / len(det_sets) if det_sets else 0
                    recall2 = matched / len(gt_sets) if gt_sets else 0
                    f1 = (2*precision2*recall2/(precision2+recall2)) if (precision2+recall2)>0 else 0
                    if f1 > best['f1']:
                        best = {'precision': precision2, 'recall': recall2, 'f1': f1, 'cfg': {'min_similarity_threshold': st, 'edge_wrong_overlap_threshold': jt, 'knn_k': kk}}
        
        if best['cfg']:
            print(f"\n⭐ Best quick sweep: precision={best['precision']:.2f}, recall={best['recall']:.2f}, f1={best['f1']:.2f}")
            print(f"   Suggested config: similarity.min_similarity_threshold={best['cfg']['min_similarity_threshold']}, similarity.edge_wrong_overlap_threshold={best['cfg']['edge_wrong_overlap_threshold']}, similarity.knn_k={best['cfg']['knn_k']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI Collusion Detector')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML')
    parser.add_argument('--data', type=str, default=None, help='Path to CSV data (optional)')
    parser.add_argument('--min-sim', type=float, help='Override similarity.min_similarity_threshold')
    parser.add_argument('--edge-wrong', type=float, help='Override similarity.edge_wrong_overlap_threshold')
    parser.add_argument('--knn-k', type=int, help='Override similarity.knn_k')
    parser.add_argument('--seed', type=int, help='Override data_generation.seed')
    args = parser.parse_args()

    overrides = {}
    if args.min_sim is not None:
        overrides['similarity.min_similarity_threshold'] = args.min_sim
    if args.edge_wrong is not None:
        overrides['similarity.edge_wrong_overlap_threshold'] = args.edge_wrong
    if args.knn_k is not None:
        overrides['similarity.knn_k'] = args.knn_k
    if args.seed is not None:
        overrides['data_generation.seed'] = args.seed

    run_pipeline(data_path=args.data, config_path=args.config, overrides=overrides)
