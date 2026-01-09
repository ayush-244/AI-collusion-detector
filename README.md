# 🎓 AI-Based Peer Behavior Similarity Detector

A production-ready AI system designed to detect collusion and coordinated cheating in online exams. By analyzing student interaction logs, this system identifies suspicious behavioral patterns using machine learning and graph theory.

## 🚀 Key Features

- **Behavioral Fingerprinting**: Extracts granular features like time-per-question, answer-changing patterns, and wrong-answer vectors.
- **Multi-Metric Similarity**: Combines Cosine Similarity (behavioral trends) and Jaccard Similarity (wrong answer overlap).
- **Graph-Based Detection**: Uses NetworkX to build similarity graphs and detect community structures representing colluding groups.
- **Synthetic Data Generator**: Built-in engine to simulate realistic normal and colluding exam behaviors.
- **Interactive Dashboard**: Streamlit-based UI for visualizing suspicious groups, heatmaps, and similarity networks.

## 📂 Project Structure

```
AI detection/
├── src/
│   ├── data_generator.py       # Simulates exam logs with colluding groups
│   ├── preprocessing.py        # Cleans and validates raw input data
│   ├── feature_engineering.py  # Extracts behavioral & timing features
│   ├── similarity_metrics.py   # Computes Cosine & Jaccard similarity
│   ├── graph_builder.py        # Constructs similarity graphs & communities
│   ├── collusion_detector.py   # Flags suspicious groups based on thresholds
│   └── visualization.py        # Generates plotting logic
├── data/                       # Directory for generated/uploaded datasets
├── outputs/                    # Reports and graph visualizations
├── app.py                      # Interactive Streamlit Dashboard
├── main.py                     # CLI Pipeline for batch processing
├── config.yaml                 # Configurable parameters (thresholds, weights)
└── requirements.txt            # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository** (or download the source code).
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

### Option 1: Interactive Dashboard (Recommended)
Launch the web interface to upload data or generate synthetic scenarios interactively.
```bash
streamlit run app.py
```

### Option 2: CLI Batch Processing
Run the full end-to-end pipeline from the command line. This will generate a synthetic dataset and produce reports in the `outputs/` folder.
```powershell
& .\.venv\Scripts\python.exe main.py
```

Optional CLI tuning flags (Windows):
```powershell
& .\.venv\Scripts\python.exe main.py --min-sim 0.65 --edge-wrong 0.50 --seed 42
```
Where:
- `--min-sim`: similarity.min_similarity_threshold (edge creation)
- `--edge-wrong`: similarity.edge_wrong_overlap_threshold (Jaccard gating)
- `--seed`: data_generation.seed (reproducibility)

## 🧠 algorithmic logic

1. **Feature Engineering**:
   - Converts raw logs into student-level vectors.
   - Captures **Time Variance**, **Answer Change Rates**, and **Option Entropy**.
2. **Similarity Computation**:
   - **Numerical**: Cosine similarity on normalized behavioral stats.
   - **Categorical**: Jaccard index on discrete "Wrong Answer" sets (students making the *same* mistakes is a strong collusion signal).
3. **Graph Analysis**:
   - Nodes = Students.
   - Edges = Composite similarity score > `0.65` (configurable).
   - Communities = Disconnected subgraphs representing isolated student clusters.
4. **Suspicion Flagging**:
   - Groups are flagged if they have unusually high internal similarity and wrong-answer overlap compared to the population baseline.

## 📊 Outputs

- **Suspicious Groups Report**: CSV listing flagged groups and reasons (e.g., "High Wrong Answer Overlap").
- **Similarity Graph**: Network visualization showing clusters.
- **Heatmaps**: Visual correlation matrix of student identifiers.

## ⚙️ Configuration

Modify `config.yaml` to tune sensitivity:
- `data_generation.seed`: reproducible dataset
- `similarity.min_similarity_threshold`: edge creation threshold
- `similarity.edge_wrong_overlap_threshold`: require wrong-overlap ≥ threshold
- `similarity.knn_k`: optional k-NN pruning (set null to disable)
- `similarity.use_community_detection` / `similarity.dynamic_prune`: enable group splitting
- `detection.*`: suspicion thresholds

## ✅ Evaluation (Synthetic)
`main.py` prints precision/recall and Jaccard overlaps versus ground truth. A quick parameter sweep suggests better thresholds for the current dataset.

---
*Built for educational integrity monitoring.*
