<div align="center">

# 🔍 AI Collusion Detector

### Production-Ready Academic Integrity Intelligence Platform

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-ai--collusion--detector.onrender.com-6366f1?style=for-the-badge)](https://ai-collusion-detector.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

> **Detect collusion and coordinated cheating in online exams** using machine learning, graph theory, and behavioral fingerprinting — in real time.

[🌐 **Open Live App**](https://ai-collusion-detector.onrender.com) · [📖 Documentation](#-algorithmic-logic) · [🐛 Report Bug](https://github.com/ayush-244/AI-collusion-detector/issues) · [💡 Request Feature](https://github.com/ayush-244/AI-collusion-detector/issues)

---

</div>

## 📸 Preview

> Launch the live dashboard at **[https://ai-collusion-detector.onrender.com](https://ai-collusion-detector.onrender.com)** to see the full interactive experience — no installation required.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **Behavioral Fingerprinting** | Extracts granular features — time-per-question, answer-change patterns, wrong-answer vectors |
| 📐 **Multi-Metric Similarity** | Combines **Cosine Similarity** (behavioral trends) + **Jaccard Similarity** (wrong-answer overlap) |
| 🕸️ **Graph-Based Detection** | Uses NetworkX to build similarity graphs and detect community clusters representing colluding groups |
| 🏭 **Synthetic Data Engine** | Built-in generator to simulate realistic normal and colluding exam behavior at scale |
| 📊 **Interactive Dashboard** | Streamlit UI with heatmaps, network graphs, and suspicious-group drill-downs |
| ⚙️ **Fully Configurable** | Tune thresholds, similarity weights, and detection sensitivity via `config.yaml` |

---

## 🌐 Live Demo

The application is deployed and publicly accessible at:

```
https://ai-collusion-detector.onrender.com
```

No installation or account required. Upload your own exam log CSV or use the built-in synthetic data generator to explore the system interactively.

---

## 📂 Project Structure

```
AI-collusion-detector/
├── src/
│   ├── data_generator.py       # Simulates exam logs with colluding groups
│   ├── preprocessing.py        # Cleans and validates raw input data
│   ├── feature_engineering.py  # Extracts behavioral & timing features
│   ├── similarity_metrics.py   # Computes Cosine & Jaccard similarity scores
│   ├── graph_builder.py        # Constructs similarity graphs & communities
│   ├── collusion_detector.py   # Flags suspicious groups based on thresholds
│   └── visualization.py        # Plotly-based chart rendering logic
├── data/                       # Directory for generated / uploaded datasets
├── outputs/                    # Reports and graph visualizations
├── app.py                      # 🖥️ Interactive Streamlit Dashboard
├── main.py                     # ⌨️  CLI Pipeline for batch processing
├── config.yaml                 # ⚙️  Configurable parameters (thresholds, weights)
├── Dockerfile                  # 🐳 Container definition for Render deployment
├── render.yaml                 # ☁️  Render service configuration
└── requirements.txt            # 📦 Python dependencies
```

---

## 🛠️ Local Setup

### Prerequisites
- Python **3.12+**
- `pip` (or `pipenv` / `conda`)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/ayush-244/AI-collusion-detector.git
cd AI-collusion-detector

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.\.venv\Scripts\Activate.ps1     # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the interactive dashboard
streamlit run app.py
```

Open your browser at **http://localhost:8501** to use the app locally.

---

## 🏃 Usage

### Option 1 — Interactive Dashboard (Recommended)

```bash
streamlit run app.py
```

Launches the full web UI at `http://localhost:8501`. Upload a CSV log file or click **"Generate Synthetic Data"** to explore a pre-built scenario instantly.

### Option 2 — CLI Batch Processing

Run the full end-to-end detection pipeline from the command line and save results to `outputs/`:

```powershell
# Windows PowerShell
& .\.venv\Scripts\python.exe main.py

# With optional tuning flags
& .\.venv\Scripts\python.exe main.py --min-sim 0.65 --edge-wrong 0.50 --seed 42
```

**CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--min-sim` | `0.65` | Minimum composite similarity score to create a graph edge |
| `--edge-wrong` | `0.50` | Jaccard wrong-answer overlap gate per edge |
| `--seed` | `42` | Random seed for reproducible synthetic datasets |

---

## 🧠 Algorithmic Logic

The detection pipeline runs in four stages:

```
Raw Logs → Feature Engineering → Similarity Matrix → Graph Analysis → Suspicious Groups
```

### 1️⃣ Feature Engineering
Converts raw exam logs into per-student behavioral vectors capturing:
- **Time Variance** — consistency of response times across questions
- **Answer Change Rate** — frequency of last-moment edits
- **Option Entropy** — distribution of answer choices selected

### 2️⃣ Similarity Computation
Two complementary metrics are computed for every student pair:

| Metric | Domain | What it captures |
|--------|--------|-----------------|
| **Cosine Similarity** | Numerical | Overall behavioral alignment |
| **Jaccard Index** | Categorical | Identical wrong-answer selection — the strongest collusion signal |

### 3️⃣ Graph Construction
- **Nodes** — individual students
- **Edges** — student pairs whose composite score exceeds the configurable threshold (default `0.65`)
- **Communities** — connected subgraphs representing isolated colluding clusters

### 4️⃣ Suspicion Flagging
Groups are flagged when their internal similarity and wrong-answer overlap exceed population-baseline thresholds, minimising false positives.

---

## 📊 Outputs

| Output | Format | Description |
|--------|--------|-------------|
| Suspicious Groups Report | CSV | Flagged groups with reasons (e.g., *"High Wrong Answer Overlap"*) |
| Similarity Graph | Interactive HTML | Network visualization of student clusters |
| Heatmaps | Plotly Chart | Colour-coded pairwise similarity matrix |

---

## ⚙️ Configuration

Edit `config.yaml` to tune detection sensitivity without changing any code:

```yaml
data_generation:
  seed: 42                        # Reproducible synthetic datasets

similarity:
  min_similarity_threshold: 0.65  # Edge creation threshold (lower = more edges)
  edge_wrong_overlap_threshold: 0.50  # Jaccard gate per edge
  knn_k: null                     # Optional k-NN pruning (null = disabled)
  use_community_detection: true   # Enable Louvain community splitting
  dynamic_prune: true             # Adaptive threshold pruning

detection:
  # Suspicion thresholds — adjust to control false positive rate
  min_group_size: 2
  suspicion_score_threshold: 0.75
```

---

## 📈 Evaluation

Running `main.py` outputs **precision**, **recall**, and **Jaccard overlaps** against ground-truth synthetic labels — useful for calibrating thresholds:

```
Precision : 0.92
Recall    : 0.88
F1-Score  : 0.90
```

> *Metrics are approximate and depend on the `--seed` and threshold configuration.*

---

## 🚀 Deployment

The app is containerised with Docker and deployed on **Render** (free tier):

```bash
# Build locally
docker build -t ai-collusion-detector .
docker run -p 8501:8501 ai-collusion-detector
```

To deploy your own instance, fork this repo and connect it to [render.com](https://render.com) — the `render.yaml` and `Dockerfile` are already configured for one-click deployment.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Commit your changes: `git commit -m 'Add my feature'`.
4. Push to the branch: `git push origin feature/my-feature`.
5. Open a Pull Request.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

Built with ❤️ for **educational integrity monitoring**

[🌐 Live App](https://ai-collusion-detector.onrender.com) · [⭐ Star this repo](https://github.com/ayush-244/AI-collusion-detector) · [🐛 Report an Issue](https://github.com/ayush-244/AI-collusion-detector/issues)

</div>
