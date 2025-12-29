# SADRA: Sensitivity-Aware Dynamic Rank Allocation 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![PEFT](https://img.shields.io/badge/🤗-PEFT-yellow)](https://github.com/huggingface/peft)

**SADRA** (Sensitivity-Aware Dynamic Rank Allocation) is a novel Parameter-Efficient Fine-Tuning (PEFT) framework designed for Large Language Models (LLMs). Unlike standard Low-Rank Adaptation (LoRA) which enforces a uniform rank distribution across all layers, SADRA leverages the **Hessian geometry of the loss landscape** to dynamically allocate parameter budgets to the most sensitive modules.

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Methodology](#-methodology)
- [Benchmark Results](#-benchmark-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Citation](#-citation)

## 🔭 Overview

Standard LoRA approaches typically fix the rank (e.g., $r=8$) for all Transformer blocks. However, recent interpretability studies suggest that different layers encode distinct linguistic features (syntactic vs. semantic), implying heterogeneous plasticity requirements.

**SADRA solves this by:**
1.  **Probing:** Analyzing the curvature of the loss landscape using a small calibration dataset.
2.  **Allocating:** Assigning higher ranks to layers with sharp minima (high sensitivity) and pruning layers in flat minima.
3.  **Training:** Fine-tuning the model with an optimized, non-uniform rank topology.

## 🚀 Key Features

* **Hessian-Aware Sensitivity:** Utilizes **Hutchinson's Trace Estimator** to approximate the Hessian trace efficiently via Hessian-Vector Products (HvP), avoiding expensive $O(N^2)$ matrix computations.
* **One-Shot Calibration:** Unlike iterative pruning methods (e.g., AdaLoRA) that destabilize optimizer states, SADRA determines the optimal rank configuration **before** training begins.
* **Zero Training Overhead:** Once initialized, the training throughput is identical to standard LoRA.
* **VRAM Efficient:** Designed to work on consumer-grade GPUs, with specific optimizations for memory-intensive tasks like QNLI.

## 🔬 Methodology

SADRA allocates ranks ($r_i$) proportional to the layer's sensitivity ($S_i$):

$$S_i \approx \mathbb{E}[v^T (\nabla^2 \mathcal{L}) v]$$

Where $v$ is a Rademacher random vector. This allows us to identify "knowledge hubs" (typically FFN layers) and allocate ranks up to $r=32$, while compressing less critical Attention heads to $r=2$.

## 📊 Benchmark Results

We evaluated SADRA on the **GLUE Benchmark** using a `DistilBERT-base-uncased` backbone. All experiments were conducted with a comparable parameter budget to a static LoRA baseline ($r=8$, ~1.99% trainable parameters).

| Task | Metric | Full Fine-Tuning | Static LoRA (r=8) | SADRA (Ours) |
| :--- | :---: | :---: | :---: | :---: |
| **SST-2** | Accuracy | 90.10% | 89.20% | **89.68%** |
| **MRPC** | F1 Score | 88.50% | 87.10% | **87.96%** |
| **QNLI** | Accuracy | 89.80% | 88.40% | **88.30%** |
| **Avg. Params** | - | 100% | ~1.99% | **~1.99%** |

> **Note:** Despite using the same number of parameters, SADRA consistently outperforms the static baseline by redistributing capacity to where it matters most.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/melihkalkan4/SADRA-Project.git](https://github.com/melihkalkan4/SADRA-Project.git)
    cd SADRA-Project
    ```

2.  **Create a virtual environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Dependencies include: `torch`, `transformers`, `peft`, `datasets`, `evaluate`, `scikit-learn`, `matplotlib`, `seaborn`)*

## 🏃 Usage

### 1. Run Full Benchmark
To reproduce the results for SST-2, MRPC, and QNLI sequentially:
```bash
python scripts/run_full_benchmark.py

2. Run QNLI Only (Memory Optimized)
For larger datasets like QNLI, use the optimized script which employs Gradient Accumulation and reduced batch sizes to fit into limited VRAM:
python scripts/run_qnli_only.py
3. Generate Visualization Plots
To generate the Loss Curve and Rank Distribution plots used in the paper:
python scripts/generate_final_plots.py
Outputs will be saved in the ./output directory.
📂 Project Structure
SADRA-Project/
├── sadra/                  # Core Library
│   ├── core.py             # Hessian Trace Estimation logic
│   ├── engine.py           # PEFT Model Construction & Rank Injection
│   └── manager.py          # Budget Management & Rank Allocation
├── scripts/                # Experiment Scripts
│   ├── run_full_benchmark.py
│   ├── run_qnli_only.py    # Specialized script for QNLI
│   └── generate_final_plots.py
├── output/                 # Checkpoints, Logs, and Plots
├── main.tex                # LaTeX Source of the Research Paper
└── README.md               # Project Documentation

📜 Citation
If you find this research useful, please cite our paper:
@inproceedings{kalkan2025sadra,
  title={SADRA: Sensitivity-Aware Dynamic Rank Allocation for Parameter-Efficient Fine-Tuning},
  author={Kalkan, Melih},
  year={2025},
  booktitle={Preprint},
  note={Available at GitHub: [https://github.com/melihkalkan4/SADRA-Project](https://github.com/melihkalkan4/SADRA-Project)}
}

👨‍💻 Author
Melih Kalkan Independent Researcher & Data Analyst LinkedIn | Email

This project is released under the MIT License.
