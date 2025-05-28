# Entropy Steady-State Regularization for Deep Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](#)

> **Official implementation of "Entropy Steady-State Regularization for Deep Graph Neural Networks"**

## 🎯 Overview

This repository contains the implementation of **Entropy Steady-State Regularization (ESR)**, a novel regularization framework for training deep Graph Neural Networks (GNNs). Our approach addresses fundamental challenges in deep GNNs including over-smoothing, information degradation, and limited expressive power.

### Key Innovation

We discovered that information entropy in GNN layers converges to stable intervals during training - a phenomenon we term the **"entropy steady-state."** Based on this theoretical foundation, ESR explicitly guides GNN layers toward optimal entropy intervals using multiple entropy measures.

### 🔥 Highlights

- **📊 Significant Performance Gains**: Up to 24.1% accuracy improvement on 8-layer networks
- **🛡️ Enhanced Robustness**: Superior resistance to noise and adversarial attacks  
- **🔬 Rigorous Validation**: Comprehensive experiments across 360 configurations with p < 0.001 statistical significance
- **🎛️ Flexible Framework**: Support for Shannon, Rényi, and Tsallis entropy measures
- **⚡ Easy Integration**: Drop-in replacement for standard GNN training

## 📈 Results

| Dataset | Depth | Baseline | ESR-Shannon | ESR-Rényi | ESR-Tsallis |
|---------|-------|----------|-------------|-----------|-------------|
| Cora    | 2     | 79.8%    | 83.7%       | 83.0%     | 82.7%       |
| Cora    | 4     | 75.5%    | 80.2%       | 80.6%     | 80.7%       |
| Cora    | 8     | 56.6%    | 63.3%       | 68.6%     | **70.7%**   |
| Citeseer| 8     | 46.3%    | 40.2%       | 46.1%     | **56.1%**   |
| PubMed  | 8     | 50.2%    | 65.8%       | 63.6%     | **69.2%**   |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/prospong/time-emergence.git
cd time-emergence

# Create conda environment
conda create -n esr python=3.8
conda activate esr

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from esr import ESRRegularizer, ESRTrainer
from torch_geometric.nn import GCN

# Initialize your GNN model
model = GCN(num_features=1433, hidden_channels=64, num_classes=7, num_layers=8)

# Create ESR regularizer
esr_regularizer = ESRRegularizer(
    entropy_type='tsallis',  # 'shannon', 'renyi', 'tsallis'
    lambda_type='adaptive',   # 'fixed', 'adaptive', 'information_driven'
    alpha=0.1
)

# Train with ESR
trainer = ESRTrainer(model, esr_regularizer)
trainer.train(data, epochs=200)
```

### Run Experiments

```bash
# Single dataset experiment
python main.py --dataset cora --entropy_type tsallis --num_layers 8

# Comprehensive evaluation
python run_experiments.py --config configs/full_evaluation.yaml

# Robustness analysis
python robustness_analysis.py --noise_levels 0.1 0.2 --attack_types fgsm
```

## 🔬 Methodology

### Entropy Steady-State Phenomenon

Our key discovery is that entropy evolution in GNNs exhibits convergence to stable intervals:

```python
# Entropy computation for layer l
def compute_entropy(X_l, entropy_type='shannon'):
    if entropy_type == 'shannon':
        return -sum(p_i * log(p_i))
    elif entropy_type == 'renyi':
        return (1/(1-alpha)) * log(sum(p_i**alpha))
    elif entropy_type == 'tsallis':
        return (1/(q-1)) * (1 - sum(p_i**q))
```

### ESR Framework

The ESR regularization term:

```
L_ESR = Σ λ^(l) · D(H^(l), H_target^(l))
```

Where:
- `H^(l)` is the entropy at layer l
- `H_target^(l)` is the target entropy
- `λ^(l)` is the regularization weight (fixed/adaptive/information-driven)

## 🧪 Experiments

### Datasets
- **Cora**: 2,708 nodes, 5,429 edges (citation network)
- **Citeseer**: 3,327 nodes, 4,732 edges (citation network)  
- **PubMed**: 19,717 nodes, 44,338 edges (citation network)

### Experimental Configurations
- **Network Depths**: 2, 4, 8 layers
- **Entropy Measures**: Shannon, Rényi (α=2), Tsallis (q=2)
- **Lambda Values**: 0.05, 0.1, 0.15, 0.2
- **Control Mechanisms**: Fixed λ, Adaptive λ, Information-driven λ
- **Robustness Tests**: Gaussian noise (σ=0.1,0.2), FGSM attacks (ε=0.01,0.05)

### Reproduce Paper Results

```bash
# Table 1: Main results
python experiments/main_results.py

# Figure 2: Performance comparison
python experiments/performance_comparison.py

# Figure 3: Entropy measures comparison  
python experiments/entropy_measures.py

# Figure 4: Robustness analysis
python experiments/robustness_analysis.py

# Statistical analysis
python experiments/statistical_validation.py
```

## 📊 Analysis & Visualization

Interactive notebooks for result analysis:

```bash
# Start Jupyter
jupyter notebook

# Open analysis notebooks
notebooks/entropy_visualization.ipynb
notebooks/results_analysis.ipynb
```

## 🛠️ Advanced Usage

### Custom Entropy Measures

```python
from esr.entropy import EntropyMeasure

class CustomEntropy(EntropyMeasure):
    def compute(self, representations):
        # Your custom entropy implementation
        return entropy_value

# Use in ESR
esr = ESRRegularizer(entropy_measure=CustomEntropy())
```

### Hyperparameter Tuning

```python
from esr.tuning import ESRTuner

tuner = ESRTuner(
    search_space={
        'lambda_base': [0.05, 0.1, 0.15, 0.2],
        'entropy_type': ['shannon', 'renyi', 'tsallis'],
        'lambda_type': ['fixed', 'adaptive']
    }
)

best_config = tuner.tune(dataset='cora', num_trials=50)
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric 2.0+
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
- YAML

## 📚 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{tian2025entropy,
  title={Entropy Steady-State Regularization for Deep Graph Neural Networks},
  author={Tian, Zhigang},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Anonymous reviewers for valuable feedback
- PyTorch Geometric team for the excellent framework
- The graph neural networks community

## 📞 Contact

**Zhigang Tian** - zt62@student.london.ac.uk

**Project Link**: [https://github.com/prospong/time-emergence](https://github.com/prospong/esr-gnn)

---

⭐ **If you find this repository helpful, please give it a star!** ⭐
