# 4D-Transformer: Constraint-Enhanced Cognitive Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> 🧠 **4D Cognitive Architecture**: Integrating Self, Desire, Ethic, and Reflection dimensions into Transformer for constraint-enhanced text classification

**中文**: [README.md](README.md)

---

## 🎯 Project Significance

### Why 4D-Transformer?

In safety-critical applications (medical, finance, legal, etc.), AI models need not only high accuracy but also **strict adherence to constraint rules**. Traditional Transformer models perform poorly in constraint compliance, often with high violation rates.

**4D-Transformer** introduces four dimensions from cognitive science and specifically designs a **constraint enhancement mechanism** that significantly reduces constraint violation rates while maintaining high accuracy.

### Core Value

1. **Constraint Compliance**: Violation rate reduced from 0.65% to 0.00-0.01% (98%+ reduction) ✅
2. **Cognitive Architecture Innovation**: First integration of Self, Desire, Ethic, and Reflection dimensions into Transformer
3. **Domain Adaptation**: Domain Steering mechanism adapts the model to different application scenarios
4. **Stable and Reproducible Results**: Verified by multi-seed testing, accuracy difference only 0.14%

## 🚀 Quick Start

### Installation

```bash
pip install torch transformers datasets tqdm numpy
```

### Basic Usage

```python
from train_medical_dataset import FourDTransformerClassifier
import torch

# Create model
model = FourDTransformerClassifier(
    vocab_size=30522,
    d_model=192,
    nhead=8,
    num_layers=4,
    dim_feedforward=768,
    dropout=0.5,
    num_classes=2,
    state_dim=64,
    default_domain='generic'  # or 'medical', 'creative', 'finance'
)

# Switch domain configuration
model.set_domain('medical')  # Switch to medical domain

# Forward pass
input_ids = torch.randint(0, 30522, (32, 128))  # [batch_size, seq_len]
constraints = torch.zeros(32, 128)  # Constraint mask
logits = model(input_ids, constraints=constraints)
```

### Training

```bash
# Train on IMDb dataset
python train_medical_dataset.py
```

## 🧠 4D Cognitive Architecture

### Four Dimensions

1. **Self (S)**: Self-awareness
   - Provides stability and consistency
   - Maintains model's internal state

2. **Desire (D)**: Goal motivation
   - Drives exploration and learning
   - Enhances model's expressive power

3. **Ethic (G)**: Ethical constraints ⭐ **Core**
   - **Specifically handles constraint compliance**
   - Significantly reduces violation rate (from 0.65% to 0.00-0.01%)

4. **Reflection (R)**: Feedback mechanism
   - Corrects errors and adjusts
   - Provides self-correction capability

### Domain Adaptation (Domain Steering)

The Domain Steering mechanism dynamically adjusts the weights of the four dimensions to adapt to different application scenarios:

- **Generic**: Balanced configuration (S=1.0, D=1.0, G=1.0, R=1.0)
- **Medical**: Emphasizes constraints (S=1.1, D=1.1, G=1.3, R=1.2)
- **Creative**: Enhances exploration (S=0.9, D=1.5, G=0.8, R=0.9)
- **Finance**: Strictest constraints (S=1.3, D=0.8, G=1.7, R=1.6)

## 📊 Experimental Results

### Performance

| Configuration | Best Val Accuracy | Violation Rate | Train-Val Gap |
|---------------|-------------------|----------------|---------------|
| Generic | 77.39% | 0.00-0.01% | 17.58% |
| Medical | 77.16% | 0.00% | 17.64% |
| Creative | 77.18% | 0.00% | 17.69% |
| Finance | 77.02% | 0.00% | 17.80% |

**Compared to Baseline**:
- ✅ **Violation Rate**: 0.00-0.01% vs 0.65% (98%+ reduction)
- ⚠️ **Accuracy**: 77.39% vs 77.90% (difference -0.51%, acceptable trade-off)

### Stability Verification

Multi-seed testing (3 seeds):
- Mean accuracy: 77.39%
- Standard deviation: 0.07%
- Range: 77.30% - 77.44%
- **Conclusion**: Results are very stable ✅

## 📁 Project Structure

```
SolveMeLLM-4.0/
├── models/                          # Model implementations
│   ├── four_d_transformer_block-v2.py  # Core 4D-Transformer implementation
│   └── baseline_transformer.py         # Baseline Transformer
├── train_medical_dataset.py         # Main training script
├── medical_constrained_classification.py  # Dataset processing
├── docs/                            # Documentation
│   ├── architecture/                # Architecture design docs
│   ├── guides/                      # Usage guides
│   ├── results/training/            # Training results
│   └── evaluation/                  # Evaluation and analysis
└── scripts/                         # Utility scripts
    ├── test_planner_head.py         # Planner head testing
    └── test_multi_seed_generic.py   # Multi-seed testing
```

## 🔬 Research Background

### Motivation

Traditional Transformer models perform poorly in constraint compliance, especially in safety-critical applications (medical, finance, legal, etc.). This project explores integrating cognitive science dimensions into deep learning models to reduce violation rates through specialized constraint handling mechanisms.

### Core Contributions

1. **4D Cognitive Architecture**: First integration of Self, Desire, Ethic, and Reflection dimensions into Transformer
2. **Constraint Enhancement Mechanism**: Specialized constraint handling through Ethic dimension, significantly reducing violation rate (98%+)
3. **Domain Adaptation**: Domain Steering mechanism adapts the model to different application scenarios
4. **Experimental Validation**: Validated the method's effectiveness on IMDb dataset

## 💡 Application Scenarios

### Suitable Applications

1. **Medical Domain**: Requires strict constraints, reduces misdiagnosis risk
2. **Finance Domain**: Requires regulatory compliance, reduces violation risk
3. **Legal Domain**: Requires legal compliance, reduces legal risk
4. **Safety-Critical Systems**: Requires strict adherence to safety rules

### Core Advantages

- ✅ **Constraint Compliance**: Violation rate reduced by 98%+
- ✅ **Domain Adaptation**: Can adjust model behavior based on scenario
- ✅ **Interpretability**: 4D states provide interpretability for model decisions

## ⚠️ Known Issues & Optimization Directions

### Current Issues

1. **Overfitting**: Train-validation gap of ~17-18%, needs further optimization
2. **Accuracy**: Slightly lower than Baseline (difference -0.51%), a trade-off between accuracy and constraint compliance
3. **Training Time**: ~3x slower than Baseline (1 minute vs 18 seconds/epoch)

### Optimization Directions

We welcome community contributions for the following optimizations:

1. **Overfitting Optimization**
   - Earlier Early Stopping strategies
   - Data augmentation techniques
   - Stronger regularization methods

2. **Accuracy Improvement**
   - Optimize constraint loss weights
   - Improve domain profile weights
   - Explore new architecture designs

3. **Performance Optimization**
   - Optimize training speed
   - Reduce memory usage
   - Improve computational efficiency

4. **Feature Extensions**
   - Support more task types
   - Add more domain configurations
   - Enhance Planner head applications

## 🤝 Contributing

We welcome all forms of contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### How to Contribute

1. **Report Issues**: Submit Issues describing problems or suggestions
2. **Submit Code**: Fork the project, create a feature branch, submit Pull Request
3. **Improve Documentation**: Improve docs, add examples, fix errors
4. **Share Experience**: Share usage experience, optimization suggestions, application cases

### Contribution Directions

- ✅ Optimize overfitting issues
- ✅ Improve accuracy
- ✅ Optimize training speed
- ✅ Add new features
- ✅ Improve documentation
- ✅ Add tests

## 📖 Documentation

- **Architecture Design**: `docs/architecture/`
- **Usage Guides**: `docs/guides/`
- **Test Results**: `docs/results/training/`
- **Evaluation & Analysis**: `docs/evaluation/`
- **Full Index**: `docs/INDEX.md`

## 📝 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

Thanks to all researchers and developers who have contributed to this project.

Special thanks to:
- Related research in cognitive science
- Original designers of Transformer architecture
- All community members who provided feedback and suggestions

## 📧 Contact

- **Issues**: Submit Issues on GitHub
- **Pull Requests**: Pull Requests are welcome
- **Discussions**: Discuss in GitHub Discussions

---

## 🎯 Project Vision

Our goal is to advance the development of **constraint-enhanced AI models**, enabling AI to strictly adhere to constraint rules while maintaining high accuracy, thus playing a greater role in safety-critical applications.

**We believe**:
- The combination of cognitive science and deep learning is valuable
- Constraint compliance is crucial for safety-critical applications
- Open source can advance this field

**We invite**:
- **Researchers**: Verify, improve, and extend our methods
- **Developers**: Apply, optimize, and contribute code
- **Users**: Use, provide feedback, and share experiences

Let's advance constraint-enhanced AI models together!

---

**Project Status**: ✅ Core features complete, ready for open source  
**Last Updated**: November 15, 2025  
**Version**: v1.0.0

