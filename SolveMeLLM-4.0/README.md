# 4D-Transformer: Constraint-Enhanced Cognitive Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> 🧠 **4D认知架构**：将Self、Desire、Ethic、Reflection四个认知维度集成到Transformer中，实现约束增强的文本分类

**English**: [README_EN.md](README_EN.md)

---

## 🎯 项目意义

### 为什么需要4D-Transformer？

在安全关键应用中（医疗、金融、法律等），AI模型不仅需要高准确率，更需要**严格遵守约束规则**。传统的Transformer模型在约束遵守方面表现不佳，违反率往往较高。

**4D-Transformer**通过引入认知科学的四个维度，专门设计了**约束增强机制**，在保持较高准确率的同时，显著降低了约束违反率。

### 核心价值

1. **约束遵守能力**：违反率从0.65%降低到0.00-0.01%（降低98%+）✅
2. **认知架构创新**：首次将Self、Desire、Ethic、Reflection四个认知维度集成到Transformer
3. **领域自适应**：通过Domain Steering机制，让模型适应不同应用场景
4. **结果稳定可复现**：多随机种子测试验证，准确率差异仅0.14%

## 🚀 快速开始

### 安装依赖

```bash
pip install torch transformers datasets tqdm numpy
```

### 基本使用

```python
from train_medical_dataset import FourDTransformerClassifier
import torch

# 创建模型
model = FourDTransformerClassifier(
    vocab_size=30522,
    d_model=192,
    nhead=8,
    num_layers=4,
    dim_feedforward=768,
    dropout=0.5,
    num_classes=2,
    state_dim=64,
    default_domain='generic'  # 或 'medical', 'creative', 'finance'
)

# 切换领域配置
model.set_domain('medical')  # 切换到医疗领域配置

# 前向传播
input_ids = torch.randint(0, 30522, (32, 128))  # [batch_size, seq_len]
constraints = torch.zeros(32, 128)  # 约束掩码
logits = model(input_ids, constraints=constraints)
```

### 训练模型

```bash
# 使用IMDb数据集训练
python train_medical_dataset.py
```

## 🧠 4D认知架构

### 四个维度

1. **Self (S)**：自我认知
   - 提供稳定性和一致性
   - 维护模型的内在状态

2. **Desire (D)**：目标动机
   - 驱动探索和学习
   - 增强模型的表达能力

3. **Ethic (G)**：伦理约束 ⭐ **核心**
   - **专门处理约束遵守**
   - 显著降低违反率（从0.65%到0.00-0.01%）

4. **Reflection (R)**：反馈机制
   - 修正错误和调整
   - 提供自我纠错能力

### 领域自适应（Domain Steering）

通过Domain Steering机制，可以动态调整四个维度的权重，适应不同应用场景：

- **Generic**：平衡配置（S=1.0, D=1.0, G=1.0, R=1.0）
- **Medical**：强调约束（S=1.1, D=1.1, G=1.3, R=1.2）
- **Creative**：增强探索（S=0.9, D=1.5, G=0.8, R=0.9）
- **Finance**：最严格约束（S=1.3, D=0.8, G=1.7, R=1.6）

## 📊 实验结果

### 性能表现

| 配置 | 最佳验证准确率 | 违反率 | 训练-验证差距 |
|------|----------------|--------|---------------|
| Generic | 77.39% | 0.00-0.01% | 17.58% |
| Medical | 77.16% | 0.00% | 17.64% |
| Creative | 77.18% | 0.00% | 17.69% |
| Finance | 77.02% | 0.00% | 17.80% |

**与Baseline对比**：
- ✅ **违反率**：0.00-0.01% vs 0.65%（降低98%+）
- ⚠️ **准确率**：77.39% vs 77.90%（差异-0.51%，可接受的权衡）

### 稳定性验证

多随机种子测试（3个种子）：
- 准确率均值：77.39%
- 标准差：0.07%
- 范围：77.30% - 77.44%
- **结论**：结果非常稳定 ✅

## 📁 项目结构

```
SolveMeLLM-4.0/
├── models/                          # 模型实现
│   ├── four_d_transformer_block-v2.py  # 4D-Transformer核心实现
│   └── baseline_transformer.py         # Baseline Transformer
├── train_medical_dataset.py         # 主训练脚本
├── medical_constrained_classification.py  # 数据集处理
├── docs/                            # 文档
│   ├── architecture/                # 架构设计文档
│   ├── guides/                      # 使用指南
│   ├── results/training/            # 训练结果
│   └── evaluation/                  # 评估和分析
└── scripts/                         # 工具脚本
    ├── test_planner_head.py         # Planner头测试
    └── test_multi_seed_generic.py   # 多随机种子测试
```

## 🔬 研究背景

### 动机

传统的Transformer模型在约束遵守方面表现不佳，特别是在安全关键应用中（医疗、金融、法律等）。本项目探索将认知科学的维度集成到深度学习模型中，通过专门的约束处理机制来降低违反率。

### 核心贡献

1. **4D认知架构**：首次将Self、Desire、Ethic、Reflection四个认知维度集成到Transformer
2. **约束增强机制**：通过Ethic维度专门处理约束，显著降低违反率（98%+）
3. **领域自适应**：通过Domain Steering机制，让模型适应不同应用场景
4. **实验验证**：在IMDb数据集上验证了方法的有效性

## 💡 应用场景

### 适合的应用

1. **医疗领域**：需要严格约束，降低误诊风险
2. **金融领域**：需要遵守法规，降低违规风险
3. **法律领域**：需要遵守法律条文，降低违法风险
4. **安全关键系统**：需要严格遵守安全规则

### 核心优势

- ✅ **约束遵守**：违反率降低98%+
- ✅ **领域自适应**：可以根据场景调整模型行为
- ✅ **可解释性**：4D状态提供了模型决策的可解释性

## ⚠️ 已知问题与优化方向

### 当前问题

1. **过拟合**：训练-验证差距约17-18%，需要进一步优化
2. **准确率**：略低于Baseline（差异-0.51%），这是准确率和约束遵守的权衡
3. **训练时间**：比Baseline慢约3倍（1分钟 vs 18秒/epoch）

### 优化方向

我们欢迎社区贡献以下优化：

1. **过拟合优化**
   - 更早的Early Stopping策略
   - 数据增强技术
   - 更强的正则化方法

2. **准确率提升**
   - 优化约束损失权重
   - 改进领域Profile权重
   - 探索新的架构设计

3. **性能优化**
   - 优化训练速度
   - 减少内存占用
   - 改进计算效率

4. **功能扩展**
   - 支持更多任务类型
   - 添加更多领域配置
   - 完善Planner头的应用

## 🤝 贡献指南

我们欢迎所有形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 如何贡献

1. **报告问题**：提交Issue描述问题或建议
2. **提交代码**：Fork项目，创建功能分支，提交Pull Request
3. **完善文档**：改进文档、添加示例、修复错误
4. **分享经验**：分享使用经验、优化建议、应用案例

### 贡献方向

- ✅ 优化过拟合问题
- ✅ 提升准确率
- ✅ 优化训练速度
- ✅ 添加新功能
- ✅ 完善文档
- ✅ 添加测试

## 📖 文档

- **架构设计**：`docs/architecture/`
- **使用指南**：`docs/guides/`
- **测试结果**：`docs/results/training/`
- **评估分析**：`docs/evaluation/`
- **完整索引**：`docs/INDEX.md`

## 📝 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 🙏 致谢

感谢所有为这个项目做出贡献的研究者和开发者。

特别感谢：
- 认知科学领域的相关研究
- Transformer架构的原始设计者
- 所有提供反馈和建议的社区成员

## 📧 联系方式

- **Issues**：在GitHub上提交Issue
- **Pull Requests**：欢迎提交PR
- **讨论**：在GitHub Discussions中讨论

---

## 🎯 项目愿景

我们的目标是推动**约束增强的AI模型**的发展，让AI在保持高准确率的同时，能够严格遵守约束规则，从而在安全关键应用中发挥更大的作用。

**我们相信**：
- 认知科学与深度学习的结合是有价值的
- 约束遵守能力对安全关键应用至关重要
- 开源可以推动这一领域的发展

**我们邀请**：
- 研究者：验证、改进、扩展我们的方法
- 开发者：应用、优化、贡献代码
- 用户：使用、反馈、分享经验

让我们一起推动约束增强AI模型的发展！

---

**项目状态**：✅ 核心功能完成，准备开源  
**最后更新**：2025年11月15日  
**版本**：v1.0.0
