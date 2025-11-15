# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-15

### Added
- 4D认知架构集成（Self、Desire、Ethic、Reflection）
- 约束增强机制（通过Ethic维度处理约束）
- 领域自适应机制（Domain Steering）
- Planner头实现（从4D状态提取规划信息）
- 多随机种子测试验证
- 领域Profile测试（Generic、Medical、Creative、Finance）

### Performance
- 违反率：0.00-0.01%（明显优于Baseline的0.65%）
- 准确率：77.39%（略低于Baseline的77.90%，差异-0.51%）
- 稳定性：多随机种子测试验证，准确率差异仅0.14%

### Documentation
- 完善的README.md
- 架构设计文档
- 使用指南
- 测试结果文档
- 开源准备文档

### Fixed
- 过拟合问题（通过正则化、参数压缩、标签平滑）
- 参数压缩问题（修复参数传递）
- 数据集加载问题（修复IMDb数据集加载）

### Changed
- 从5D架构简化为4D架构（移除Path维度）
- Path维度重新设计为外部Planner头
- 优化领域Profile权重
- 增强正则化（Dropout 0.5, Weight Decay 1e-3, Label Smoothing 0.1）

### Removed
- 过时的训练脚本
- 过时的模型文件
- 临时文件和缓存
- 过时文档（已归档）

---

[1.0.0]: https://github.com/yourusername/SolveMeLLM-4.0/releases/tag/v1.0.0

