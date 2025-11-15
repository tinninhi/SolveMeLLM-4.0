# Planner头使用指南

## 🎯 设计理念

**核心思想**：保持4D内核（S, D, G, R）不变，在外层添加Planner头，从4D状态中提取"规划调整"信息。

**关键区别**：
- **R (Reflection)**：面向**过去**（修正错误）
- **P (Planner)**：面向**未来**（规划调整）- 作为4D状态的readout

## 📊 架构设计

### 4D内核 + Planner头

```
输入 [B, T] 
  ↓
Embedding + Positional Encoding
  ↓
4D-Transformer Blocks (S, D, G, R)
  ↓
┌─────────────────┬──────────────────┐
│  Output Head    │  Planner Head    │
│  (logits)       │  (plan)          │
└─────────────────┴──────────────────┘
```

### Planner头的工作原理

1. **输入**：4D状态 (S, D, G, R)，每个 [B, T, state_dim]
2. **Pooling**：
   - `mean`：沿时间维求平均 → "全局规划"
   - `last`：取最后一个token的状态 → "当前局部决策"
3. **融合**：拼接S, D, G, R → [B, 4*state_dim]
4. **输出**：plan向量 [B, plan_dim]

## 💻 使用方法

### 1. 创建带Planner头的模型

```python
from models.four_d_transformer_block_v2 import FourDTransformer

model = FourDTransformer(
    vocab_size=tokenizer.vocab_size,
    d_model=192,
    nhead=8,
    num_layers=4,
    state_dim=64,
    planner_dim=128,         # ⭐ 启用 Planner
    planner_pooling="mean",  # 用整段平均状态做规划
    domain_profiles=DEFAULT_DOMAIN_PROFILES,
    default_domain="generic",
)
```

### 2. 基本使用（兼容旧代码）

```python
# 旧代码仍然可以正常工作
logits = model(src, constraint_mask=constraint_mask)
```

### 3. 获取4D状态和Plan

```python
# 获取logits、4D状态和plan
logits, four_d_states, plan = model(
    src,
    constraint_mask=constraint_mask,
    return_states=True,
    return_plan=True,
)

# four_d_states = (S, D, G, R)，每个 [B, T, state_dim]
# plan: [B, planner_dim]
```

### 4. 在训练中使用

```python
# 例如医疗任务
model.set_domain("medical")

logits, four_d_states, plan = model(
    input_ids,
    attention_mask=attention_mask,
    constraints=constraints,
    return_states=True,
    return_plan=True,
)

# 1. 使用logits做正常分类训练
ce_loss = criterion(logits, labels)

# 2. 使用plan做额外任务
# 例如：
# - 用plan预测当前样本的风险等级
# - 用plan预测是否有潜在违反
# - 用plan做RL policy-head等
```

## 🔍 观察和分析

### 1. Plan的范数变化

```python
plan_norm = plan.norm(dim=-1).mean().item()
print(f"Plan norm: {plan_norm:.4f}")
```

### 2. 不同领域profile下Plan的分布差异

```python
# 切换到不同领域
model.set_domain("medical")
plan_medical = model(..., return_plan=True)[2]

model.set_domain("creative")
plan_creative = model(..., return_plan=True)[2]

# 对比plan的分布差异
print(f"Medical plan mean: {plan_medical.mean().item():.4f}")
print(f"Creative plan mean: {plan_creative.mean().item():.4f}")
```

### 3. Plan与约束违反的关系

```python
# 分析plan与约束违反的关系
violations = compute_violations(predictions, constraints)
plan_norms = plan.norm(dim=-1)

# 看看高plan norm是否对应高违反率
correlation = torch.corrcoef(torch.stack([plan_norms, violations]))[0, 1]
print(f"Plan norm vs violations correlation: {correlation:.4f}")
```

## 🎯 应用场景

### 1. 风险预测

```python
# 用plan预测风险等级
risk_head = nn.Linear(planner_dim, 1)
risk_score = risk_head(plan)  # [B, 1]
```

### 2. 违反预测

```python
# 用plan预测是否有潜在违反
violation_head = nn.Linear(planner_dim, 1)
violation_prob = torch.sigmoid(violation_head(plan))  # [B, 1]
```

### 3. 任务级决策

```python
# 用plan做任务级决策
task_head = nn.Linear(planner_dim, num_tasks)
task_logits = task_head(plan)  # [B, num_tasks]
```

## 📝 注意事项

### 1. 向后兼容

- 不传`return_states`和`return_plan`时，行为与旧代码完全一致
- 现有训练脚本不需要修改

### 2. Planner头的可选性

- `planner_dim=None`时，不创建Planner头（节省参数）
- 需要时才启用Planner头

### 3. Pooling策略

- `mean`：适合全局规划任务
- `last`：适合当前局部决策任务

## 🚀 下一步

1. **观察Plan的行为**：
   - 在训练过程中记录plan的范数变化
   - 对比不同领域profile下plan的分布差异

2. **实验Plan的应用**：
   - 用plan做风险预测
   - 用plan做违反预测
   - 用plan做任务级决策

3. **评估效果**：
   - 如果plan有用，可以考虑进一步优化
   - 如果plan没用，可以移除（不影响4D内核）

---

**关键优势**：
- ✅ 不破坏现有4D实验体系
- ✅ 向后兼容，现有代码无需修改
- ✅ 可选的Planner头，需要时才启用
- ✅ 方便观察和分析plan的行为

