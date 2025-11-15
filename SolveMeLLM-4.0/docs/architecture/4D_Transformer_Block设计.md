# 4D-Transformer Block 设计

## 目标

将4D认知架构融合到Transformer，实现token-level的4D状态管理。

---

## 设计思路

### 核心概念

1. **Token-level 4D状态**：
   - 每个token维护自己的4D状态（S, D, G, R）
   - 状态在序列中传播和更新

2. **4D-aware Self-Attention**：
   - Self-attention考虑4D状态
   - 不同维度有不同的attention模式

3. **Constraint-friendly Representation**：
   - Ethic维度确保constraint-aware representation
   - Reflection维度学习历史错误

---

## 架构设计

### 方案1：4D状态作为额外输入

```python
class FourDTransformerBlock(nn.Module):
    """
    4D-Transformer Block
    
    每个token维护4D状态：
    - Self (S): 自我认知
    - Desire (D): 目标/欲望
    - Ethic (G): 伦理/约束
    - Reflection (R): 反思/学习
    """
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 标准Transformer组件
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 4D状态管理
        self.state_dim = d_model // 4  # 每个维度占1/4
        self.four_d_state_manager = FourDStateManager(d_model, self.state_dim)
        
        # 4D-aware attention
        self.four_d_attention = FourDAwareAttention(d_model, nhead)
    
    def forward(self, x, four_d_states=None, constraint_mask=None):
        """
        x: [seq_len, batch, d_model]
        four_d_states: (S, D, G, R) each [seq_len, batch, state_dim]
        constraint_mask: [seq_len, batch] 约束掩码
        """
        # 1. 更新4D状态
        if four_d_states is None:
            four_d_states = self.four_d_state_manager.init_states(x.size(1), x.size(0))
        
        four_d_states = self.four_d_state_manager.update(x, four_d_states, constraint_mask)
        
        # 2. 4D-aware self-attention
        # 将4D状态融入attention
        x_enhanced = self.four_d_attention(x, four_d_states)
        
        # 3. 标准self-attention
        attn_out, _ = self.self_attn(x_enhanced, x_enhanced, x_enhanced)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # 4. Feedforward
        ff_out = self.feedforward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x, four_d_states
```

---

### 方案2：4D状态作为bias

```python
class FourDTransformerBlockV2(nn.Module):
    """
    4D-Transformer Block (4D状态作为bias)
    
    将4D状态作为attention和feedforward的bias
    """
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 标准Transformer组件
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 4D状态管理
        self.state_dim = d_model // 4
        self.four_d_state_manager = FourDStateManager(d_model, self.state_dim)
        
        # 4D bias生成器
        self.four_d_bias = FourDBiasGenerator(d_model, self.state_dim)
    
    def forward(self, x, four_d_states=None, constraint_mask=None):
        """
        x: [seq_len, batch, d_model]
        four_d_states: (S, D, G, R) each [seq_len, batch, state_dim]
        constraint_mask: [seq_len, batch] 约束掩码
        """
        # 1. 更新4D状态
        if four_d_states is None:
            four_d_states = self.four_d_state_manager.init_states(x.size(1), x.size(0))
        
        four_d_states = self.four_d_state_manager.update(x, four_d_states, constraint_mask)
        
        # 2. 生成4D bias
        bias = self.four_d_bias(four_d_states, constraint_mask)
        
        # 3. Self-attention with 4D bias
        x_bias = x + bias
        attn_out, _ = self.self_attn(x_bias, x_bias, x_bias)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # 4. Feedforward with 4D bias
        ff_out = self.feedforward(x + bias)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x, four_d_states
```

---

## 关键组件

### 1. FourDStateManager

```python
class FourDStateManager(nn.Module):
    """管理token-level的4D状态"""
    
    def __init__(self, d_model, state_dim):
        super().__init__()
        self.state_dim = state_dim
        
        # 4D状态初始化
        self.S0 = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.D0 = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.G0 = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.R0 = nn.Parameter(torch.randn(state_dim) * 0.1)
        
        # 状态更新器（类似小模型的gate机制）
        self.state_updater = FourDStateUpdater(d_model, state_dim)
    
    def init_states(self, batch_size, seq_len):
        """初始化4D状态"""
        S = self.S0.unsqueeze(0).unsqueeze(0).expand(seq_len, batch_size, -1)
        D = self.D0.unsqueeze(0).unsqueeze(0).expand(seq_len, batch_size, -1)
        G = self.G0.unsqueeze(0).unsqueeze(0).expand(seq_len, batch_size, -1)
        R = self.R0.unsqueeze(0).unsqueeze(0).expand(seq_len, batch_size, -1)
        return (S, D, G, R)
    
    def update(self, x, four_d_states, constraint_mask=None):
        """更新4D状态"""
        S, D, G, R = four_d_states
        S_new, D_new, G_new, R_new = self.state_updater(x, S, D, G, R, constraint_mask)
        return (S_new, D_new, G_new, R_new)
```

---

### 2. FourDAwareAttention

```python
class FourDAwareAttention(nn.Module):
    """4D-aware attention机制"""
    
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # 4D状态投影
        self.proj_S = nn.Linear(state_dim, d_model)
        self.proj_D = nn.Linear(state_dim, d_model)
        self.proj_G = nn.Linear(state_dim, d_model)
        self.proj_R = nn.Linear(state_dim, d_model)
        
        # 融合权重
        self.weight_S = nn.Parameter(torch.ones(1))
        self.weight_D = nn.Parameter(torch.ones(1))
        self.weight_G = nn.Parameter(torch.ones(1))
        self.weight_R = nn.Parameter(torch.ones(1))
    
    def forward(self, x, four_d_states):
        """
        x: [seq_len, batch, d_model]
        four_d_states: (S, D, G, R)
        """
        S, D, G, R = four_d_states
        
        # 投影4D状态
        S_proj = self.proj_S(S)
        D_proj = self.proj_D(D)
        G_proj = self.proj_G(G)
        R_proj = self.proj_R(R)
        
        # 融合
        four_d_enhanced = (
            self.weight_S * S_proj +
            self.weight_D * D_proj +
            self.weight_G * G_proj +
            self.weight_R * R_proj
        )
        
        # 增强x
        x_enhanced = x + four_d_enhanced
        
        return x_enhanced
```

---

## 实现优先级

### 第1阶段：最小可行版本（MVP）

**目标**：验证4D-Transformer的可行性

**实现**：
1. 基本的4D状态管理
2. 简单的4D-aware attention
3. 在小规模模型上测试

**时间**：1-2周

---

### 第2阶段：完整实现

**目标**：完整的4D-Transformer Block

**实现**：
1. 完整的4D状态管理
2. 4D-aware attention机制
3. Constraint-friendly representation
4. 在中等规模模型上测试

**时间**：2-4周

---

### 第3阶段：优化和扩展

**目标**：优化性能，扩展到更大规模

**实现**：
1. 优化4D状态更新机制
2. 优化attention机制
3. 扩展到7B/13B模型
4. 跨任务泛化测试

**时间**：1-2月

---

## 下一步行动

### 立即开始

1. **设计4D-Transformer Block MVP**
   - 基本的4D状态管理
   - 简单的4D-aware attention
   - 在小规模模型上测试

2. **实现最小可行版本**
   - 创建4D-Transformer Block
   - 集成到标准Transformer
   - 测试基本功能

3. **验证可行性**
   - 在小规模模型上测试
   - 验证4D状态是否正确更新
   - 验证性能是否提升

---

**记住**：这是真正能创新的地方！开始设计4D-Transformer Block吧！

