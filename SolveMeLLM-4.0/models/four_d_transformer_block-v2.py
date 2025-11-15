"""
4D-Transformer Block MVP (v2)
=============================

修订内容：
- 统一 batch_first = [B, T, C]
- 约束 mask 语义修正：无约束时不向 G 注入噪声特征
- 状态初始化更温和，避免过大偏置
- FourD 状态更新加入 LayerNorm 提升稳定性
- 设备/shape 管理更明确，减少 .to(device) 和 transpose 混乱
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # 现在没怎么用，可按需删

# 默认参数
D_MODEL = 512
NHEAD = 8
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
STATE_DIM = 128  # 每个维度 128 维

# ===== 4D 领域 profile：操作杆配置 =====
DEFAULT_DOMAIN_PROFILES = {
    # 通用 / 默认
    "generic": {
        "S": 1.0,  # Self / 稳定
        "D": 1.0,  # Desire / 探索
        "G": 1.0,  # Ethic / 规则
        "R": 1.0,  # Reflex / 纠错
    },

    # 医疗：更在意 G/R（规则 + 纠错），但平衡准确率
    "medical": {
        "S": 1.1,  # 从1.2降到1.1，稍微放松稳定性
        "D": 1.1,  # 从1.0提升到1.1，更积极的学习
        "G": 1.3,  # 从1.5降到1.3，进一步放松规则约束
        "R": 1.2,  # 从1.4降到1.2，进一步放松纠错
    },

    # 文案 / 创作：增强 D，放松 G 一点
    "creative": {
        "S": 0.9,
        "D": 1.5,
        "G": 0.8,
        "R": 0.9,
    },

    # 金融 / 风控：G/R 拉高，S 也偏高，D 稍微压低
    "finance": {
        "S": 1.3,
        "D": 0.8,
        "G": 1.7,
        "R": 1.6,
    },
}


class FourDStateManager(nn.Module):
    """管理 token-level 的 4D 状态，batch_first = [B, T, C]"""

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        scale = 0.02
        self.S0 = nn.Parameter(torch.randn(state_dim) * scale)
        self.D0 = nn.Parameter(torch.randn(state_dim) * scale)
        self.G0 = nn.Parameter(torch.randn(state_dim) * scale)
        self.R0 = nn.Parameter(torch.randn(state_dim) * scale)

        self.state_updater = FourDStateUpdater(d_model, state_dim)

    def init_states(self, batch_size: int, seq_len: int, device):
        """初始化 4D 状态: [B, T, C]"""

        def expand(v: torch.Tensor) -> torch.Tensor:
            base = v.to(device).view(1, 1, -1)  # [1,1,C]
            return base.expand(batch_size, seq_len, -1).contiguous()

        S = expand(self.S0)
        D = expand(self.D0)
        G = expand(self.G0)
        R = expand(self.R0)
        return (S, D, G, R)

    def update(self, x, four_d_states, constraint_mask=None):
        """包装一下 updater.forward 方便调用"""
        S, D, G, R = four_d_states
        return self.state_updater(x, S, D, G, R, constraint_mask=constraint_mask)


class FourDStateUpdater(nn.Module):
    """4D 状态更新器，batch_first = [B, T, *]"""

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # 共享编码
        shared_dim = d_model // 2
        self.shared_enc = nn.Sequential(
            nn.Linear(d_model + state_dim, shared_dim),
            nn.Tanh(),
        )

        # 约束编码（Ethic 维度用）
        constraint_dim = 32
        self.constraint_enc = nn.Sequential(
            nn.Linear(d_model, constraint_dim),
            nn.ReLU(),
        )
        self.shared_enc_G = nn.Sequential(
            nn.Linear(d_model + state_dim + constraint_dim, shared_dim),
            nn.Tanh(),
        )

        # 每个维度的编码器
        self.enc_S = nn.Linear(shared_dim, state_dim)
        self.enc_D = nn.Linear(shared_dim, state_dim)
        self.enc_G = nn.Linear(shared_dim, state_dim)
        self.enc_R = nn.Linear(shared_dim, state_dim)

        # 门控机制
        gate_hid = 16
        self.gate_base = nn.Linear(state_dim, gate_hid)
        self.cand_base = nn.Linear(state_dim, gate_hid)
        self.gate_proj = nn.Linear(gate_hid, state_dim)
        self.cand_proj = nn.Linear(gate_hid, state_dim)

        # 缩放因子（标量）
        self.gate_scale = nn.Parameter(torch.ones(1) * 0.5)
        self.cand_scale = nn.Parameter(torch.ones(1) * 0.5)

        # 稍微稳一点
        self.norm_S = nn.LayerNorm(state_dim)
        self.norm_D = nn.LayerNorm(state_dim)
        self.norm_G = nn.LayerNorm(state_dim)
        self.norm_R = nn.LayerNorm(state_dim)

    def _upd(self, enc: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        """门控更新：enc/prev: [B, T, C]"""
        h_gate = self.gate_base(enc)
        h_cand = self.cand_base(enc)
        gate = torch.sigmoid(self.gate_proj(h_gate) * self.gate_scale)
        cand = torch.tanh(self.cand_proj(h_cand) * self.cand_scale)
        new = gate * cand + (1.0 - gate) * prev
        return new

    def forward(self, x, S, D, G, R, constraint_mask=None):
        """
        x: [B, T, d_model]
        S, D, G, R: [B, T, state_dim]
        constraint_mask: [B, T]，bool 或 0/1
        """
        # shared encoding
        shared_S = self.shared_enc(torch.cat([x, S], dim=-1))
        shared_D = self.shared_enc(torch.cat([x, D], dim=-1))
        shared_R = self.shared_enc(torch.cat([x, R], dim=-1))

        # Ethic 维度：有约束时注入特征，否则为 0
        raw_constraint = self.constraint_enc(x)  # [B,T,constraint_dim]
        if constraint_mask is not None:
            cm = constraint_mask.to(dtype=raw_constraint.dtype).unsqueeze(-1)  # [B,T,1]
            constraint_feat = raw_constraint * cm
        else:
            constraint_feat = torch.zeros_like(raw_constraint)

        shared_G = self.shared_enc_G(torch.cat([x, G, constraint_feat], dim=-1))

        S_enc = self.enc_S(shared_S)
        D_enc = self.enc_D(shared_D)
        G_enc = self.enc_G(shared_G)
        R_enc = self.enc_R(shared_R)

        S_new = self.norm_S(self._upd(S_enc, S))
        D_new = self.norm_D(self._upd(D_enc, D))
        G_new = self.norm_G(self._upd(G_enc, G))
        R_new = self.norm_R(self._upd(R_enc, R))

        return S_new, D_new, G_new, R_new


class FourDSteering(nn.Module):
    """
    4D 领域操作杆：
    根据 current_domain 里的 (S,D,G,R) 系数，缩放四个标量权重。
    """

    def __init__(self, domain_profiles=None, default_domain: str = "generic"):
        super().__init__()
        self.domain_profiles = domain_profiles or DEFAULT_DOMAIN_PROFILES
        if default_domain not in self.domain_profiles:
            raise ValueError(f"Unknown default_domain: {default_domain}")
        self.current_domain = default_domain

    def set_domain(self, domain_name: str):
        if domain_name not in self.domain_profiles:
            raise ValueError(f"Unknown domain: {domain_name}")
        self.current_domain = domain_name

    def scale_weights(self, wS: torch.Tensor, wD: torch.Tensor,
                      wG: torch.Tensor, wR: torch.Tensor):
        """
        输入：四个标量参数（nn.Parameter）
        输出：按领域 profile 缩放后的四个标量
        """
        profile = self.domain_profiles[self.current_domain]
        s = profile["S"]
        d = profile["D"]
        g = profile["G"]
        r = profile["R"]
        return wS * s, wD * d, wG * g, wR * r


class FourDBiasGenerator(nn.Module):
    """把 4D 状态融合成一个 token-level bias（支持领域操作杆）"""

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        domain_profiles=None,
        default_domain: str = "generic",
    ):
        super().__init__()
        self.proj_S = nn.Linear(state_dim, d_model)
        self.proj_D = nn.Linear(state_dim, d_model)
        self.proj_G = nn.Linear(state_dim, d_model)
        self.proj_R = nn.Linear(state_dim, d_model)

        # 初值大概表示"轻微偏置"，后续可学习
        self.weight_S = nn.Parameter(torch.tensor(0.3))
        self.weight_D = nn.Parameter(torch.tensor(0.2))
        self.weight_G = nn.Parameter(torch.tensor(0.3))
        self.weight_R = nn.Parameter(torch.tensor(0.2))

        # 🔥 领域操作杆
        self.steering = FourDSteering(
            domain_profiles=domain_profiles,
            default_domain=default_domain,
        )

    def set_domain(self, domain_name: str):
        """外部调用，用于切换领域（medical / creative 等）"""
        self.steering.set_domain(domain_name)

    def forward(self, four_d_states):
        """
        four_d_states: (S, D, G, R)，each [B, T, state_dim]
        return: [B, T, d_model]
        """
        S, D, G, R = four_d_states
        S_proj = self.proj_S(S)
        D_proj = self.proj_D(D)
        G_proj = self.proj_G(G)
        R_proj = self.proj_R(R)

        # 🔥 按领域缩放权重
        wS, wD, wG, wR = self.steering.scale_weights(
            self.weight_S, self.weight_D, self.weight_G, self.weight_R
        )

        bias = wS * S_proj + wD * D_proj + wG * G_proj + wR * R_proj
        return bias


class FourDPlannerHead(nn.Module):
    """
    4D → Path/Planner 头
    - 输入: (S,D,G,R) 四个状态，[B, T, state_dim]
    - 输出: plan 向量，[B, plan_dim]，表示"规划调整"摘要
      （比如可以接一个分类头、RL policy、或者你自定义的 action 模块）
    """

    def __init__(self, state_dim: int, plan_dim: int = 128, pooling: str = "mean"):
        super().__init__()
        assert pooling in ["mean", "last"], "pooling must be 'mean' or 'last'"
        self.pooling = pooling

        in_dim = state_dim * 4  # 拼接 S,D,G,R
        hidden = state_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, plan_dim),
        )

    def forward(self, four_d_states):
        """
        four_d_states: (S,D,G,R)，each [B, T, state_dim]
        return: plan [B, plan_dim]
        """
        S, D, G, R = four_d_states  # [B,T,C]

        if self.pooling == "mean":
            # 沿时间维求平均 → "全局规划"
            S_p = S.mean(dim=1)
            D_p = D.mean(dim=1)
            G_p = G.mean(dim=1)
            R_p = R.mean(dim=1)
        else:  # "last"
            # 取最后一个 token 的状态 → "当前局部决策"
            S_p = S[:, -1, :]
            D_p = D[:, -1, :]
            G_p = G[:, -1, :]
            R_p = R[:, -1, :]

        fused = torch.cat([S_p, D_p, G_p, R_p], dim=-1)  # [B, 4*state_dim]
        plan = self.mlp(fused)  # [B, plan_dim]
        return plan


class FourDTransformerBlock(nn.Module):
    """
    4D-Transformer Block（batch_first 版本）

    x: [B, T, d_model]
    """

    def __init__(
        self,
        d_model=D_MODEL,
        nhead=NHEAD,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        state_dim=STATE_DIM,
        domain_profiles=None,
        default_domain: str = "generic",
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.state_dim = state_dim

        # 标准 Transformer 组件（batch_first=True）
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 4D 状态管理
        self.four_d_state_manager = FourDStateManager(d_model, state_dim)
        self.four_d_bias = FourDBiasGenerator(
            d_model,
            state_dim,
            domain_profiles=domain_profiles,
            default_domain=default_domain,
        )

    def set_domain(self, domain_name: str):
        """把领域切换传给 bias 模块"""
        self.four_d_bias.set_domain(domain_name)

    def forward(self, x, four_d_states=None, constraint_mask=None):
        """
        x: [B, T, d_model]
        four_d_states: (S, D, G, R) each [B, T, state_dim] 或 None
        constraint_mask: [B, T] 可选
        """
        B, T, _ = x.shape

        if four_d_states is None:
            four_d_states = self.four_d_state_manager.init_states(
                B, T, device=x.device
            )

        four_d_states = self.four_d_state_manager.update(
            x, four_d_states, constraint_mask=constraint_mask
        )

        bias = self.four_d_bias(four_d_states)  # [B,T,d_model]
        x_bias = x + bias

        # Self-Attention（标准自注意力）
        attn_out, _ = self.self_attn(x_bias, x_bias, x_bias)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feedforward
        ff_out = self.feedforward(x + bias)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)

        return x, four_d_states


class PositionalEncoding(nn.Module):
    """正弦位置编码，接口为 batch_first=[B, T, C]"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len,1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)  # [max_len,1,C]
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # buffer 不算到 parameters 里
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, C]
        """
        T = x.size(1)
        x = x + self.pe[:T].transpose(0, 1)  # [1,T,C] 广播到 [B,T,C]
        return self.dropout(x)


class FourDTransformer(nn.Module):
    """
    完整的 4D-Transformer 模型（MVP 版本，batch_first）

    用于验证 4D-Transformer 的可行性
    """

    def __init__(
        self,
        vocab_size: int,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers: int = 6,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        state_dim=STATE_DIM,
        domain_profiles=None,
        default_domain: str = "generic",
        planner_dim: int | None = None,   # ⭐ 新增：是否启用 Planner 头
        planner_pooling: str = "mean",    # "mean" 或 "last"
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.domain_profiles = domain_profiles or DEFAULT_DOMAIN_PROFILES
        self.current_domain = default_domain

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # 4D-Transformer Blocks
        self.layers = nn.ModuleList(
            [
                FourDTransformerBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    state_dim,
                    domain_profiles=self.domain_profiles,
                    default_domain=default_domain,
                )
                for _ in range(num_layers)
            ]
        )

        # Output head
        self.output_head = nn.Linear(d_model, vocab_size)

        # ⭐ 可选的 Planner 头
        if planner_dim is not None:
            self.planner_head = FourDPlannerHead(
                state_dim=state_dim,
                plan_dim=planner_dim,
                pooling=planner_pooling,
            )
        else:
            self.planner_head = None

        self.init_weights()

    def set_domain(self, domain_name: str):
        """
        外部接口：一键切换整个模型的领域 profile
        """
        if domain_name not in self.domain_profiles:
            raise ValueError(f"Unknown domain: {domain_name}")
        self.current_domain = domain_name
        for layer in self.layers:
            layer.set_domain(domain_name)

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_head.bias.data.zero_()
        self.output_head.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        src,
        constraint_mask=None,
        return_states: bool = False,
        return_plan: bool = False,
    ):
        """
        src: [B, T] token indices
        constraint_mask: [B, T] 可选
        return_states:
            - False: 只返回 logits（兼容旧代码）
            - True: 同时返回四维状态 (S,D,G,R)
        return_plan:
            - True 并且 model 有 planner_head 时，返回 plan 向量
        -------
        返回：
          - 默认: logits
          - return_states=True / return_plan=True:
                (logits, four_d_states, plan)
                其中 plan 可能为 None（如果没启用 planner_head）
        """
        x = self.embedding(src) * math.sqrt(self.d_model)  # [B,T,C]
        x = self.pos_encoding(x)

        four_d_states = None
        for layer in self.layers:
            x, four_d_states = layer(x, four_d_states, constraint_mask)

        logits = self.output_head(x)  # [B,T,V]

        plan = None
        if return_plan and (self.planner_head is not None):
            plan = self.planner_head(four_d_states)  # [B, planner_dim]

        if return_states or return_plan:
            return logits, four_d_states, plan

        return logits


if __name__ == "__main__":
    # 简单自测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试1: 不带Planner头（兼容旧代码）
    print("=" * 60)
    print("Test 1: Without Planner Head (backward compatible)")
    print("=" * 60)
    model1 = FourDTransformer(vocab_size=10000, d_model=512, nhead=8, num_layers=6)
    model1.to(device)
    
    total_params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f"[INFO] Total trainable parameters: {total_params1:,}")
    
    batch_size = 2
    seq_len = 10
    src = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    constraint_mask = torch.randint(0, 2, (batch_size, seq_len), device=device).bool()
    
    with torch.no_grad():
        logits1 = model1(src, constraint_mask)
    
    print(f"[INFO] Input shape: {src.shape}")
    print(f"[INFO] Output shape: {logits1.shape}")
    print("[OK] 4D-Transformer without Planner Head works!")
    
    # 测试2: 带Planner头
    print("\n" + "=" * 60)
    print("Test 2: With Planner Head")
    print("=" * 60)
    model2 = FourDTransformer(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_layers=6,
        state_dim=128,
        planner_dim=128,
        planner_pooling="mean",
    )
    model2.to(device)
    
    total_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"[INFO] Total trainable parameters: {total_params2:,}")
    print(f"[INFO] Additional parameters from Planner Head: {total_params2 - total_params1:,}")
    
    with torch.no_grad():
        # 测试默认行为（兼容旧代码）
        logits2 = model2(src, constraint_mask)
        print(f"[INFO] Default output shape: {logits2.shape}")
        
        # 测试返回states和plan
        logits2_full, four_d_states, plan = model2(
            src,
            constraint_mask=constraint_mask,
            return_states=True,
            return_plan=True,
        )
        
        S, D, G, R = four_d_states
        print(f"[INFO] Logits shape: {logits2_full.shape}")
        print(f"[INFO] S shape: {S.shape}")
        print(f"[INFO] D shape: {D.shape}")
        print(f"[INFO] G shape: {G.shape}")
        print(f"[INFO] R shape: {R.shape}")
        print(f"[INFO] Plan shape: {plan.shape}")
        print(f"[INFO] Plan norm: {plan.norm(dim=-1).mean().item():.4f}")
    
    print("[OK] 4D-Transformer with Planner Head works!")
    
    # 测试3: 不同pooling策略
    print("\n" + "=" * 60)
    print("Test 3: Different Pooling Strategies")
    print("=" * 60)
    
    model3_mean = FourDTransformer(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_layers=6,
        state_dim=128,
        planner_dim=128,
        planner_pooling="mean",
    )
    model3_mean.to(device)
    
    model3_last = FourDTransformer(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_layers=6,
        state_dim=128,
        planner_dim=128,
        planner_pooling="last",
    )
    model3_last.to(device)
    
    with torch.no_grad():
        _, _, plan_mean = model3_mean(src, constraint_mask=constraint_mask, return_plan=True)
        _, _, plan_last = model3_last(src, constraint_mask=constraint_mask, return_plan=True)
        
        print(f"[INFO] Plan (mean pooling) norm: {plan_mean.norm(dim=-1).mean().item():.4f}")
        print(f"[INFO] Plan (last pooling) norm: {plan_last.norm(dim=-1).mean().item():.4f}")
        print(f"[INFO] Plan difference: {(plan_mean - plan_last).abs().mean().item():.4f}")
    
    print("[OK] Different pooling strategies work!")
    print("\n" + "=" * 60)
    print("All tests passed! [OK]")
    print("=" * 60)