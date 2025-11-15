"""
标准Transformer Baseline
======================

用于对比4D-Transformer
"""

import torch
import torch.nn as nn
import math


class StandardTransformerBlock(nn.Module):
    """标准Transformer Block"""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
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
    
    def forward(self, x):
        """x: [seq_len, batch, d_model]"""
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        ff_out = self.feedforward(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class StandardTransformer(nn.Module):
    """标准Transformer模型"""
    
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, 
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        self.layers = nn.ModuleList([
            StandardTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_head = nn.Linear(d_model, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_head.bias.data.zero_()
        self.output_head.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, constraint_mask=None):
        """
        src: [batch, seq_len]
        constraint_mask: 忽略（标准Transformer不支持）
        返回: [batch, seq_len, vocab_size]
        """
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        
        logits = self.output_head(x)
        return logits


if __name__ == "__main__":
    model = StandardTransformer(vocab_size=1000, d_model=128, nhead=4, num_layers=2)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Standard Transformer parameters: {total_params:,}")

