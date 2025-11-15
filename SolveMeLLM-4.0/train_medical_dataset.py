#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗领域数据集训练4D-Transformer
==================================

使用医疗文本分类数据集，添加医疗安全约束，训练和评估
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import os
import sys
import importlib.util
import copy

# 导入数据集和模型
# 医疗数据集在根目录，不在tasks目录
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)  # 添加脚本所在目录到路径
sys.path.append(os.path.join(script_dir, 'models'))

from medical_constrained_classification import (
    MedicalConstrainedTextClassificationDataset,
    collate_fn
)

# 使用v2版本（batch_first格式）
v2_file_path = os.path.join(script_dir, 'models', 'four_d_transformer_block-v2.py')
v2_file_path = os.path.normpath(v2_file_path)
spec = importlib.util.spec_from_file_location('four_d_transformer_block_v2', v2_file_path)
four_d_transformer_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(four_d_transformer_v2)
FourDTransformerBlock = four_d_transformer_v2.FourDTransformerBlock
FourDPlannerHead = four_d_transformer_v2.FourDPlannerHead
DEFAULT_DOMAIN_PROFILES = four_d_transformer_v2.DEFAULT_DOMAIN_PROFILES

from baseline_transformer import StandardTransformerBlock

# 配置
DATASET_NAME = 'imdb'  # 'synthetic', 'pubmed_qa', 或 'imdb'（真实数据集）
DOMAIN_CONFIG = 'generic'  # 'generic', 'medical', 'creative', 'finance' - 领域配置（先试generic提升准确率）
MAX_SAMPLES_TRAIN = None  # None表示使用全部数据（PubMedQA约1000个样本，80%用于训练）
MAX_SAMPLES_VAL = None     # None表示使用全部验证数据（20%用于验证）
MAX_LENGTH = 128
BATCH_SIZE = 16  # 减小batch size，因为数据量小
EPOCHS = 20  # 增加epochs，给模型更多学习机会
LR = 1e-4  # 降低学习率，防止过快过拟合
D_MODEL = 192  # 从256压缩到192（减少25%，减少过拟合）
NHEAD = 8  # 保持8（192能被8整除）
NUM_LAYERS = 4  # 从5压缩到4（减少20%，减少过拟合）
DIM_FEEDFORWARD = 768  # 从1024压缩到768（减少25%，减少过拟合）
DROPOUT = 0.5  # 进一步增加dropout，防止过拟合（从0.4提升到0.5）
BASELINE_DROPOUT = 0.5  # Baseline使用更强的正则化（从0.4提升到0.5）
LABEL_SMOOTHING = 0.1  # 添加label smoothing，进一步减少过拟合
STATE_DIM = 64  # 从96压缩到64（减少33%，减少过拟合）
CONSTRAINT_LOSS_WEIGHT = 1.2  # 进一步平衡准确率和约束遵守（从1.5降低到1.2）
USE_WARMUP = True
WARMUP_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 10  # 减少patience，更早停止

# Planner头配置（用于测试）
PLANNER_DIM = None
PLANNER_POOLING = 'mean'

# BERT tokenizer的vocab size
VOCAB_SIZE = 30522  # bert-base-uncased

# 随机种子（用于可复现性）
SEED = 456  # 可以改为123, 456等用于多随机种子测试
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")
print(f"[INFO] Random seed: {SEED}")


class FourDTransformerClassifier(nn.Module):
    """4D-Transformer分类器（适配医疗数据集，支持领域自适应）"""
    
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 dropout=DROPOUT, num_classes=2, state_dim=STATE_DIM,
                 max_length=MAX_LENGTH, domain_profiles=None, default_domain='generic',
                 planner_dim=None, planner_pooling='mean'):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_length = max_length
        self.domain_profiles = domain_profiles or DEFAULT_DOMAIN_PROFILES
        self.current_domain = default_domain
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_length, d_model))
        
        # 4D-Transformer Blocks（支持领域配置）
        self.blocks = nn.ModuleList([
            FourDTransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                state_dim=state_dim,
                domain_profiles=domain_profiles,
                default_domain=default_domain
            )
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        # ⭐ 可选的 Planner 头
        if planner_dim is not None:
            self.planner_head = FourDPlannerHead(
                state_dim=state_dim,
                plan_dim=planner_dim,
                pooling=planner_pooling,
            )
        else:
            self.planner_head = None
        
        self.state_dim = state_dim
        self.planner_dim = planner_dim
    
    def set_domain(self, domain_name: str):
        """切换领域配置"""
        if domain_name not in self.domain_profiles:
            raise ValueError(f"Unknown domain: {domain_name}")
        self.current_domain = domain_name
        for block in self.blocks:
            block.set_domain(domain_name)
    
    def forward(self, input_ids, attention_mask=None, constraints=None, return_plan=False):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            constraints: [batch_size, seq_len] 约束掩码
            return_plan: 是否返回plan向量
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        x = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        constraint_mask = constraints  # [batch_size, seq_len]
        
        # 通过4D-Transformer Blocks
        four_d_states = None
        for block in self.blocks:
            x, four_d_states = block(x, four_d_states, constraint_mask=constraint_mask)
        
        # 池化（使用CLS token）
        cls_representation = x[:, 0, :]  # [batch_size, d_model]
        
        # 分类
        logits = self.classifier(cls_representation)  # [batch_size, num_classes]
        
        # ⭐ 可选的 Planner 头
        plan = None
        if return_plan and (self.planner_head is not None):
            plan = self.planner_head(four_d_states)  # [batch_size, planner_dim]
        
        if return_plan:
            return logits, plan
        return logits


class StandardTransformerClassifier(nn.Module):
    """标准Transformer分类器（Baseline）"""
    
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 dropout=BASELINE_DROPOUT, num_classes=2, max_length=MAX_LENGTH):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_length, d_model))
        
        # Standard Transformer Blocks
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
    
    def forward(self, input_ids, attention_mask=None, constraints=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            constraints: 忽略（Baseline不使用约束）
        """
        batch_size, seq_len = input_ids.shape
        
        # Embedding
        x = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # 转置为标准Transformer格式 [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)
        
        # 通过Standard Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # 转回 [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)
        
        # 池化（使用CLS token）
        cls_representation = x[:, 0, :]  # [batch_size, d_model]
        
        # 分类
        logits = self.classifier(cls_representation)  # [batch_size, num_classes]
        
        return logits


def train_epoch(model, dataloader, optimizer, criterion, constraint_criterion, 
                constraint_weight, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_constraint_loss = 0.0
    correct = 0
    total = 0
    total_violations = 0
    
    # 减少刷新频率：只在epoch结束时显示，或每100个batch更新一次
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", 
                mininterval=2.0, maxinterval=10.0, ncols=100)
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        constraints = batch['constraints'].to(device)
        violations = batch['violations'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(input_ids, attention_mask=attention_mask, constraints=constraints)
        
        # 分类损失
        ce_loss = criterion(logits, labels)
        
        # 约束损失（改进：使用softmax概率，更平滑的梯度）
        probs = torch.softmax(logits, dim=-1)
        predicted_labels = logits.argmax(dim=-1)
        predicted_violations = (predicted_labels == 1) & (violations > 0.5)
        
        if predicted_violations.sum() > 0:
            # 惩罚：预测为异常类别的概率 * 违反程度
            constraint_loss = (probs[predicted_violations, 1] * violations[predicted_violations]).mean()
        else:
            constraint_loss = torch.tensor(0.0, device=device)
        
        # 总损失
        total_loss_batch = ce_loss + constraint_weight * constraint_loss
        
        # Backward
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 统计
        total_loss += total_loss_batch.item()
        total_ce_loss += ce_loss.item()
        total_constraint_loss += constraint_loss.item() if isinstance(constraint_loss, torch.Tensor) else constraint_loss
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.size(0)
        total_violations += predicted_violations.sum().item()
        
        # 减少更新频率：每100个batch或最后一个batch才更新
        if step % 100 == 0 or step == len(dataloader) - 1:
            pbar.set_postfix({
                'loss': f'{total_loss/(step+1):.4f}',
                'acc': f'{100*correct/total:.2f}%',
                'viol': f'{100*total_violations/total:.2f}%'
            }, refresh=True)
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_constraint_loss = total_constraint_loss / len(dataloader)
    accuracy = 100 * correct / total
    violation_rate = 100 * total_violations / total
    
    return avg_loss, avg_ce_loss, avg_constraint_loss, accuracy, violation_rate


def evaluate(model, dataloader, criterion, constraint_criterion, constraint_weight, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_constraint_loss = 0.0
    correct = 0
    total = 0
    total_violations = 0
    
    with torch.no_grad():
        # 评估时减少刷新
        pbar = tqdm(dataloader, desc="Evaluating", mininterval=1.0, ncols=80)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            constraints = batch['constraints'].to(device)
            violations = batch['violations'].to(device)
            
            # Forward
            logits = model(input_ids, attention_mask=attention_mask, constraints=constraints)
            
            # 分类损失
            ce_loss = criterion(logits, labels)
            
            # 约束损失（改进：使用softmax概率）
            probs = torch.softmax(logits, dim=-1)
            predicted_labels = logits.argmax(dim=-1)
            predicted_violations = (predicted_labels == 1) & (violations > 0.5)
            
            if predicted_violations.sum() > 0:
                constraint_loss = (probs[predicted_violations, 1] * violations[predicted_violations]).mean()
            else:
                constraint_loss = torch.tensor(0.0, device=device)
            
            # 总损失
            total_loss_batch = ce_loss + constraint_weight * constraint_loss
            
            # 统计
            total_loss += total_loss_batch.item()
            total_ce_loss += ce_loss.item()
            total_constraint_loss += constraint_loss.item() if isinstance(constraint_loss, torch.Tensor) else constraint_loss
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
            total_violations += predicted_violations.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_constraint_loss = total_constraint_loss / len(dataloader)
    accuracy = 100 * correct / total
    violation_rate = 100 * total_violations / total
    
    return avg_loss, avg_ce_loss, avg_constraint_loss, accuracy, violation_rate


def main():
    print("=" * 80)
    print("Medical Domain: 4D-Transformer vs Baseline Training")
    print("=" * 80)
    
    # 加载数据集
    print("\n[1/4] Loading datasets...")
    try:
        train_dataset = MedicalConstrainedTextClassificationDataset(
            dataset_name=DATASET_NAME,
            split='train',
            max_samples=MAX_SAMPLES_TRAIN,
            max_length=MAX_LENGTH,
            add_constraints=True,
            use_synthetic=(DATASET_NAME == 'synthetic')
        )
        
        val_dataset = MedicalConstrainedTextClassificationDataset(
            dataset_name=DATASET_NAME,
            split='validation' if DATASET_NAME == 'synthetic' else 'test',  # PubMedQA使用test作为validation
            max_samples=MAX_SAMPLES_VAL,
            max_length=MAX_LENGTH,
            add_constraints=True,
            use_synthetic=(DATASET_NAME == 'synthetic')
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
        )
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型
    print("\n[2/4] Creating models...")
    print(f"  Domain configuration: {DOMAIN_CONFIG}")
    print(f"  Domain profile: {DEFAULT_DOMAIN_PROFILES.get(DOMAIN_CONFIG, 'Not found')}")
    
    model_4d = FourDTransformerClassifier(
        domain_profiles=DEFAULT_DOMAIN_PROFILES,
        default_domain=DOMAIN_CONFIG,
        planner_dim=PLANNER_DIM,
        planner_pooling=PLANNER_POOLING
    ).to(device)
    model_baseline = StandardTransformerClassifier().to(device)
    
    # 确保使用正确的领域配置
    model_4d.set_domain(DOMAIN_CONFIG)
    print(f"  4D-Transformer domain set to: {model_4d.current_domain}")
    if PLANNER_DIM is not None:
        print(f"  Planner head enabled: dim={PLANNER_DIM}, pooling={PLANNER_POOLING}")
    else:
        print(f"  Planner head disabled")
    
    print(f"  4D-Transformer parameters: {sum(p.numel() for p in model_4d.parameters()):,}")
    print(f"  Baseline parameters: {sum(p.numel() for p in model_baseline.parameters()):,}")
    
    # 优化器和损失函数（增加weight decay防止过拟合）
    optimizer_4d = torch.optim.AdamW(model_4d.parameters(), lr=LR, weight_decay=1e-3)  # 从5e-4提升到1e-3
    optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=LR, weight_decay=1e-3)  # 从5e-4提升到1e-3
    
    # 使用label smoothing的CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    constraint_criterion = nn.BCEWithLogitsLoss()
    
    # 训练历史
    history_4d = {
        'train_loss': [], 'train_acc': [], 'train_viol': [],
        'val_loss': [], 'val_acc': [], 'val_viol': []
    }
    history_baseline = {
        'train_loss': [], 'train_acc': [], 'train_viol': [],
        'val_loss': [], 'val_acc': [], 'val_viol': []
    }
    
    # Early stopping
    best_val_acc_4d = 0.0
    best_val_viol_4d = 100.0  # 初始化为最差违反率
    best_val_acc_baseline = 0.0
    patience_counter_4d = 0
    patience_counter_baseline = 0
    best_state_4d = None
    best_state_baseline = None
    
    # 学习率调度器
    if USE_WARMUP:
        def lr_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / WARMUP_EPOCHS
            else:
                progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
                return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler_4d = torch.optim.lr_scheduler.LambdaLR(optimizer_4d, lr_lambda)
        scheduler_baseline = torch.optim.lr_scheduler.LambdaLR(optimizer_baseline, lr_lambda)
    else:
        scheduler_4d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_4d, T_max=EPOCHS)
        scheduler_baseline = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_baseline, T_max=EPOCHS)
    
    # 训练4D-Transformer
    print("\n[3/4] Training 4D-Transformer...")
    for epoch in range(EPOCHS):
        try:
            print(f"\n  Epoch {epoch+1}/{EPOCHS} [4D-Transformer]")
            train_loss, train_ce_loss, train_constraint_loss, train_acc, train_viol = train_epoch(
                model_4d, train_loader, optimizer_4d, criterion, constraint_criterion,
                CONSTRAINT_LOSS_WEIGHT, device, epoch
            )
            val_loss, val_ce_loss, val_constraint_loss, val_acc, val_viol = evaluate(
                model_4d, val_loader, criterion, constraint_criterion,
                CONSTRAINT_LOSS_WEIGHT, device
            )
        except Exception as e:
            print(f"\n[ERROR] 4D-Transformer training interrupted at epoch {epoch+1}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            break
        
        history_4d['train_loss'].append(train_loss)
        history_4d['train_acc'].append(train_acc)
        history_4d['train_viol'].append(train_viol)
        history_4d['val_loss'].append(val_loss)
        history_4d['val_acc'].append(val_acc)
        history_4d['val_viol'].append(val_viol)
        
        scheduler_4d.step()
        
        # Early stopping（基于综合指标：准确率 - 违反率惩罚）
        # 这样会优先选择准确率高且违反率低的模型
        VIOLATION_PENALTY = 2.0  # 违反率惩罚权重
        composite_score = val_acc - val_viol * VIOLATION_PENALTY
        best_composite = best_val_acc_4d - best_val_viol_4d * VIOLATION_PENALTY
        
        if composite_score > best_composite:
            best_val_acc_4d = val_acc
            best_val_viol_4d = val_viol
            patience_counter_4d = 0
            best_state_4d = copy.deepcopy(model_4d.state_dict())
        else:
            patience_counter_4d += 1
        
        print(f"    Train: loss={train_loss:.4f}, acc={train_acc:.2f}%, viol={train_viol:.2f}%")
        print(f"    Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%, viol={val_viol:.2f}% (composite={composite_score:.2f}, best={best_composite:.2f}, patience: {patience_counter_4d}/{EARLY_STOPPING_PATIENCE})")
        
        if patience_counter_4d >= EARLY_STOPPING_PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            model_4d.load_state_dict(best_state_4d)
            break
    
    # 训练Baseline
    print("\n[4/4] Training Baseline...")
    for epoch in range(EPOCHS):
        try:
            print(f"\n  Epoch {epoch+1}/{EPOCHS} [Baseline]")
            train_loss, train_ce_loss, train_constraint_loss, train_acc, train_viol = train_epoch(
                model_baseline, train_loader, optimizer_baseline, criterion, constraint_criterion,
                0.0, device, epoch  # Baseline不使用约束损失
            )
            val_loss, val_ce_loss, val_constraint_loss, val_acc, val_viol = evaluate(
                model_baseline, val_loader, criterion, constraint_criterion,
                0.0, device  # Baseline不使用约束损失
            )
        except Exception as e:
            print(f"\n[ERROR] Baseline training interrupted at epoch {epoch+1}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            break
        
        history_baseline['train_loss'].append(train_loss)
        history_baseline['train_acc'].append(train_acc)
        history_baseline['train_viol'].append(train_viol)
        history_baseline['val_loss'].append(val_loss)
        history_baseline['val_acc'].append(val_acc)
        history_baseline['val_viol'].append(val_viol)
        
        scheduler_baseline.step()
        
        # Early stopping
        if val_acc > best_val_acc_baseline:
            best_val_acc_baseline = val_acc
            patience_counter_baseline = 0
            best_state_baseline = copy.deepcopy(model_baseline.state_dict())
        else:
            patience_counter_baseline += 1
        
        print(f"    Train: loss={train_loss:.4f}, acc={train_acc:.2f}%, viol={train_viol:.2f}%")
        print(f"    Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%, viol={val_viol:.2f}% (best: {best_val_acc_baseline:.2f}%, patience: {patience_counter_baseline}/{EARLY_STOPPING_PATIENCE})")
        
        if patience_counter_baseline >= EARLY_STOPPING_PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            model_baseline.load_state_dict(best_state_baseline)
            break
    
    # 结果对比
    print("\n" + "=" * 80)
    print("Final Comparison - Medical Domain")
    print("=" * 80)
    print(f"{'Metric':<25} {'4D-Transformer':<20} {'Baseline':<20} {'Difference':<20}")
    print("-" * 80)
    
    # 最终epoch的结果
    final_epoch_4d = len(history_4d['val_acc']) - 1
    final_epoch_base = len(history_baseline['val_acc']) - 1
    
    # Val Accuracy
    acc_4d = history_4d['val_acc'][final_epoch_4d]
    acc_base = history_baseline['val_acc'][final_epoch_base]
    acc_diff = acc_4d - acc_base
    print(f"{'Val Accuracy (%)':<25} {acc_4d:<20.2f} {acc_base:<20.2f} {acc_diff:+.2f}")
    
    # Val Loss
    loss_4d = history_4d['val_loss'][final_epoch_4d]
    loss_base = history_baseline['val_loss'][final_epoch_base]
    loss_diff = loss_4d - loss_base
    print(f"{'Val Loss':<25} {loss_4d:<20.4f} {loss_base:<20.4f} {loss_diff:+.4f}")
    
    # Constraint Violation
    viol_4d = history_4d['val_viol'][final_epoch_4d]
    viol_base = history_baseline['val_viol'][final_epoch_base]
    viol_diff = viol_4d - viol_base
    print(f"{'Val Violation (%)':<25} {viol_4d:<20.2f} {viol_base:<20.2f} {viol_diff:+.2f}")
    
    # 最佳epoch的结果（基于综合指标）
    VIOLATION_PENALTY = 2.0
    composite_scores_4d = [acc - viol * VIOLATION_PENALTY for acc, viol in zip(history_4d['val_acc'], history_4d['val_viol'])]
    best_epoch_4d = np.argmax(composite_scores_4d)
    best_epoch_base = np.argmax(history_baseline['val_acc'])  # Baseline只看准确率
    
    print("\n" + "=" * 80)
    print("Best Epoch Results")
    print("=" * 80)
    print(f"{'Metric':<25} {'4D-Transformer':<20} {'Baseline':<20} {'Difference':<20}")
    print("-" * 80)
    
    # Best Val Accuracy
    best_acc_4d = history_4d['val_acc'][best_epoch_4d]
    best_acc_base = history_baseline['val_acc'][best_epoch_base]
    best_acc_diff = best_acc_4d - best_acc_base
    print(f"{'Best Val Acc (%)':<25} {best_acc_4d:<20.2f} {best_acc_base:<20.2f} {best_acc_diff:+.2f}")
    
    # Best Val Loss
    best_loss_4d = history_4d['val_loss'][best_epoch_4d]
    best_loss_base = history_baseline['val_loss'][best_epoch_base]
    best_loss_diff = best_loss_4d - best_loss_base
    print(f"{'Best Val Loss':<25} {best_loss_4d:<20.4f} {best_loss_base:<20.4f} {best_loss_diff:+.4f}")
    
    # Best Constraint Violation
    best_viol_4d = history_4d['val_viol'][best_epoch_4d]
    best_viol_base = history_baseline['val_viol'][best_epoch_base]
    best_viol_diff = best_viol_4d - best_viol_base
    print(f"{'Best Val Viol (%)':<25} {best_viol_4d:<20.2f} {best_viol_base:<20.2f} {best_viol_diff:+.2f}")
    
    print("\n" + "=" * 80)
    print("Summary - Medical Domain")
    print("=" * 80)
    
    if acc_4d > acc_base:
        print(f"✅ 4D-Transformer has higher accuracy (+{acc_diff:.2f}%)")
    else:
        print(f"❌ Baseline has higher accuracy ({acc_diff:.2f}%)")
    
    if loss_4d < loss_base:
        print(f"✅ 4D-Transformer has lower loss ({loss_diff:.4f})")
    else:
        print(f"❌ Baseline has lower loss ({loss_diff:+.4f})")
    
    if viol_4d < viol_base:
        print(f"✅ 4D-Transformer has lower constraint violation ({viol_diff:.2f}%)")
    else:
        print(f"❌ Baseline has lower constraint violation ({viol_diff:+.2f}%)")
    
    print("\n" + "=" * 80)
    print("Medical Domain Training Completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()