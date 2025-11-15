#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗领域约束文本分类任务
========================

使用医疗相关的文本分类数据集，添加医疗安全约束规则

约束设计（目前都是基于“文本内容 + 标注标签”的静态约束）：
- 某些症状不应该预测为某些疾病（安全约束）
- 某些药物组合不应该同时出现（药物相互作用约束）
- 某些年龄组不应该出现某些疾病（合理性约束）

标签约定：
- 0: 正常 / 轻微 / 无明显风险
- 1: 异常 / 疾病 / 高风险
"""

import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset

# ====================
# 配置
# ====================

MAX_LENGTH = 128          # 最大序列长度
BATCH_SIZE = 32
NUM_CLASSES = 2           # 二分类：正常(0) vs 异常/疾病(1)
TOKENIZER_NAME = "bert-base-uncased"


# ====================
# 医疗约束定义
# ====================

# 1. 轻微症状词：不应该预测为严重疾病
MILD_SYMPTOMS = [
    "headache", "cough", "sneeze", "runny nose", "mild pain", "slight fever",
    "tired", "fatigue", "minor discomfort", "slight headache", "mild cough",
    "stuffy nose", "sore throat", "mild soreness", "slight ache"
]

# 2. 严重疾病词：不应该出现在轻微症状的文本中
SEVERE_DISEASES = [
    "cancer", "tumor", "malignant", "metastasis", "stroke", "heart attack",
    "seizure", "coma", "critical", "emergency", "life-threatening", "fatal",
    "terminal", "advanced disease", "severe infection", "organ failure"
]

# 3. 药物相互作用词：某些药物不应该同时出现
DRUG_INTERACTION_PAIRS = [
    ("aspirin", "warfarin"),   # 抗凝药物相互作用
    ("ibuprofen", "aspirin"),  # NSAIDs 相互作用
    ("alcohol", "sedative"),   # 酒精与镇静剂
]

# 4. 年龄相关约束词
PEDIATRIC_TERMS = [
    "child", "infant", "baby", "toddler", "pediatric", "young"
]
ADULT_ONLY_DISEASES = [
    "osteoporosis", "menopause", "prostate", "elderly condition"
]


# ====================
# 数据集类
# ====================

class MedicalConstrainedTextClassificationDataset(torch.utils.data.Dataset):
    """医疗领域约束文本分类数据集"""

    def __init__(
        self,
        dataset_name: str = "medical_qa",
        split: str = "train",
        max_samples: int | None = None,
        max_length: int = MAX_LENGTH,
        add_constraints: bool = True,
        use_synthetic: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            dataset_name: 数据集名称
                - 'medical_qa': 医疗问答数据集（如果可用）
                - 'synthetic': 使用合成的医疗文本数据
            split: 数据集分割
            max_samples: 最大样本数（会做简单的类别平衡）
            max_length: 最大序列长度
            add_constraints: 是否添加约束（会生成 constraint_mask 与 violation 标记）
            use_synthetic: 如果真实数据集不可用，使用合成数据
            seed: 随机种子，保证可复现
        """
        self.max_length = max_length
        self.add_constraints = add_constraints
        self.rng = random.Random(seed)

        # -------------------------
        # 1. 加载原始数据
        # -------------------------
        print(f"[INFO] Loading medical dataset: {dataset_name} ({split})...")

        if use_synthetic or dataset_name == "synthetic":
            dataset = self._create_synthetic_medical_data(max_samples or 5000)
        else:
            try:
                # 尝试加载真实医疗数据集
                if dataset_name == "pubmed_qa":
                    # PubMedQA: 医疗问答数据集
                    # 需要先安装: pip install datasets
                    # PubMedQA的pqa_labeled只有train split，需要手动分割
                    try:
                        full_dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
                    except Exception as e:
                        print(f"[WARNING] pqa_labeled not available: {e}")
                        raise
                    
                    # 转换为统一格式: text, label (0=no, 1=yes)
                    def convert_format(x):
                        text = str(x.get("question", "")) + " " + str(x.get("context", ""))
                        # 处理final_decision字段
                        final_decision = str(x.get("final_decision", "")).lower()
                        label = 1 if final_decision == "yes" else 0
                        return {"text": text, "label": label}
                    
                    full_dataset = full_dataset.map(convert_format)
                    
                    # 手动分割train/validation (80/20)
                    # 使用固定的随机种子确保可复现
                    full_dataset = full_dataset.shuffle(seed=42)
                    total_size = len(full_dataset)
                    train_size = int(0.8 * total_size)
                    
                    if split == "train":
                        dataset = full_dataset.select(range(train_size))
                    elif split in ["validation", "test"]:
                        dataset = full_dataset.select(range(train_size, total_size))
                    else:
                        raise ValueError(f"Unknown split: {split}")
                elif dataset_name == "imdb":
                    # IMDb电影评论数据集（用于情感分析，可以模拟医疗场景）
                    try:
                        dataset = load_dataset("imdb", split=split)
                        # 转换为统一格式
                        def convert_format(x):
                            return {"text": str(x.get("text", "")), "label": int(x.get("label", 0))}
                        dataset = dataset.map(convert_format)
                    except Exception as e:
                        print(f"[WARNING] Failed to load imdb: {e}")
                        raise
                elif dataset_name == "medical_questions":
                    # 尝试加载医疗问答数据集（如果可用）
                    try:
                        dataset = load_dataset("medical_questions_pairs", split=split)
                    except Exception:
                        print("[WARNING] medical_questions_pairs not available, trying alternatives...")
                        raise
                elif dataset_name == "mimic_iii":
                    # MIMIC-III 数据集（需要申请访问权限）
                    # 注意：这个数据集需要申请访问，不能直接使用
                    print("[WARNING] MIMIC-III requires access approval. Using synthetic data instead.")
                    raise ValueError("MIMIC-III requires access approval")
                else:
                    raise ValueError(f"Unknown dataset: {dataset_name}. Available: 'pubmed_qa', 'synthetic'")
            except Exception as e:
                print(f"[WARNING] Failed to load real dataset '{dataset_name}': {e}")
                print("[INFO] Using synthetic medical data instead...")
                print("[INFO] To use real datasets, try:")
                print("  1. pubmed_qa: pip install datasets, then use dataset_name='pubmed_qa'")
                print("  2. Or download your own medical dataset and modify the code")
                dataset = self._create_synthetic_medical_data(max_samples or 5000)

        # -------------------------
        # 2. 限制样本数 & 类别平衡
        # -------------------------
        if max_samples and len(dataset) > max_samples:
            # 分别获取正负样本
            positive_samples = [x for x in dataset if int(x["label"]) == 1]
            negative_samples = [x for x in dataset if int(x["label"]) == 0]

            num_per_class = max_samples // 2
            self.rng.shuffle(positive_samples)
            self.rng.shuffle(negative_samples)

            positive_selected = positive_samples[:num_per_class]
            negative_selected = negative_samples[:num_per_class]

            selected_data = positive_selected + negative_selected
            self.rng.shuffle(selected_data)

            dataset_dict = {
                "text": [x["text"] for x in selected_data],
                "label": [int(x["label"]) for x in selected_data],
            }
            dataset = HFDataset.from_dict(dataset_dict)

        print(f"[INFO] Raw dataset size: {len(dataset)}")
        label_counter = Counter(int(x["label"]) for x in dataset)
        print(f"[INFO] Label distribution (raw): {dict(label_counter)}")

        # -------------------------
        # 3. 加载 tokenizer
        # -------------------------
        print(f"[INFO] Loading tokenizer: {TOKENIZER_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        if self.tokenizer.pad_token is None:
            # 对于 BERT 一般 pad_token 已经存在，这里只是兜底
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.sep_token
            )

        # -------------------------
        # 4. 处理数据 & 构造约束
        # -------------------------
        self.data = []
        self.labels = []
        self.constraints = []
        self.violations = []

        print(f"[INFO] Processing {len(dataset)} samples...")
        for i, example in enumerate(dataset):
            if i % 1000 == 0:
                print(f"  Processing {i}/{len(dataset)}...")

            text = str(example["text"])
            label = int(example["label"])  # 0=normal, 1=abnormal/disease

            # Tokenize
            tokens = self.tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)

            # 约束 mask：默认 0，违反时全 1（你后面可以在 4D 里面选不同维度去乘）
            constraint_mask = torch.zeros(max_length, dtype=torch.float)
            violation = 0.0

            if add_constraints:
                text_lower = text.lower()

                # 约束1：轻微症状不应该被标为“疾病/高风险”
                if label == 1:
                    has_mild_symptom = any(mild in text_lower for mild in MILD_SYMPTOMS)
                    has_severe_disease = any(
                        severe in text_lower for severe in SEVERE_DISEASES
                    )

                    # 如果只有轻微症状，没有严重疾病词，但被打成“异常”，标记为违反
                    if has_mild_symptom and not has_severe_disease:
                        violation = 1.0
                        constraint_mask.fill_(1.0)

                # 约束2：药物相互作用
                for drug1, drug2 in DRUG_INTERACTION_PAIRS:
                    if drug1 in text_lower and drug2 in text_lower:
                        violation = 1.0
                        constraint_mask.fill_(1.0)
                        break

                # 约束3：年龄相关
                has_pediatric = any(term in text_lower for term in PEDIATRIC_TERMS)
                has_adult_disease = any(
                    disease in text_lower for disease in ADULT_ONLY_DISEASES
                )
                if has_pediatric and has_adult_disease:
                    violation = 1.0
                    constraint_mask.fill_(1.0)

            self.data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )
            self.labels.append(label)
            self.constraints.append(constraint_mask)
            self.violations.append(violation)

        print(f"[INFO] Dataset loaded: {len(self.data)} samples")
        total_viol = sum(self.violations)
        print(
            f"[INFO] Constraint violations (label-based, static): "
            f"{int(total_viol)}/{len(self.violations)} "
            f"({total_viol / len(self.violations) * 100:.2f}%)"
        )

    # ------------------------------------------------------------------
    # 合成医疗数据
    # ------------------------------------------------------------------
    def _create_synthetic_medical_data(self, num_samples: int = 5000) -> HFDataset:
        """创建合成的医疗文本数据"""
        print(f"[INFO] Creating synthetic medical data ({num_samples} samples)...")

        normal_templates = [
            "Patient reports {symptom}. Vital signs are normal. No immediate concerns.",
            "Routine checkup. Patient feels well. {symptom} is mild and manageable.",
            "Patient presents with {symptom}. Physical examination shows no abnormalities.",
            "Follow-up visit. Patient reports improvement. {symptom} has resolved.",
            "Annual physical. All systems normal. Patient reports occasional {symptom}.",
        ]

        abnormal_templates = [
            "Patient presents with severe {disease}. Immediate medical attention required.",
            "Diagnosis: {disease}. Patient shows multiple symptoms. Treatment plan initiated.",
            "Emergency case: {disease} confirmed. Patient in critical condition.",
            "Advanced {disease} detected. Comprehensive treatment protocol started.",
            "Patient diagnosed with {disease}. Multiple complications observed.",
        ]

        mild_symptoms = MILD_SYMPTOMS
        severe_diseases = SEVERE_DISEASES

        data = []

        # 生成正常样本（label=0）
        for _ in range(num_samples // 2):
            symptom = self.rng.choice(mild_symptoms)
            template = self.rng.choice(normal_templates)
            text = template.format(symptom=symptom)
            data.append({"text": text, "label": 0})

        # 生成异常样本（label=1）
        for _ in range(num_samples // 2):
            if self.rng.random() < 0.5:
                # 合理的：严重疾病
                disease = self.rng.choice(severe_diseases)
                template = self.rng.choice(abnormal_templates)
                text = template.format(disease=disease)
            else:
                # 潜在违反约束的：轻微症状被当成疾病
                symptom = self.rng.choice(mild_symptoms)
                template = self.rng.choice(abnormal_templates)
                text = template.format(disease=symptom)
            data.append({"text": text, "label": 1})

        self.rng.shuffle(data)

        dataset_dict = {
            "text": [x["text"] for x in data],
            "label": [x["label"] for x in data],
        }
        return HFDataset.from_dict(dataset_dict)

    # ------------------------------------------------------------------
    # PyTorch Dataset 接口
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.data[idx]["input_ids"],
            "attention_mask": self.data[idx]["attention_mask"],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            # constraint_mask 可以直接喂进你 4D 模型的某一维度 / 多维组合
            "constraint": self.constraints[idx],
            # violation 是一个标量标签（当前数据点是否违反静态约束）
            "violation": torch.tensor(self.violations[idx], dtype=torch.float),
        }


# ====================
# 批处理函数（给 DataLoader 用）
# ====================

def collate_fn(batch):
    """批处理函数：把单条 sample 堆叠成 batch"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    constraints = torch.stack([item["constraint"] for item in batch])
    violations = torch.stack([item["violation"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "constraints": constraints,
        "violations": violations,
    }


# ====================
# 一个简单的 DataLoader 工具函数（方便你直接接到训练脚本）
# ====================

def get_medical_dataloaders(
    dataset_name: str = "synthetic",
    max_train_samples: int = 5000,
    max_val_samples: int = 1000,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    seed: int = 42,
):
    train_dataset = MedicalConstrainedTextClassificationDataset(
        dataset_name=dataset_name,
        split="train",
        max_samples=max_train_samples,
        max_length=max_length,
        add_constraints=True,
        use_synthetic=(dataset_name == "synthetic"),
        seed=seed,
    )

    # 对于合成数据，直接再生成一份做 val；真实数据就可以用 split="test"
    val_dataset = MedicalConstrainedTextClassificationDataset(
        dataset_name=dataset_name,
        split="validation",
        max_samples=max_val_samples,
        max_length=max_length,
        add_constraints=True,
        use_synthetic=(dataset_name == "synthetic"),
        seed=seed + 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


# ====================
# 本地快速测试
# ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Medical Dataset Loading")
    print("=" * 60)

    print("\n[1/1] Testing synthetic medical dataset...")
    try:
        train_dataset = MedicalConstrainedTextClassificationDataset(
            dataset_name="synthetic",
            split="train",
            max_samples=100,  # 只加载 100 个样本用于测试
            max_length=128,
            add_constraints=True,
            use_synthetic=True,
            seed=123,
        )

        print(f"  Dataset size: {len(train_dataset)}")
        sample = train_dataset[0]
        print(f"  Sample 0:")
        print(f"    Input shape: {sample['input_ids'].shape}")
        print(f"    Label: {sample['label']}")
        print(f"    Violation: {sample['violation']}")
        print(f"    Constraint mask sum: {sample['constraint'].sum()}")

        print("\n  Sample texts (meta-info only):")
        for i in range(min(5, len(train_dataset))):
            s = train_dataset[i]
            print(
                f"    Sample {i}: "
                f"Label={int(s['label'])}, "
                f"Violation={float(s['violation'])}, "
                f"ConstraintSum={float(s['constraint'].sum())}"
            )

        # 再测一把 DataLoader
        loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
        )
        batch = next(iter(loader))
        print("\n  Batch test:")
        print(f"    input_ids: {batch['input_ids'].shape}")
        print(f"    labels: {batch['labels'].shape}")
        print(f"    constraints: {batch['constraints'].shape}")
        print(f"    violations: {batch['violations'].shape}")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)