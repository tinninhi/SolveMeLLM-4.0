#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
恢复核心文档脚本
从archive目录恢复核心文档到正确位置
"""

import os
import shutil
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
ARCHIVE_DIR = DOCS_DIR / "archive"

# 需要恢复的核心文档
CORE_FILES = {
    # 架构规范
    "4D架构v1.0规范.md": "optimization/4D架构v1.0规范.md",
    "诚实评估_最终版.md": "optimization/诚实评估_最终版.md",
    
    # 项目总结
    "项目总结报告.md": "evaluation/项目总结报告.md",
    "最终结论.md": "evaluation/最终结论.md",
    
    # 核心测试结果
    "约束分类任务测试结果.md": "results/约束分类任务测试结果.md",
    "序列标注任务测试结果.md": "results/序列标注任务测试结果.md",
    "可扩展性验证报告.md": "results/可扩展性验证报告.md",
    "真实数据集完整结果分析.md": "results/真实数据集完整结果分析.md",
    "真实数据集训练结果分析.md": "results/真实数据集训练结果分析.md",
    "完整多随机种子测试结果.md": "results/完整多随机种子测试结果.md",
    "20种子更难任务测试结果.md": "results/20种子更难任务测试结果.md",
    
    # 医疗数据集
    "PubMedQA问题总结与解决方案.md": "results/medical/PubMedQA问题总结与解决方案.md",
    
    # 当前计划
    "当前使用的模型说明.md": "plans/当前使用的模型说明.md",
}

def restore_core_docs():
    """恢复核心文档"""
    print("=" * 60)
    print("Restoring core documents...")
    print("=" * 60)
    
    restored_count = 0
    
    for archive_name, target_path in CORE_FILES.items():
        archive_file = ARCHIVE_DIR / archive_name
        target_file = DOCS_DIR / target_path
        
        if archive_file.exists():
            try:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(archive_file), str(target_file))
                print(f"  [OK] Restored: {target_path}")
                restored_count += 1
            except Exception as e:
                print(f"  [FAIL] Restore failed: {target_path} - {e}")
        else:
            print(f"  [SKIP] Not found in archive: {archive_name}")
    
    print("=" * 60)
    print(f"Restore completed! Restored {restored_count} files")
    print("=" * 60)

if __name__ == "__main__":
    restore_core_docs()

