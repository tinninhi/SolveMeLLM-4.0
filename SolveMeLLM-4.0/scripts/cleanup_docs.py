#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档清理脚本
将过时和重复的文档移动到archive目录
"""

import os
import shutil
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
ARCHIVE_DIR = DOCS_DIR / "archive"

# 确保archive目录存在
ARCHIVE_DIR.mkdir(exist_ok=True)

# 要保留的核心文档（相对路径）
KEEP_FILES = {
    # 架构设计（全部保留）
    "architecture/4D_Transformer_Block设计.md",
    "architecture/4D架构战略分析.md",
    "architecture/4维架构全面测试结果.md",
    
    # 核心测试结果
    "results/约束分类任务测试结果.md",
    "results/序列标注任务测试结果.md",
    "results/可扩展性验证报告.md",
    "results/真实数据集完整结果分析.md",
    "results/真实数据集训练结果分析.md",
    "results/完整多随机种子测试结果.md",
    "results/20种子更难任务测试结果.md",
    "results/medical/PubMedQA问题总结与解决方案.md",
    
    # 架构规范
    "optimization/4D架构v1.0规范.md",
    "optimization/诚实评估_最终版.md",
    
    # 项目总结
    "evaluation/项目总结报告.md",
    "evaluation/最终结论.md",
    
    # 使用指南（全部保留）
    "guides/GPU安装指南_手动.md",
    "guides/GPU手动安装步骤.md",
    "guides/GPU训练说明.md",
    "guides/GPU配置说明.md",
    "guides/GPU训练结果分析.md",
    
    # 当前计划
    "plans/当前使用的模型说明.md",
    
    # 索引文件
    "INDEX.md",
}

def move_to_archive(file_path, relative_path):
    """移动文件到archive目录"""
    try:
        archive_path = ARCHIVE_DIR / relative_path
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file_path), str(archive_path))
        print(f"  [OK] Archived: {relative_path}")
        return True
    except Exception as e:
        print(f"  [FAIL] Archive failed: {relative_path} - {e}")
        return False

def cleanup_docs():
    """清理文档"""
    print("=" * 60)
    print("Starting document cleanup...")
    print("=" * 60)
    
    moved_count = 0
    
    # 遍历docs目录下的所有md文件
    for md_file in DOCS_DIR.rglob("*.md"):
        # 跳过archive目录和INDEX.md
        if "archive" in md_file.parts:
            continue
        
        # 计算相对路径
        try:
            relative_path = md_file.relative_to(DOCS_DIR)
            relative_str = str(relative_path).replace("\\", "/")
        except:
            continue
        
        # 检查是否应该保留
        if relative_str in KEEP_FILES:
            print(f"  [KEEP] {relative_str}")
            continue
        
        # 移动到archive
        if move_to_archive(md_file, relative_path):
            moved_count += 1
    
    print("=" * 60)
    print(f"Cleanup completed! Archived {moved_count} files")
    print(f"Archive directory: {ARCHIVE_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    cleanup_docs()

