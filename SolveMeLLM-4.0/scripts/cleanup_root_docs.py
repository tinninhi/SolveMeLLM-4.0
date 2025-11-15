#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理根目录下的文档文件
将文档移动到docs目录的对应位置或archive
"""

import os
import shutil
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
ARCHIVE_DIR = DOCS_DIR / "archive"

# 确保目录存在
ARCHIVE_DIR.mkdir(exist_ok=True)
(DOCS_DIR / "evaluation").mkdir(exist_ok=True)
(DOCS_DIR / "guides").mkdir(exist_ok=True)
(DOCS_DIR / "results" / "training").mkdir(parents=True, exist_ok=True)

# 文件移动规则
MOVE_RULES = {
    # 评估文档 -> evaluation
    "下一步建议.md": "evaluation/下一步建议.md",
    "严格自我评估_诚实版.md": "evaluation/严格自我评估_诚实版.md",
    "最终诚实评估.md": "evaluation/最终诚实评估.md",
    
    # 使用指南 -> guides
    "真实医疗数据集使用指南.md": "guides/真实医疗数据集使用指南.md",
    "真实数据集使用说明.md": "guides/真实数据集使用说明.md",
    "训练时间估算.md": "guides/训练时间估算.md",
    
    # 训练结果分析 -> results/training 或 archive（过时的）
    "当前结果分析.md": "archive/当前结果分析.md",  # 过时的结果
    "优化后结果分析.md": "archive/优化后结果分析.md",  # 过时的结果
    "修复后训练结果分析.md": "archive/修复后训练结果分析.md",  # 过时的结果
    "训练结果详细分析.md": "archive/训练结果详细分析.md",  # 过时的结果
    "训练epoch分析.md": "archive/训练epoch分析.md",  # 过时的分析
    "训练中断分析.md": "archive/训练中断分析.md",  # 过时的分析
    
    # 过时的说明和计划 -> archive
    "优化计划.md": "archive/优化计划.md",
    "优化调整说明.md": "archive/优化调整说明.md",
    "继续优化说明.md": "archive/继续优化说明.md",
    "修复说明.md": "archive/修复说明.md",
    "快速迁移完成.md": "archive/快速迁移完成.md",
    "重新训练说明.md": "archive/重新训练说明.md",
}

def cleanup_root_docs():
    """清理根目录下的文档"""
    print("=" * 60)
    print("Cleaning up root directory documents...")
    print("=" * 60)
    
    moved_count = 0
    
    for root_file, target_path in MOVE_RULES.items():
        source_file = BASE_DIR / root_file
        
        if source_file.exists():
            try:
                target_file = DOCS_DIR / target_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_file), str(target_file))
                print(f"  [OK] Moved: {root_file} -> {target_path}")
                moved_count += 1
            except Exception as e:
                print(f"  [FAIL] Move failed: {root_file} - {e}")
        else:
            print(f"  [SKIP] Not found: {root_file}")
    
    print("=" * 60)
    print(f"Cleanup completed! Moved {moved_count} files")
    print("=" * 60)

if __name__ == "__main__":
    cleanup_root_docs()

