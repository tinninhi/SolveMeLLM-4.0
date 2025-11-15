#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激进文档清理脚本
================

进一步清理过时和重复的文档，只保留最核心的文档
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
DOCS_DIR = ROOT_DIR / 'docs'
ARCHIVE_DIR = DOCS_DIR / 'archive'

# 需要归档的evaluation文档（过时或重复）
EVALUATION_TO_ARCHIVE = [
    # 重复的评估文档
    'docs/evaluation/严格自我评估_诚实版.md',
    'docs/evaluation/最终诚实评估.md',  # 保留一个即可
    'docs/evaluation/下一步建议.md',  # 过时
    'docs/evaluation/优化总结.md',  # 有优化实施说明就够了
    'docs/evaluation/策略调整说明.md',  # 过时
    'docs/evaluation/训练结果分析与策略调整.md',  # 过时
    'docs/evaluation/真实数据集测试策略.md',  # 过时
    'docs/evaluation/参数压缩实施说明.md',  # 有参数压缩方案就够了
    'docs/evaluation/强化正则化实施总结.md',  # 有方案就够了
    'docs/evaluation/过拟合严重-强化正则化方案.md',  # 有实施总结就够了
    'docs/evaluation/优化实施说明.md',  # 过时
    'docs/evaluation/完整测试计划.md',  # 过时，测试已完成
]

# 需要归档的guides文档（重复或过时）
GUIDES_TO_ARCHIVE = [
    # 重复的GPU文档
    'docs/guides/GPU安装指南_手动.md',  # 与GPU手动安装步骤重复
    'docs/guides/GPU训练结果分析.md',  # 过时
    'docs/guides/GPU训练说明.md',  # 过时
    'docs/guides/GPU配置说明.md',  # 过时
    
    # 重复的真实数据集文档
    'docs/guides/真实数据测试指南.md',  # 与真实数据集使用说明重复
    'docs/guides/真实数据集使用说明.md',  # 与真实医疗数据集使用指南重复
]

# 需要归档的results文档（过时或重复）
RESULTS_TO_ARCHIVE = [
    # 过时的结果文档
    'docs/results/20种子更难任务测试结果.md',  # 过时
    'docs/results/完整多随机种子测试结果.md',  # 有training目录下的就够了
    'docs/results/序列标注任务测试结果.md',  # 过时
    'docs/results/约束分类任务测试结果.md',  # 过时
    'docs/results/可扩展性验证报告.md',  # 过时
    'docs/results/真实数据集完整结果分析.md',  # 有training目录下的就够了
    'docs/results/真实数据集训练结果分析.md',  # 有training目录下的就够了
]

# 需要归档的其他文档
OTHER_TO_ARCHIVE = [
    # 根目录的临时文档
    'docs/开源准备清单.md',  # 已完成，可以归档
    'docs/清理报告_开源准备.md',  # 已完成，可以归档
]

# 需要保留的核心文档（不归档）
CORE_DOCS = [
    # evaluation核心文档
    'docs/evaluation/项目价值评估与开源建议.md',
    'docs/evaluation/项目总结报告.md',
    'docs/evaluation/最终结论.md',
    'docs/evaluation/领域自适应方案评估.md',
    'docs/evaluation/Planner头实施总结.md',
    'docs/evaluation/Path维度规划调整功能分析.md',
    'docs/evaluation/Path维度重新评估与建议.md',
    'docs/evaluation/参数压缩方案.md',
    
    # guides核心文档
    'docs/guides/Planner头使用指南.md',
    'docs/guides/领域自适应实施总结.md',
    'docs/guides/真实医疗数据集使用指南.md',
    'docs/guides/训练时间估算.md',
    'docs/guides/GPU手动安装步骤.md',  # 保留一个GPU文档
    
    # results核心文档（training目录下的）
    'docs/results/training/阶段1测试总结.md',
    'docs/results/training/领域Profile测试结果汇总.md',
    'docs/results/training/领域Profile测试详细分析.md',
    'docs/results/training/多随机种子测试结果汇总.md',
    'docs/results/training/多随机种子测试详细分析.md',
    'docs/results/training/Generic配置训练结果分析.md',
    'docs/results/training/强化正则化训练结果分析.md',
    'docs/results/training/IMDb真实数据集训练结果分析.md',
    
    # 其他核心文档
    'docs/开源准备完成总结.md',
    'docs/最终整理总结.md',
]

def archive_docs(docs, archive_dir=None):
    """归档文档到archive目录"""
    if archive_dir is None:
        archive_dir = ARCHIVE_DIR
    
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    archived = []
    failed = []
    
    for doc_path in docs:
        src = ROOT_DIR / doc_path
        if src.exists():
            try:
                dst = archive_dir / src.name
                # 如果目标文件已存在，添加时间戳
                if dst.exists():
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    dst = archive_dir / f"{src.stem}_{timestamp}{src.suffix}"
                
                shutil.move(str(src), str(dst))
                archived.append((str(src), str(dst)))
            except Exception as e:
                failed.append((str(src), str(e)))
    
    return archived, failed

def main():
    """主函数"""
    import sys
    
    # 检查是否有--yes参数
    auto_confirm = '--yes' in sys.argv or '-y' in sys.argv
    
    print("="*80)
    print("激进文档清理脚本")
    print("="*80)
    print(f"项目目录：{ROOT_DIR}")
    print(f"\n将归档以下文档：")
    print(f"  - evaluation文档：{len(EVALUATION_TO_ARCHIVE)}个")
    print(f"  - guides文档：{len(GUIDES_TO_ARCHIVE)}个")
    print(f"  - results文档：{len(RESULTS_TO_ARCHIVE)}个")
    print(f"  - 其他文档：{len(OTHER_TO_ARCHIVE)}个")
    print(f"  总计：{len(EVALUATION_TO_ARCHIVE) + len(GUIDES_TO_ARCHIVE) + len(RESULTS_TO_ARCHIVE) + len(OTHER_TO_ARCHIVE)}个")
    
    if not auto_confirm:
        try:
            response = input("\n是否继续？(y/n): ")
            if response.lower() != 'y':
                print("已取消")
                return
        except EOFError:
            print("\n[INFO] 非交互式模式，自动确认")
            auto_confirm = True
    
    print("\n开始归档...")
    
    # 归档evaluation文档
    print(f"\n[1/4] 归档evaluation文档...")
    archived_eval, failed_eval = archive_docs(EVALUATION_TO_ARCHIVE)
    print(f"  归档 {len(archived_eval)} 个文档")
    
    # 归档guides文档
    print(f"\n[2/4] 归档guides文档...")
    archived_guides, failed_guides = archive_docs(GUIDES_TO_ARCHIVE)
    print(f"  归档 {len(archived_guides)} 个文档")
    
    # 归档results文档
    print(f"\n[3/4] 归档results文档...")
    archived_results, failed_results = archive_docs(RESULTS_TO_ARCHIVE)
    print(f"  归档 {len(archived_results)} 个文档")
    
    # 归档其他文档
    print(f"\n[4/4] 归档其他文档...")
    archived_other, failed_other = archive_docs(OTHER_TO_ARCHIVE)
    print(f"  归档 {len(archived_other)} 个文档")
    
    # 汇总
    all_archived = archived_eval + archived_guides + archived_results + archived_other
    all_failed = failed_eval + failed_guides + failed_results + failed_other
    
    # 生成报告
    report = f"""# 激进文档清理报告

**清理时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 清理统计

### 归档的文档
- **evaluation文档**：{len(archived_eval)}个
- **guides文档**：{len(archived_guides)}个
- **results文档**：{len(archived_results)}个
- **其他文档**：{len(archived_other)}个
- **总计**：{len(all_archived)}个

### 失败
- **失败数量**：{len(all_failed)}

## ✅ 成功归档的文档

### evaluation文档
"""
    
    for src, dst in archived_eval:
        report += f"- {src} → {dst}\n"
    
    report += "\n### guides文档\n"
    for src, dst in archived_guides:
        report += f"- {src} → {dst}\n"
    
    report += "\n### results文档\n"
    for src, dst in archived_results:
        report += f"- {src} → {dst}\n"
    
    report += "\n### 其他文档\n"
    for src, dst in archived_other:
        report += f"- {src} → {dst}\n"
    
    if all_failed:
        report += "\n## ❌ 归档失败的文档\n\n"
        for path, error in all_failed:
            report += f"- {path}: {error}\n"
    
    report += "\n## 📝 保留的核心文档\n\n"
    report += "以下文档已保留（核心文档）：\n\n"
    for doc in CORE_DOCS:
        report += f"- {doc}\n"
    
    report_path = DOCS_DIR / '清理报告_激进清理.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到：{report_path}")
    
    # 汇总
    print("\n" + "="*80)
    print("清理完成！")
    print("="*80)
    print(f"归档文档：{len(all_archived)}")
    print(f"失败：{len(all_failed)}")
    print(f"清理报告：{report_path}")
    print("\n保留的核心文档：")
    print(f"  - evaluation: 8个核心文档")
    print(f"  - guides: 5个核心文档")
    print(f"  - results/training: 8个核心文档")
    print(f"  - 其他: 2个核心文档")

if __name__ == '__main__':
    main()

