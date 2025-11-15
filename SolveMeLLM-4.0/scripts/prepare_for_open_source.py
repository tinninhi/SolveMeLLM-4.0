#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开源准备脚本
============

自动清理项目，准备开源：
1. 删除临时文件和缓存
2. 归档过时文档
3. 清理无用脚本
4. 创建.gitignore
5. 生成清理报告
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 需要删除的文件和目录
FILES_TO_DELETE = [
    # Python缓存
    '__pycache__',
    '**/__pycache__',
    '*.pyc',
    '*.pyo',
    '*.pyd',
    
    # 临时文件
    '*.whl',  # PyTorch wheel文件
    '*.log',
    '*.tmp',
    '*.swp',
    '*.swo',
    '*~',
    
    # 测试输出文件（根目录）
    'creative',
    'finance',
    'medical',
    'CS.TXT',
    
    # 临时Python文件
    'monitor_training.py',
    'test_pubmed_qa.py',
]

# 需要归档的文档（移动到archive）
DOCS_TO_ARCHIVE = [
    # evaluation目录中的过时文档
    'docs/evaluation/今日测试执行清单.md',
    'docs/evaluation/快速测试指南.md',
    'docs/evaluation/测试结果汇总模板.md',
    'docs/evaluation/多随机种子测试建议.md',
    'docs/evaluation/多随机种子测试执行计划.md',
    'docs/evaluation/多随机种子测试评估.md',
    'docs/evaluation/多随机种子测试进度.md',
    'docs/evaluation/Planner头测试开始.md',
    'docs/evaluation/Planner头测试计划.md',
    
    # 根目录的清理文档
    'docs/根目录清理总结.md',
    'docs/清理总结.md',
]

# 需要删除的脚本（过时或不再使用）
SCRIPTS_TO_DELETE = [
    # 过时的训练脚本（使用train_medical_dataset.py代替）
    'scripts/train_real_dataset.py',
    'scripts/train_constrained_classification.py',
    'scripts/train_constrained_classification_optimized.py',
    'scripts/train_constraint_enhanced.py',
    'scripts/train_safe_generation.py',
    'scripts/train_sequence_labeling.py',
    'scripts/train_sequence_labeling_quick.py',
    'scripts/train_4d_transformer_small.py',
    'scripts/train_large_optimized.py',
    'scripts/train_large_scale.py',
    'scripts/train_multi_seed.py',
    'scripts/train_multi_seed_quick.py',
    'scripts/train_hard_multi_seed.py',
    
    # 过时的测试脚本
    'scripts/test_optimized.py',
    'scripts/test_four_dim_comprehensive.py',
    'scripts/test_domains.py',
    'scripts/test_domain_comparison.py',
    'scripts/test_domain_steering.py',
    'scripts/run_all_tests.py',
    'scripts/run_domain_tests.py',
    'scripts/run_planner_tests.py',
    
    # 调试脚本
    'scripts/debug_constraint.py',
    'scripts/test_constraint_detection.py',
    'scripts/quick_test_dataset.py',
]

# 需要保留的核心脚本
CORE_SCRIPTS = [
    'scripts/train_medical_dataset.py',  # 主训练脚本（在根目录）
    'scripts/test_planner_head.py',
    'scripts/test_multi_seed_generic.py',
    'scripts/cleanup_docs.py',
    'scripts/cleanup_root_docs.py',
    'scripts/restore_core_docs.py',
]

# 需要删除的模型文件（过时版本）
MODELS_TO_DELETE = [
    'models/four_d_transformer_block.py',  # 旧版本，使用v2
    'models/four_d_transformer_constraint_enhanced.py',  # 已整合
    'models/four_dim_agent.py',  # 旧版本
    'models/four_dim_agent_optimized.py',  # 旧版本
    'models/four_dim_agent_optimized_v2.py',  # 旧版本
    'models/four_dim_agent_optimized_v3.py',  # 旧版本
]

# 需要保留的核心模型
CORE_MODELS = [
    'models/four_d_transformer_block-v2.py',  # 当前使用的版本
    'models/baseline_transformer.py',
]

def delete_files(patterns, base_dir=ROOT_DIR):
    """删除匹配模式的文件和目录"""
    deleted = []
    failed = []
    
    for pattern in patterns:
        # 处理通配符
        if '**' in pattern:
            # 递归搜索
            for path in base_dir.rglob(pattern.replace('**/', '')):
                try:
                    if path.is_file():
                        path.unlink()
                        deleted.append(str(path))
                    elif path.is_dir():
                        shutil.rmtree(path)
                        deleted.append(str(path))
                except Exception as e:
                    failed.append((str(path), str(e)))
        else:
            # 直接路径
            path = base_dir / pattern
            if path.exists():
                try:
                    if path.is_file():
                        path.unlink()
                        deleted.append(str(path))
                    elif path.is_dir():
                        shutil.rmtree(path)
                        deleted.append(str(path))
                except Exception as e:
                    failed.append((str(path), str(e)))
    
    return deleted, failed

def archive_docs(docs, archive_dir=None):
    """归档文档到archive目录"""
    if archive_dir is None:
        archive_dir = ROOT_DIR / 'docs' / 'archive'
    
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

def create_gitignore():
    """创建.gitignore文件"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt
*.ckpt

# Data
*.csv
*.json
*.pkl
*.h5
*.hdf5

# Logs
*.log
*.out

# OS
.DS_Store
Thumbs.db

# Project specific
*.whl
creative
finance
medical
CS.TXT
"""
    
    gitignore_path = ROOT_DIR / '.gitignore'
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    return str(gitignore_path)

def generate_report(deleted, archived, failed_delete, failed_archive, gitignore_path):
    """生成清理报告"""
    report = f"""# 开源准备清理报告

**清理时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 清理统计

### 删除的文件
- **总数**：{len(deleted)}
- **成功**：{len(deleted) - len(failed_delete)}
- **失败**：{len(failed_delete)}

### 归档的文档
- **总数**：{len(archived)}
- **成功**：{len(archived) - len(failed_archive)}
- **失败**：{len(failed_archive)}

### 创建的文件
- `.gitignore`：{gitignore_path}

## ✅ 成功删除的文件

"""
    
    for item in deleted[:50]:  # 只显示前50个
        report += f"- {item}\n"
    
    if len(deleted) > 50:
        report += f"\n... 还有 {len(deleted) - 50} 个文件已删除\n"
    
    report += "\n## 📦 成功归档的文档\n\n"
    for src, dst in archived:
        report += f"- {src} → {dst}\n"
    
    if failed_delete:
        report += "\n## ❌ 删除失败的文件\n\n"
        for path, error in failed_delete:
            report += f"- {path}: {error}\n"
    
    if failed_archive:
        report += "\n## ❌ 归档失败的文档\n\n"
        for path, error in failed_archive:
            report += f"- {path}: {error}\n"
    
    report += "\n## 📝 下一步\n\n"
    report += "1. 检查清理结果\n"
    report += "2. 完善README.md\n"
    report += "3. 添加LICENSE\n"
    report += "4. 准备发布\n"
    
    return report

def main():
    """主函数"""
    import sys
    
    # 检查是否有--yes参数
    auto_confirm = '--yes' in sys.argv or '-y' in sys.argv
    
    print("="*80)
    print("开源准备清理脚本")
    print("="*80)
    print(f"项目目录：{ROOT_DIR}")
    print(f"\n将执行以下操作：")
    print(f"  1. 删除临时文件和缓存")
    print(f"  2. 归档过时文档")
    print(f"  3. 删除过时脚本")
    print(f"  4. 删除过时模型文件")
    print(f"  5. 创建.gitignore")
    print(f"  6. 生成清理报告")
    
    if not auto_confirm:
        try:
            response = input("\n是否继续？(y/n): ")
            if response.lower() != 'y':
                print("已取消")
                return
        except EOFError:
            print("\n[INFO] 非交互式模式，自动确认")
            auto_confirm = True
    
    print("\n开始清理...")
    
    # 1. 删除临时文件和缓存
    print("\n[1/6] 删除临时文件和缓存...")
    deleted_temp, failed_temp = delete_files(FILES_TO_DELETE)
    print(f"  删除 {len(deleted_temp)} 个文件/目录")
    
    # 2. 归档过时文档
    print("\n[2/6] 归档过时文档...")
    archived, failed_archive = archive_docs(DOCS_TO_ARCHIVE)
    print(f"  归档 {len(archived)} 个文档")
    
    # 3. 删除过时脚本
    print("\n[3/6] 删除过时脚本...")
    deleted_scripts, failed_scripts = delete_files(SCRIPTS_TO_DELETE)
    print(f"  删除 {len(deleted_scripts)} 个脚本")
    
    # 4. 删除过时模型文件
    print("\n[4/6] 删除过时模型文件...")
    deleted_models, failed_models = delete_files(MODELS_TO_DELETE)
    print(f"  删除 {len(deleted_models)} 个模型文件")
    
    # 5. 创建.gitignore
    print("\n[5/6] 创建.gitignore...")
    gitignore_path = create_gitignore()
    print(f"  创建 {gitignore_path}")
    
    # 6. 生成清理报告
    print("\n[6/6] 生成清理报告...")
    all_deleted = deleted_temp + deleted_scripts + deleted_models
    all_failed = failed_temp + failed_scripts + failed_models
    
    report = generate_report(
        all_deleted, archived, all_failed, failed_archive, gitignore_path
    )
    
    report_path = ROOT_DIR / 'docs' / '清理报告_开源准备.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"  报告已保存到：{report_path}")
    
    # 汇总
    print("\n" + "="*80)
    print("清理完成！")
    print("="*80)
    print(f"删除文件：{len(all_deleted)}")
    print(f"归档文档：{len(archived)}")
    print(f"创建文件：.gitignore")
    print(f"清理报告：{report_path}")
    print("\n下一步：")
    print("  1. 检查清理结果")
    print("  2. 完善README.md")
    print("  3. 添加LICENSE")
    print("  4. 准备发布")

if __name__ == '__main__':
    main()

