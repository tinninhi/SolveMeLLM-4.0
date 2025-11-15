#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planner头测试脚本
================

自动测试Planner头的效果（无Planner、mean pooling、last pooling）
"""

import os
import sys
import subprocess
import re
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
TRAIN_SCRIPT = ROOT_DIR / 'train_medical_dataset.py'

# 测试配置
PLANNER_CONFIGS = [
    (None, None, 'no_planner'),
    (128, 'mean', 'planner_mean'),
    (128, 'last', 'planner_last'),
]

def modify_planner_config(planner_dim, planner_pooling):
    """修改train_medical_dataset.py中的Planner配置"""
    script_path = TRAIN_SCRIPT
    
    # 读取文件
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换PLANNER_DIM
    pattern_dim = r"PLANNER_DIM = .*"
    if planner_dim is None:
        replacement_dim = "PLANNER_DIM = None"
    else:
        replacement_dim = f"PLANNER_DIM = {planner_dim}"
    content = re.sub(pattern_dim, replacement_dim, content)
    
    # 替换PLANNER_POOLING
    pattern_pooling = r"PLANNER_POOLING = .*"
    if planner_pooling is None:
        replacement_pooling = "PLANNER_POOLING = 'mean'"
    else:
        replacement_pooling = f"PLANNER_POOLING = '{planner_pooling}'"
    content = re.sub(pattern_pooling, replacement_pooling, content)
    
    # 写回文件
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if planner_dim is None:
        print(f"[OK] Modified PLANNER_DIM to None (disabled)")
    else:
        print(f"[OK] Modified PLANNER_DIM to {planner_dim}, PLANNER_POOLING to '{planner_pooling}'")

def run_test(planner_dim, planner_pooling, test_name):
    """运行单个Planner配置的测试"""
    print(f"\n{'='*80}")
    print(f"Testing Planner: {test_name}")
    if planner_dim:
        print(f"  planner_dim={planner_dim}, pooling={planner_pooling}")
    else:
        print(f"  No Planner")
    print(f"{'='*80}\n")
    
    # 修改配置
    modify_planner_config(planner_dim, planner_pooling)
    
    # 运行训练脚本
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=str(ROOT_DIR),
    )
    
    return result.returncode == 0

def main():
    """主函数"""
    print("="*80)
    print("Planner Head Testing")
    print("="*80)
    print(f"Configs to test: {len(PLANNER_CONFIGS)}")
    print(f"Each test will take ~20 minutes (20 epochs)")
    print(f"Total estimated time: ~{len(PLANNER_CONFIGS) * 20} minutes")
    
    input("\nPress Enter to start testing...")
    
    results = {}
    for i, (planner_dim, planner_pooling, test_name) in enumerate(PLANNER_CONFIGS, 1):
        print(f"\n[{i}/{len(PLANNER_CONFIGS)}] Testing {test_name}...")
        success = run_test(planner_dim, planner_pooling, test_name)
        results[test_name] = 'success' if success else 'failed'
    
    # 汇总
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    for test_name, status in results.items():
        print(f"  {test_name}: {status}")
    print("="*80)
    print("\nPlease check the results and compare:")
    print("  - Best validation accuracy")
    print("  - Best validation loss")
    print("  - Violation rate")
    print("  - Training-validation gap")
    print("\nIf Planner head improves performance, we can keep it.")
    print("Otherwise, we can remove it (it's optional).")

if __name__ == '__main__':
    main()

