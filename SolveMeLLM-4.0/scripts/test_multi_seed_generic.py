#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic配置多随机种子测试
========================

快速验证Generic配置的稳定性（3个随机种子）
"""

import os
import sys
import subprocess
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
TRAIN_SCRIPT = ROOT_DIR / 'train_medical_dataset.py'

# 测试配置
SEEDS = [42, 123, 456]  # 3个随机种子
DOMAIN = 'generic'  # 只测试Generic配置

def modify_seed(seed: int):
    """修改train_medical_dataset.py中的SEED"""
    script_path = TRAIN_SCRIPT
    
    # 读取文件
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换SEED
    import re
    pattern = r"SEED = \d+"
    replacement = f"SEED = {seed}"
    new_content = re.sub(pattern, replacement, content)
    
    # 写回文件
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"[OK] Modified SEED to {seed}")

def modify_domain(domain: str):
    """修改train_medical_dataset.py中的DOMAIN_CONFIG"""
    script_path = TRAIN_SCRIPT
    
    # 读取文件
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换DOMAIN_CONFIG
    import re
    pattern = r"DOMAIN_CONFIG = ['\"].*?['\"]"
    replacement = f"DOMAIN_CONFIG = '{domain}'"
    new_content = re.sub(pattern, replacement, content)
    
    # 写回文件
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"[OK] Modified DOMAIN_CONFIG to '{domain}'")

def run_test(seed: int):
    """运行单个随机种子的测试"""
    print(f"\n{'='*80}")
    print(f"Testing Generic configuration with seed: {seed}")
    print(f"{'='*80}\n")
    
    # 修改配置
    modify_domain(DOMAIN)
    modify_seed(seed)
    
    # 运行训练
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        cwd=str(ROOT_DIR),
    )
    
    return result.returncode == 0

def main():
    """主函数"""
    print("="*80)
    print("Multi-Seed Testing: Generic Configuration")
    print("="*80)
    print(f"Domain: {DOMAIN}")
    print(f"Seeds to test: {SEEDS}")
    print(f"Each test will take ~20 minutes (20 epochs)")
    print(f"Total estimated time: ~{len(SEEDS) * 20} minutes")
    
    input("\nPress Enter to start testing...")
    
    results = {}
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n[{i}/{len(SEEDS)}] Testing seed {seed}...")
        success = run_test(seed)
        results[seed] = 'success' if success else 'failed'
    
    # 汇总
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")
    for seed, status in results.items():
        print(f"  Seed {seed}: {status}")
    print("="*80)
    print("\nPlease check the results and compare:")
    print("  - Best validation accuracy")
    print("  - Best validation loss")
    print("  - Violation rate")
    print("  - Training-validation gap")
    print("\nIf the results are stable (accuracy difference < 0.5%),")
    print("you can proceed to Planner head testing.")

if __name__ == '__main__':
    main()

