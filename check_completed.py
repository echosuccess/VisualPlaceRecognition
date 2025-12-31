#!/usr/bin/env python3
"""检查实际完成的实验"""
from pathlib import Path
import re

def check_all_experiments():
    """检查所有实验目录"""
    # 检查两个可能的目录
    log_dirs = [
        Path("logs/baseline"),
        Path("logs/logs/baseline")
    ]
    
    completed = []
    incomplete = []
    
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
            
        print(f"\n检查目录: {log_dir}")
        print("=" * 80)
        
        for exp_dir in sorted(log_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
                
            # 查找最新的运行结果
            subdirs = [d for d in exp_dir.iterdir() if d.is_dir()]
            if not subdirs:
                incomplete.append(str(exp_dir))
                continue
                
            # 检查最新的子目录
            latest_dir = max(subdirs, key=lambda x: x.name)
            info_log = latest_dir / "info.log"
            
            if info_log.exists():
                content = info_log.read_text(encoding='utf-8', errors='ignore')
                
                # 查找R@1结果
                match = re.search(r'R@1:\s*([\d.]+)', content)
                if match:
                    r1 = match.group(1)
                    completed.append({
                        'path': str(exp_dir.relative_to(log_dir.parent)),
                        'name': exp_dir.name,
                        'R@1': r1
                    })
                    print(f"[OK] {exp_dir.name:50s} R@1 = {r1}%")
                else:
                    incomplete.append(str(exp_dir))
                    print(f"[INCOMPLETE] {exp_dir.name:50s}")
            else:
                incomplete.append(str(exp_dir))
                print(f"[NO LOG] {exp_dir.name:50s}")
    
    print("\n" + "=" * 80)
    print(f"统计结果:")
    print("=" * 80)
    print(f"已完成: {len(completed)}/32")
    print(f"未完成: {32 - len(completed)}/32")
    
    if incomplete:
        print(f"\n未完成的实验:")
        for exp in incomplete:
            print(f"  - {exp}")
    
    return completed, incomplete

if __name__ == "__main__":
    completed, incomplete = check_all_experiments()

