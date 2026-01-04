#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的验证脚本
"""

from pathlib import Path
import json

results_dir = Path("results/image_matching")
completed = list(results_dir.glob("*.json"))

print(f"已完成实验: {len(completed)}")
print("\n检查每个文件:")

ok_count = 0
error_count = 0

for f in sorted(completed):
    size_mb = f.stat().st_size / (1024*1024)
    
    # 快速验证
    try:
        with open(str(f), 'r') as file:  # 修复：使用 str(f)
            data = json.load(file)
        num_queries = len(data.get('results', {}).get('query_ids', []))
        
        if num_queries > 0:
            print(f"✅ {f.name}")
            print(f"   大小: {size_mb:.1f}MB, 查询数: {num_queries}")
            ok_count += 1
        else:
            print(f"❌ {f.name} - 数据为空")
            error_count += 1
    except Exception as e:
        print(f"❌ {f.name} - 错误: {e}")
        error_count += 1

print(f"\n总结:")
print(f"  正常: {ok_count}")
print(f"  有问题: {error_count}")
print(f"  总计: {len(completed)}")

if error_count == 0:
    print("\n✅ 所有结果文件验证通过！数据完整，不需要重新运行。")
else:
    print(f"\n⚠️  发现 {error_count} 个有问题的文件，需要检查。")
