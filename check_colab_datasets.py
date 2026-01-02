#!/usr/bin/env python3
"""
检查 Colab 中数据集的位置和结构
"""

from pathlib import Path
import os

print("="*80)
print("检查 Colab 中的数据集")
print("="*80)

# 检查当前工作目录
print(f"\n当前工作目录: {os.getcwd()}")

# 检查可能的 data 目录位置
possible_data_paths = [
    "data",
    "/content/data",
    "/content/drive/MyDrive/Visual-Place-Recognition-Project/data",
    "/content/drive/MyDrive/data",
    "./data",
    "../data",
]

print("\n检查可能的 data 目录:")
for path_str in possible_data_paths:
    path = Path(path_str)
    if path.exists():
        print(f"✅ 找到: {path.absolute()}")
        # 列出子目录
        if path.is_dir():
            subdirs = [d.name for d in path.iterdir() if d.is_dir()]
            print(f"   子目录: {subdirs}")
    else:
        print(f"❌ 不存在: {path_str}")

# 检查 tokyo_xs 数据集
print("\n" + "="*80)
print("检查 tokyo_xs 数据集")
print("="*80)

tokyo_paths = [
    "data/tokyo_xs",
    "/content/data/tokyo_xs",
    "/content/drive/MyDrive/Visual-Place-Recognition-Project/data/tokyo_xs",
]

for path_str in tokyo_paths:
    path = Path(path_str)
    if path.exists():
        print(f"\n✅ 找到 tokyo_xs: {path.absolute()}")
        # 检查结构
        test_path = path / "test"
        if test_path.exists():
            print(f"   test/ 目录存在")
            queries_path = test_path / "queries"
            database_path = test_path / "database"
            
            if queries_path.exists():
                files = list(queries_path.glob("*.jpg"))
                print(f"   queries/ 存在，包含 {len(files)} 个 .jpg 文件")
            else:
                print(f"   queries/ 不存在")
                # 尝试递归搜索
                all_files = list(test_path.rglob("*.jpg"))
                print(f"   在 test/ 下递归找到 {len(all_files)} 个 .jpg 文件")
                if all_files:
                    # 找到包含最多文件的目录
                    from collections import Counter
                    dirs = [f.parent for f in all_files]
                    most_common_dir = Counter(dirs).most_common(1)[0]
                    print(f"   大多数文件在: {most_common_dir[0]} ({most_common_dir[1]} 个文件)")
            
            if database_path.exists():
                files = list(database_path.glob("*.jpg"))
                print(f"   database/ 存在，包含 {len(files)} 个 .jpg 文件")
            else:
                print(f"   database/ 不存在")
        else:
            print(f"   test/ 目录不存在")
            # 列出所有子目录
            subdirs = [d.name for d in path.iterdir() if d.is_dir()]
            print(f"   子目录: {subdirs}")
        break
else:
    print("\n❌ 未找到 tokyo_xs 数据集")
    print("\n建议:")
    print("1. 检查数据集是否已下载到 Colab")
    print("2. 运行: python download_datasets.py")
    print("3. 或者检查数据集是否在其他位置")

# 检查其他数据集
print("\n" + "="*80)
print("检查其他数据集")
print("="*80)

datasets = ['sf_xs', 'svox', 'gsv_xs']
for dataset in datasets:
    for path_str in possible_data_paths:
        dataset_path = Path(path_str) / dataset
        if dataset_path.exists():
            print(f"✅ {dataset}: {dataset_path.absolute()}")
            break
    else:
        print(f"❌ {dataset}: 未找到")

print("\n" + "="*80)
