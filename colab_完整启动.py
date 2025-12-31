#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VPR Project - Colab完整启动脚本（修复所有已知问题）
最后更新: 2025-12-31
"""

import os
import sys
import subprocess
from pathlib import Path

def print_section(title):
    """打印分隔线"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def check_environment():
    """检查环境"""
    print_section("Step 1: 检查环境")
    
    # 检查Colab
    try:
        import google.colab
        print("✅ Colab环境检测成功")
        IN_COLAB = True
    except:
        print("⚠️  非Colab环境")
        IN_COLAB = False
    
    # 检查GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU可用")
            # 提取GPU信息
            for line in result.stdout.split('\n'):
                if 'Tesla' in line or 'V100' in line or 'T4' in line or 'P100' in line or 'A100' in line:
                    print(f"   GPU型号: {line.strip()}")
                    break
        else:
            print("❌ GPU不可用")
    except:
        print("❌ nvidia-smi命令失败")
    
    return IN_COLAB

def install_dependencies(in_colab=True):
    """安装所有依赖（包括loguru）"""
    print_section("Step 2: 安装依赖包")
    
    # 完整的依赖列表（包括新发现的loguru）
    packages = [
        "faiss-cpu",      # 相似度搜索（使用CPU版本，更稳定）
        "h5py",           # HDF5文件处理
        "pillow",         # 图像读取
        "tqdm",           # 进度条
        "scikit-learn",   # 机器学习工具
        "gdown",          # Google Drive下载
        "kornia",         # 图像增强
        "loguru"          # 日志记录（重要！）
    ]
    
    print(f"将安装 {len(packages)} 个包...\n")
    
    # 逐个安装并检查
    success_count = 0
    failed_packages = []
    
    for i, pkg in enumerate(packages, 1):
        print(f"[{i}/{len(packages)}] 安装 {pkg}...", end=" ")
        try:
            result = subprocess.run(
                f"pip install -q {pkg}", 
                shell=True, 
                capture_output=True,
                timeout=120  # 2分钟超时
            )
            if result.returncode == 0:
                print("✅")
                success_count += 1
            else:
                print("❌")
                failed_packages.append(pkg)
                if result.stderr:
                    print(f"   错误: {result.stderr.decode()[:100]}")
        except subprocess.TimeoutExpired:
            print("⏱️ 超时")
            failed_packages.append(pkg)
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            failed_packages.append(pkg)
    
    print(f"\n{'='*50}")
    print(f"成功安装: {success_count}/{len(packages)}")
    if failed_packages:
        print(f"失败的包: {', '.join(failed_packages)}")
        print("⚠️  请手动安装失败的包")
    else:
        print("✅ 所有依赖已成功安装！")
    print("="*50)

def verify_installation():
    """验证所有关键模块"""
    print_section("Step 3: 验证安装")
    
    modules_to_check = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("faiss", "Faiss"),
        ("h5py", "H5Py"),
        ("PIL", "Pillow"),
        ("tqdm", "TQDM"),
        ("sklearn", "Scikit-Learn"),
        ("gdown", "GDown"),
        ("kornia", "Kornia"),
        ("loguru", "Loguru"),  # 重要！
        ("numpy", "NumPy")
    ]
    
    success_count = 0
    for module_name, display_name in modules_to_check:
        try:
            if module_name == "torch":
                import torch
                print(f"✅ {display_name:15} : {torch.__version__}")
                if torch.cuda.is_available():
                    print(f"   └─ CUDA可用: {torch.cuda.get_device_name(0)}")
                success_count += 1
            elif module_name == "faiss":
                import faiss
                print(f"✅ {display_name:15} : {faiss.__version__}")
                success_count += 1
            elif module_name == "loguru":
                from loguru import logger
                print(f"✅ {display_name:15} : 已安装（logger可用）")
                success_count += 1
            else:
                mod = __import__(module_name)
                version = getattr(mod, "__version__", "未知版本")
                print(f"✅ {display_name:15} : {version}")
                success_count += 1
        except ImportError as e:
            print(f"❌ {display_name:15} : 未安装")
            print(f"   错误: {str(e)[:60]}")
    
    print(f"\n{'='*50}")
    print(f"验证结果: {success_count}/{len(modules_to_check)} 模块可用")
    if success_count == len(modules_to_check):
        print("✅ 所有依赖验证通过！")
        return True
    else:
        print("⚠️  部分依赖缺失，请检查")
        return False
    print("="*50)

def check_datasets():
    """检查数据集"""
    print_section("Step 4: 检查数据集")
    
    datasets = {
        'GSV-XS (训练集)': 'data/gsv_xs',
        'SF-XS (测试集)': 'data/sf_xs',
        'Tokyo-XS (测试集)': 'data/tokyo_xs',
        'SVOX (测试集)': 'data/svox'
    }
    
    all_exist = True
    for name, path in datasets.items():
        if Path(path).exists():
            # 尝试获取大小
            try:
                result = subprocess.run(
                    f'du -sh {path}', 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                size = result.stdout.split()[0] if result.returncode == 0 else 'unknown'
                print(f"✅ {name:20} : 已存在 ({size})")
            except:
                print(f"✅ {name:20} : 已存在")
        else:
            print(f"❌ {name:20} : 不存在")
            all_exist = False
    
    print()
    if not all_exist:
        print("⚠️  数据集未完整，请运行: python download_datasets.py")
        return False
    else:
        print("✅ 所有数据集已准备好！")
        return True

def run_quick_test():
    """运行快速测试"""
    print_section("Step 5: 快速测试（可选）")
    
    response = input("是否运行快速测试实验？(约5-10分钟) [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("跳过测试")
        return
    
    print("\n运行测试实验...\n")
    
    cmd = [
        "python", "VPR-methods-evaluation/main.py",
        "--method", "cosplace",
        "--backbone", "ResNet18",
        "--descriptors_dimension", "512",
        "--image_size", "512", "512",
        "--database_folder", "data/sf_xs/test/database",
        "--queries_folder", "data/sf_xs/test/queries",
        "--distance_metric", "l2",
        "--log_dir", "baseline/quick_test",
        "--num_preds_to_save", "20",
        "--max_queries_to_save", "3",
        "--recall_values", "1", "5", "10", "20",
        "--save_for_uncertainty",
        "--num_workers", "4",
        "--batch_size", "32"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ 测试成功完成！")
        print("查看结果: logs/baseline/quick_test/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 测试失败: {e}")
        print("请检查错误信息")

def main():
    """主函数"""
    print("\n" + "="*70)
    print("  🚀 VPR Project - Colab完整启动脚本")
    print("  📦 包含所有依赖（包括 loguru）")
    print("  🔧 修复所有已知问题")
    print("="*70)
    
    # Step 1: 检查环境
    in_colab = check_environment()
    
    # Step 2: 安装依赖
    install_dependencies(in_colab)
    
    # Step 3: 验证安装
    if not verify_installation():
        print("\n⚠️  验证未通过，请检查错误信息")
        sys.exit(1)
    
    # Step 4: 检查数据集
    datasets_ready = check_datasets()
    
    # Step 5: 可选测试
    if datasets_ready:
        run_quick_test()
    
    # 完成
    print_section("🎉 启动完成！")
    print("✅ 环境已就绪，可以开始实验了！")
    print("\n下一步:")
    if not datasets_ready:
        print("1. 下载数据集: python download_datasets.py")
    print("1. 列出实验批次: python colab_experiments_batches.py list")
    print("2. 运行批次: python colab_experiments_batches.py run batch1_cosplace")
    print("3. 查看进度: python check_completed.py")
    print("4. 查看结果: python analyze_results.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
