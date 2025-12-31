#!/usr/bin/env python3
"""
Colab快速启动脚本
在Colab中运行此脚本来自动设置环境
"""

import os
import sys
import subprocess
from pathlib import Path

def print_step(step_num, description):
    """打印步骤信息"""
    print(f"\n{'='*80}")
    print(f"Step {step_num}: {description}")
    print(f"{'='*80}\n")

def check_gpu():
    """检查GPU"""
    print_step(1, "检查GPU环境")
    subprocess.run("nvidia-smi", shell=True)
    
    import torch
    print(f"\n✅ PyTorch版本: {torch.__version__}")
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  警告: 未检测到GPU!")
        return False
    return True

def check_disk_space():
    """检查磁盘空间"""
    result = subprocess.run("df -h /", shell=True, capture_output=True, text=True)
    print(f"\n磁盘空间:")
    print(result.stdout)

def install_dependencies():
    """安装依赖"""
    print_step(2, "安装依赖包")
    
    # 检测是否在Colab环境
    try:
        import google.colab
        IN_COLAB = True
        print("✅ 检测到Colab环境")
    except:
        IN_COLAB = False
        print("⚠️  本地环境")
    
    # Colab环境下faiss-gpu需要特殊安装
    if IN_COLAB:
        print("安装 faiss-gpu (Colab)...")
        # Colab已经预装了faiss，或者使用conda
        result = subprocess.run("pip install -q faiss-gpu", shell=True, capture_output=True)
        if result.returncode != 0:
            print("⚠️  faiss-gpu安装失败，尝试faiss-cpu...")
            subprocess.run("pip install -q faiss-cpu", shell=True)
    else:
        print("安装 faiss-gpu...")
        subprocess.run("pip install -q faiss-gpu", shell=True)
    
    # 其他包
    packages = [
        "h5py",
        "pillow",
        "tqdm",
        "scikit-learn",
        "gdown",
        "kornia",
        "loguru"
    ]
    
    for pkg in packages:
        print(f"安装 {pkg}...")
        subprocess.run(f"pip install -q {pkg}", shell=True)
    
    print("\n✅ 所有依赖已安装")
    
    # 验证
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except:
        print("❌ PyTorch未安装")
    
    try:
        import faiss
        print(f"✅ Faiss: {faiss.__version__}")
    except:
        print("⚠️  Faiss未安装（可能需要手动安装）")

def check_datasets():
    """检查数据集"""
    print_step(3, "检查数据集")
    
    datasets = {
        'GSV-XS (训练集)': 'data/gsv_xs',
        'SF-XS (测试集)': 'data/sf_xs',
        'Tokyo-XS (测试集)': 'data/tokyo_xs',
        'SVOX (测试集)': 'data/svox'
    }
    
    all_exist = True
    for name, path in datasets.items():
        if Path(path).exists():
            print(f"✅ {name}: 已存在")
        else:
            print(f"❌ {name}: 不存在")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  需要下载数据集，运行: python download_datasets.py")
    else:
        print("\n✅ 所有数据集已准备好")
        # 显示大小
        subprocess.run("du -sh data/*", shell=True)
    
    return all_exist

def verify_code():
    """验证代码文件"""
    print_step(4, "验证项目文件")
    
    required_files = [
        'VPR-methods-evaluation/main.py',
        'VPR-methods-evaluation/parser.py',
        'VPR-methods-evaluation/visualizations.py',
        'run_all_remaining.py',
        'check_completed.py'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} 缺失!")
            all_exist = False
    
    return all_exist

def run_test_experiment():
    """运行测试实验"""
    print_step(5, "运行测试实验")
    
    print("开始运行快速测试实验...")
    print("方法: CosPlace, 数据集: SF-XS, 距离: L2")
    print("预计时间: 5-10分钟\n")
    
    cmd = """
    python VPR-methods-evaluation/main.py \
        --method cosplace \
        --backbone ResNet18 \
        --descriptors_dimension 512 \
        --image_size 512 512 \
        --database_folder data/sf_xs/test/database \
        --queries_folder data/sf_xs/test/queries \
        --distance_metric l2 \
        --log_dir baseline/cosplace_l2_sf_xs_test \
        --num_preds_to_save 20 \
        --max_queries_to_save 3 \
        --recall_values 1 5 10 20 \
        --save_for_uncertainty \
        --num_workers 4 \
        --batch_size 32
    """
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print("\n✅ 测试实验成功!")
        return True
    else:
        print("\n❌ 测试实验失败")
        return False

def show_results():
    """显示结果"""
    print_step(6, "查看测试结果")
    
    log_dirs = list(Path('logs/baseline').glob('*/*'))
    if not log_dirs:
        print("❌ 没有找到实验结果")
        return
    
    import os
    latest = max(log_dirs, key=os.path.getmtime)
    print(f"最新实验: {latest}\n")
    
    # 读取debug.log
    debug_log = latest / 'debug.log'
    if debug_log.exists():
        print("实验结果:")
        print(debug_log.read_text())
    
    # 检查可视化
    preds_dir = latest / 'preds'
    if preds_dir.exists():
        images = list(preds_dir.glob('*.jpg'))
        print(f"\n✅ 生成了 {len(images)} 张可视化图片")

def main():
    """主函数"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         VPR Project - Colab 快速启动脚本                      ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # 检查是否在Colab环境
    try:
        import google.colab
        IN_COLAB = True
        print("✅ 检测到Colab环境")
    except:
        IN_COLAB = False
        print("⚠️  不在Colab环境，但仍可运行")
    
    # 步骤1: 检查GPU
    if not check_gpu():
        print("\n⚠️  警告: 建议使用GPU运行")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            return
    
    check_disk_space()
    
    # 步骤2: 安装依赖
    try:
        install_dependencies()
    except Exception as e:
        print(f"❌ 安装依赖失败: {e}")
        return
    
    # 步骤3: 检查数据集
    datasets_ready = check_datasets()
    
    # 步骤4: 验证代码
    if not verify_code():
        print("\n❌ 缺少必要的项目文件!")
        print("请确保上传了完整的项目代码")
        return
    
    print("\n" + "="*80)
    print("环境检查完成!")
    print("="*80)
    
    # 询问是否运行测试
    if datasets_ready:
        print("\n📝 接下来可以:")
        print("  1. 运行测试实验 (快速验证)")
        print("  2. 运行所有实验 (长时间)")
        print("  3. 退出")
        
        choice = input("\n请选择 (1/2/3): ")
        
        if choice == '1':
            if run_test_experiment():
                show_results()
        elif choice == '2':
            print("\n开始运行所有实验...")
            subprocess.run("python run_all_remaining.py", shell=True)
        else:
            print("\n👋 退出")
    else:
        print("\n⚠️  请先下载数据集:")
        print("运行: python download_datasets.py")

if __name__ == "__main__":
    main()
