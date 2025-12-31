#!/usr/bin/env python3
"""
Colab实验批次配置
帮助你在Colab中分批运行实验
"""

import subprocess
import sys
from datetime import datetime

# 基础命令模板
BASE_CMD = """python VPR-methods-evaluation/main.py \
    --method {method} \
    --backbone {backbone} \
    --descriptors_dimension {dim} \
    --image_size {image_size} \
    --database_folder {db_folder} \
    --queries_folder {q_folder} \
    --distance_metric {metric} \
    --log_dir baseline/{log_name} \
    --num_preds_to_save 20 \
    --max_queries_to_save 3 \
    --recall_values 1 5 10 20 \
    --save_for_uncertainty \
    --num_workers 4 \
    --batch_size 32"""

# VPR方法配置
VPR_METHODS = {
    'cosplace': {'backbone': 'ResNet18', 'dim': 512, 'image_size': '512 512'},
    'netvlad': {'backbone': 'VGG16', 'dim': 4096, 'image_size': '512 512'},
    'mixvpr': {'backbone': 'ResNet50', 'dim': 4096, 'image_size': '320 320'},
    'megaloc': {'backbone': 'Dinov2', 'dim': 8448, 'image_size': '224 224'},
}

# 数据集配置
DATASETS = {
    'SF-XS': {
        'db_folder': 'data/sf_xs/test/database',
        'q_folder': 'data/sf_xs/test/queries',
        'short_name': 'sf_xs'
    },
    'Tokyo-XS': {
        'db_folder': 'data/tokyo_xs/test/database',
        'q_folder': 'data/tokyo_xs/test/queries',
        'short_name': 'tokyo_xs'
    },
    'SVOX-sun': {
        'db_folder': 'data/svox/images/test/gallery',
        'q_folder': 'data/svox/images/test/queries',
        'short_name': 'svox_sun'
    },
    'SVOX-night': {
        'db_folder': 'data/svox/images/test/gallery',
        'q_folder': 'data/svox/images/test/queries_night',
        'short_name': 'svox_night'
    },
}

METRICS = ['l2', 'dot_product']

# 实验批次定义
BATCHES = {
    'batch1_cosplace': {
        'name': 'CosPlace实验',
        'methods': ['cosplace'],
        'datasets': ['SF-XS', 'Tokyo-XS', 'SVOX-sun', 'SVOX-night'],
        'metrics': ['l2', 'dot_product'],
        'estimated_time': '1.5-2小时'
    },
    'batch2_netvlad': {
        'name': 'NetVLAD实验',
        'methods': ['netvlad'],
        'datasets': ['SF-XS', 'Tokyo-XS', 'SVOX-sun', 'SVOX-night'],
        'metrics': ['l2', 'dot_product'],
        'estimated_time': '2-2.5小时'
    },
    'batch3_mixvpr': {
        'name': 'MixVPR实验',
        'methods': ['mixvpr'],
        'datasets': ['SF-XS', 'Tokyo-XS', 'SVOX-sun', 'SVOX-night'],
        'metrics': ['l2', 'dot_product'],
        'estimated_time': '2-2.5小时'
    },
    'batch4_megaloc': {
        'name': 'MegaLoc实验',
        'methods': ['megaloc'],
        'datasets': ['SF-XS', 'Tokyo-XS', 'SVOX-sun', 'SVOX-night'],
        'metrics': ['l2', 'dot_product'],
        'estimated_time': '2.5-3小时'
    },
    'batch_quick_test': {
        'name': '快速测试（仅SF-XS）',
        'methods': ['cosplace'],
        'datasets': ['SF-XS'],
        'metrics': ['l2'],
        'estimated_time': '10-15分钟'
    }
}


def generate_command(method, dataset_name, metric):
    """生成单个实验命令"""
    method_config = VPR_METHODS[method]
    dataset_config = DATASETS[dataset_name]
    
    log_name = f"{method}_{metric}_{dataset_config['short_name']}_test"
    
    cmd = BASE_CMD.format(
        method=method,
        backbone=method_config['backbone'],
        dim=method_config['dim'],
        image_size=method_config['image_size'],
        db_folder=dataset_config['db_folder'],
        q_folder=dataset_config['q_folder'],
        metric=metric,
        log_name=log_name
    )
    
    return cmd, log_name


def run_batch(batch_name):
    """运行一个批次的实验"""
    if batch_name not in BATCHES:
        print(f"❌ 批次 '{batch_name}' 不存在")
        print(f"可用批次: {', '.join(BATCHES.keys())}")
        return
    
    batch = BATCHES[batch_name]
    
    print(f"\n{'='*80}")
    print(f"🚀 开始批次: {batch['name']}")
    print(f"预计时间: {batch['estimated_time']}")
    print(f"{'='*80}\n")
    
    # 统计实验数量
    total_experiments = len(batch['methods']) * len(batch['datasets']) * len(batch['metrics'])
    current = 0
    
    start_time = datetime.now()
    results = []
    
    for method in batch['methods']:
        for dataset_name in batch['datasets']:
            for metric in batch['metrics']:
                current += 1
                cmd, log_name = generate_command(method, dataset_name, metric)
                
                print(f"\n{'─'*80}")
                print(f"实验 {current}/{total_experiments}: {log_name}")
                print(f"{'─'*80}\n")
                
                try:
                    result = subprocess.run(cmd, shell=True, check=True)
                    status = "✅ 成功"
                    results.append((log_name, True))
                except subprocess.CalledProcessError as e:
                    status = "❌ 失败"
                    results.append((log_name, False))
                    print(f"\n错误: {e}")
                
                print(f"\n{status}: {log_name}")
    
    # 总结
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"批次完成: {batch['name']}")
    print(f"总用时: {duration}")
    print(f"{'='*80}\n")
    
    success_count = sum(1 for _, success in results if success)
    print(f"成功: {success_count}/{total_experiments}")
    
    if success_count < total_experiments:
        print("\n失败的实验:")
        for name, success in results:
            if not success:
                print(f"  ❌ {name}")


def list_batches():
    """列出所有批次"""
    print("\n📋 可用的实验批次:\n")
    for i, (batch_id, batch) in enumerate(BATCHES.items(), 1):
        exp_count = len(batch['methods']) * len(batch['datasets']) * len(batch['metrics'])
        print(f"{i}. {batch_id}")
        print(f"   名称: {batch['name']}")
        print(f"   实验数: {exp_count}个")
        print(f"   预计时间: {batch['estimated_time']}")
        print()


def print_commands(batch_name):
    """打印批次的所有命令（不执行）"""
    if batch_name not in BATCHES:
        print(f"❌ 批次 '{batch_name}' 不存在")
        return
    
    batch = BATCHES[batch_name]
    print(f"\n批次: {batch['name']}\n")
    
    for method in batch['methods']:
        for dataset_name in batch['datasets']:
            for metric in batch['metrics']:
                cmd, log_name = generate_command(method, dataset_name, metric)
                print(f"# {log_name}")
                print(cmd)
                print()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("""
╔═══════════════════════════════════════════════════════════════╗
║         Colab实验批次管理工具                                  ║
╚═══════════════════════════════════════════════════════════════╝

用法:
  python colab_experiments_batches.py list              - 列出所有批次
  python colab_experiments_batches.py run <batch_name>  - 运行指定批次
  python colab_experiments_batches.py print <batch_name> - 打印命令（不运行）

示例:
  python colab_experiments_batches.py run batch_quick_test
  python colab_experiments_batches.py run batch1_cosplace
        """)
        list_batches()
        return
    
    command = sys.argv[1]
    
    if command == 'list':
        list_batches()
    
    elif command == 'run':
        if len(sys.argv) < 3:
            print("❌ 请指定批次名称")
            list_batches()
            return
        batch_name = sys.argv[2]
        run_batch(batch_name)
    
    elif command == 'print':
        if len(sys.argv) < 3:
            print("❌ 请指定批次名称")
            return
        batch_name = sys.argv[2]
        print_commands(batch_name)
    
    else:
        print(f"❌ 未知命令: {command}")
        print("可用命令: list, run, print")


if __name__ == "__main__":
    main()
