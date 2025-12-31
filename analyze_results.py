#!/usr/bin/env python3
"""
结果分析工具
用于分析和可视化VPR实验结果
"""

import sys
from pathlib import Path
import re
import torch
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ResultAnalyzer:
    """VPR结果分析器"""
    
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.results = []
        
    def parse_info_log(self, log_path: Path):
        """解析info.log文件"""
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取关键信息
        info = {}
        
        # 提取方法名
        method_match = re.search(r'--method\s+(\w+)', content)
        if method_match:
            info['method'] = method_match.group(1)
        
        # 提取backbone
        backbone_match = re.search(r'--backbone\s+(\w+)', content)
        if backbone_match:
            info['backbone'] = backbone_match.group(1)
        
        # 提取距离度量
        distance_match = re.search(r'--distance_metric\s+(\w+)', content)
        if distance_match:
            info['distance_metric'] = distance_match.group(1)
        else:
            info['distance_metric'] = 'l2'  # 默认
        
        # 提取数据集信息
        dataset_match = re.search(r'--database_folder\s+([^\s]+)', content)
        if dataset_match:
            dataset_path = dataset_match.group(1)
            if 'sf_xs' in dataset_path:
                info['dataset'] = 'SF-XS'
            elif 'tokyo' in dataset_path:
                info['dataset'] = 'Tokyo-XS'
            elif 'svox' in dataset_path:
                if 'sun' in dataset_path:
                    info['dataset'] = 'SVOX-Sun'
                elif 'night' in dataset_path:
                    info['dataset'] = 'SVOX-Night'
                else:
                    info['dataset'] = 'SVOX'
            else:
                info['dataset'] = 'Unknown'
        
        # 提取查询和数据库数量
        num_match = re.search(r'#queries:\s+(\d+);\s+#database:\s+(\d+)', content)
        if num_match:
            info['num_queries'] = int(num_match.group(1))
            info['num_database'] = int(num_match.group(2))
        
        # 提取Recall值
        recall_match = re.search(r'R@1:\s+([\d.]+),\s+R@5:\s+([\d.]+),\s+R@10:\s+([\d.]+),\s+R@20:\s+([\d.]+)', content)
        if recall_match:
            info['R@1'] = float(recall_match.group(1))
            info['R@5'] = float(recall_match.group(2))
            info['R@10'] = float(recall_match.group(3))
            info['R@20'] = float(recall_match.group(4))
        
        # 提取时间
        time_lines = content.split('\n')
        if len(time_lines) >= 2:
            start_time_str = time_lines[0].split()[0] + ' ' + time_lines[0].split()[1]
            end_time_str = time_lines[-2].split()[0] + ' ' + time_lines[-2].split()[1] if len(time_lines) > 1 else start_time_str
            
            try:
                start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
                end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
                info['duration_seconds'] = (end_time - start_time).total_seconds()
            except:
                info['duration_seconds'] = 0
        
        info['log_path'] = str(log_path)
        
        return info
    
    def scan_all_results(self):
        """扫描所有实验结果"""
        print("正在扫描实验结果...")
        
        # 查找所有info.log文件
        info_logs = list(self.logs_dir.rglob("*/info.log"))
        
        print(f"找到 {len(info_logs)} 个实验结果")
        
        for log_path in info_logs:
            try:
                info = self.parse_info_log(log_path)
                if 'R@1' in info:  # 只保留有结果的实验
                    self.results.append(info)
            except Exception as e:
                print(f"解析失败 {log_path}: {e}")
        
        print(f"成功解析 {len(self.results)} 个实验")
        
        return self.results
    
    def print_summary_table(self):
        """打印结果汇总表格"""
        if not self.results:
            print("没有找到实验结果！")
            return
        
        print("\n" + "="*120)
        print("实验结果汇总")
        print("="*120)
        
        # 表头
        header = f"{'方法':<15} {'Backbone':<12} {'数据集':<12} {'距离':<8} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'R@20':<8} {'时间(s)':<10}"
        print(header)
        print("-"*120)
        
        # 按数据集和方法排序
        sorted_results = sorted(self.results, key=lambda x: (x.get('dataset', ''), x.get('method', ''), x.get('distance_metric', '')))
        
        for result in sorted_results:
            method = result.get('method', 'N/A')
            backbone = result.get('backbone', 'N/A')
            dataset = result.get('dataset', 'N/A')
            distance = result.get('distance_metric', 'N/A')
            r1 = result.get('R@1', 0)
            r5 = result.get('R@5', 0)
            r10 = result.get('R@10', 0)
            r20 = result.get('R@20', 0)
            duration = result.get('duration_seconds', 0)
            
            row = f"{method:<15} {backbone:<12} {dataset:<12} {distance:<8} {r1:<8.1f} {r5:<8.1f} {r10:<8.1f} {r20:<8.1f} {duration:<10.1f}"
            print(row)
        
        print("="*120)
    
    def plot_comparison(self, save_path="results/comparison.png"):
        """绘制对比图"""
        if not self.results:
            print("没有实验结果可以绘制！")
            return
        
        # 按数据集分组
        datasets = {}
        for result in self.results:
            dataset = result.get('dataset', 'Unknown')
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(result)
        
        # 为每个数据集创建子图
        n_datasets = len(datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 6))
        
        if n_datasets == 1:
            axes = [axes]
        
        for idx, (dataset_name, dataset_results) in enumerate(datasets.items()):
            ax = axes[idx]
            
            # 准备数据
            methods = []
            r1_scores = []
            distances = []
            
            for result in dataset_results:
                method_name = f"{result.get('method', 'N/A')}\n({result.get('distance_metric', 'N/A')})"
                methods.append(method_name)
                r1_scores.append(result.get('R@1', 0))
                distances.append(result.get('distance_metric', 'l2'))
            
            # 绘制柱状图
            colors = ['steelblue' if d == 'l2' else 'coral' for d in distances]
            bars = ax.bar(range(len(methods)), r1_scores, color=colors, alpha=0.8)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10)
            
            ax.set_xlabel('方法', fontsize=12)
            ax.set_ylabel('R@1 (%)', fontsize=12)
            ax.set_title(f'{dataset_name}', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ 对比图已保存到: {save_path}")
        plt.close()
    
    def compare_distance_metrics(self):
        """比较L2 vs Dot product"""
        print("\n" + "="*80)
        print("距离度量对比（L2 vs Dot Product）")
        print("="*80)
        
        # 按方法和数据集分组
        comparisons = defaultdict(lambda: {'l2': None, 'dot_product': None})
        
        for result in self.results:
            method = result.get('method', 'N/A')
            dataset = result.get('dataset', 'N/A')
            distance = result.get('distance_metric', 'l2')
            
            key = f"{method}_{dataset}"
            comparisons[key][distance] = result
        
        # 打印对比
        print(f"\n{'方法-数据集':<30} {'L2 R@1':<12} {'Dot R@1':<12} {'差值':<10} {'更好':<10}")
        print("-"*80)
        
        for key, metrics in comparisons.items():
            if metrics['l2'] and metrics['dot_product']:
                l2_r1 = metrics['l2'].get('R@1', 0)
                dot_r1 = metrics['dot_product'].get('R@1', 0)
                diff = dot_r1 - l2_r1
                better = "Dot" if diff > 0 else "L2" if diff < 0 else "相同"
                
                print(f"{key:<30} {l2_r1:<12.1f} {dot_r1:<12.1f} {diff:+<10.1f} {better:<10}")
        
        print("="*80)
    
    def export_to_latex(self, output_file="results/table.tex"):
        """导出为LaTeX表格"""
        if not self.results:
            print("没有实验结果可以导出！")
            return
        
        latex_content = []
        latex_content.append("\\begin{table}[h]")
        latex_content.append("\\centering")
        latex_content.append("\\begin{tabular}{l|l|l|c|c|c|c}")
        latex_content.append("\\hline")
        latex_content.append("Method & Dataset & Distance & R@1 & R@5 & R@10 & R@20 \\\\")
        latex_content.append("\\hline")
        
        for result in sorted(self.results, key=lambda x: (x.get('dataset', ''), x.get('method', ''))):
            method = result.get('method', 'N/A')
            dataset = result.get('dataset', 'N/A')
            distance = result.get('distance_metric', 'N/A')
            r1 = result.get('R@1', 0)
            r5 = result.get('R@5', 0)
            r10 = result.get('R@10', 0)
            r20 = result.get('R@20', 0)
            
            latex_content.append(f"{method} & {dataset} & {distance} & {r1:.1f} & {r5:.1f} & {r10:.1f} & {r20:.1f} \\\\")
        
        latex_content.append("\\hline")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\caption{VPR Baseline Results}")
        latex_content.append("\\label{tab:baseline_results}")
        latex_content.append("\\end{table}")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_content))
        
        print(f"\n✅ LaTeX表格已导出到: {output_path}")
    
    def export_to_json(self, output_file="results/all_results.json"):
        """导出为JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON结果已导出到: {output_path}")


def main():
    """主函数"""
    print("="*80)
    print("VPR实验结果分析工具")
    print("="*80)
    
    # 创建分析器
    analyzer = ResultAnalyzer(logs_dir="logs")
    
    # 扫描所有结果
    analyzer.scan_all_results()
    
    if not analyzer.results:
        print("\n⚠️  没有找到实验结果！")
        print("请先运行实验：python run_baseline_experiments.py")
        return
    
    # 打印汇总表格
    analyzer.print_summary_table()
    
    # 比较距离度量
    analyzer.compare_distance_metrics()
    
    # 绘制对比图
    analyzer.plot_comparison()
    
    # 导出结果
    analyzer.export_to_json()
    analyzer.export_to_latex()
    
    print("\n" + "="*80)
    print("✅ 分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()

