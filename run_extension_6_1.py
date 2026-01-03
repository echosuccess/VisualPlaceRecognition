#!/usr/bin/env python3
"""
Extension 6.1 自动化实验脚本
完整流程：数据准备 → 训练 → 评估 → 可视化
"""

import sys
import subprocess
from pathlib import Path
import numpy as np
import torch
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from extension_6_1_implementation import AdaptiveReranking, plot_threshold_analysis

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Extension61Pipeline:
    """Extension 6.1 完整实验流程"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.checkpoint_path = Path("checkpoints/extension_6_1.json")
        
    def step1_check_prerequisites(self):
        """检查前置条件：VPR结果是否存在"""
        print("\n" + "="*80)
        print("步骤1：检查前置条件")
        print("="*80)
        
        missing_files = []
        
        for vpr_method in self.config['vpr_methods']:
            for dataset in self.config['datasets']:
                # 检查VPR预测结果
                preds_dir = Path(f"logs/baseline/{vpr_method}_l2_{dataset['name']}")
                z_data_files = list(preds_dir.glob("**/z_data.torch"))
                
                if not z_data_files:
                    missing_files.append(f"VPR预测: {preds_dir}")
        
        if missing_files:
            print("\n❌ 缺少以下文件，请先运行baseline实验：")
            for f in missing_files:
                print(f"   - {f}")
            return False
        
        print("\n✅ 所有前置条件已满足")
        return True
    
    def step2_run_image_matching(self):
        """运行Image Matching获取inliers数据"""
        print("\n" + "="*80)
        print("步骤2：运行Image Matching（如果需要）")
        print("="*80)
        
        for vpr_method in self.config['vpr_methods']:
            for im_method in self.config['im_methods']:
                for dataset in self.config['datasets']:
                    
                    # 检查是否已经运行过
                    inliers_dir = Path(
                        f"logs/inliers/{vpr_method}_{im_method}_{dataset['name']}"
                    )
                    
                    if (inliers_dir / "inliers.npy").exists():
                        print(f"⏭️  跳过（已存在）: {vpr_method} + {im_method} + {dataset['name']}")
                        continue
                    
                    print(f"\n🔄 运行: {vpr_method} + {im_method} + {dataset['name']}")
                    
                    # 构建命令
                    preds_dir = Path(f"logs/baseline/{vpr_method}_l2_{dataset['name']}")
                    preds_folder = list(preds_dir.glob("**/predictions"))[0]
                    
                    cmd = [
                        "python", "match_queries_preds.py",
                        "--preds-dir", str(preds_folder),
                        "--matcher", im_method,
                        "--device", "cuda",
                        "--num-preds", "20",
                        "--output-dir", str(inliers_dir)
                    ]
                    
                    try:
                        subprocess.run(cmd, check=True)
                        print(f"✅ 完成")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ 失败: {e}")
                        return False
        
        print("\n✅ Image Matching完成")
        return True
    
    def step3_prepare_data(self):
        """准备训练/验证/测试数据"""
        print("\n" + "="*80)
        print("步骤3：准备数据")
        print("="*80)
        
        data_splits = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for vpr_method in self.config['vpr_methods']:
            for im_method in self.config['im_methods']:
                
                # 训练集：SVOX (排除GSV-XS)
                for train_dataset in ['svox_sun', 'svox_night']:
                    preds_dir = f"logs/baseline/{vpr_method}_l2_{train_dataset}"
                    inliers_dir = f"logs/inliers/{vpr_method}_{im_method}_{train_dataset}"
                    
                    if Path(preds_dir).exists() and Path(inliers_dir).exists():
                        data_splits['train'].append({
                            'vpr': vpr_method,
                            'im': im_method,
                            'dataset': train_dataset,
                            'preds_dir': preds_dir,
                            'inliers_dir': inliers_dir
                        })
                
                # 验证集：SF-XS val
                val_dataset = 'sfxs_val'
                preds_dir = f"logs/baseline/{vpr_method}_l2_{val_dataset}"
                inliers_dir = f"logs/inliers/{vpr_method}_{im_method}_{val_dataset}"
                
                if Path(preds_dir).exists() and Path(inliers_dir).exists():
                    data_splits['val'].append({
                        'vpr': vpr_method,
                        'im': im_method,
                        'dataset': val_dataset,
                        'preds_dir': preds_dir,
                        'inliers_dir': inliers_dir
                    })
                
                # 测试集：SF-XS test, Tokyo-XS, SVOX test
                for test_dataset in ['sfxs_test', 'tokyo_xs', 'svox_test']:
                    preds_dir = f"logs/baseline/{vpr_method}_l2_{test_dataset}"
                    inliers_dir = f"logs/inliers/{vpr_method}_{im_method}_{test_dataset}"
                    
                    if Path(preds_dir).exists() and Path(inliers_dir).exists():
                        data_splits['test'].append({
                            'vpr': vpr_method,
                            'im': im_method,
                            'dataset': test_dataset,
                            'preds_dir': preds_dir,
                            'inliers_dir': inliers_dir
                        })
        
        print(f"训练集样本数: {len(data_splits['train'])}")
        print(f"验证集样本数: {len(data_splits['val'])}")
        print(f"测试集样本数: {len(data_splits['test'])}")
        
        self.data_splits = data_splits
        return True
    
    def step4_train_threshold(self):
        """训练硬阈值模型"""
        print("\n" + "="*80)
        print("步骤4：训练硬阈值模型")
        print("="*80)
        
        for vpr_method in self.config['vpr_methods']:
            for im_method in self.config['im_methods']:
                
                print(f"\n🔧 训练: {vpr_method} + {im_method}")
                
                # 创建模型
                adaptive = AdaptiveReranking(method="threshold")
                
                # 加载训练数据
                train_data_list = [
                    d for d in self.data_splits['train']
                    if d['vpr'] == vpr_method and d['im'] == im_method
                ]
                
                if not train_data_list:
                    print(f"⚠️  没有训练数据，跳过")
                    continue
                
                # 合并训练数据
                train_data = self._merge_data(adaptive, train_data_list)
                
                # 加载验证数据
                val_data_list = [
                    d for d in self.data_splits['val']
                    if d['vpr'] == vpr_method and d['im'] == im_method
                ]
                
                if not val_data_list:
                    print(f"⚠️  没有验证数据，跳过")
                    continue
                
                val_data = adaptive.load_data(
                    Path(val_data_list[0]['preds_dir']),
                    Path(val_data_list[0]['inliers_dir'])
                )
                
                # 训练
                threshold_results = adaptive.fit_threshold(train_data, val_data)
                
                # 保存模型
                model_dir = Path(f"models/extension_6_1/threshold/{vpr_method}_{im_method}")
                adaptive.save(model_dir)
                
                # 绘图
                plot_path = Path(f"results/extension_6_1/threshold_analysis_{vpr_method}_{im_method}.png")
                plot_threshold_analysis(threshold_results, plot_path)
                
                # 保存结果
                self.results[f"{vpr_method}_{im_method}_threshold"] = {
                    'method': 'threshold',
                    'vpr': vpr_method,
                    'im': im_method,
                    'threshold': adaptive.threshold,
                    'threshold_results': threshold_results
                }
        
        print("\n✅ 硬阈值模型训练完成")
        return True
    
    def step5_train_logistic(self):
        """训练逻辑回归模型"""
        print("\n" + "="*80)
        print("步骤5：训练逻辑回归模型")
        print("="*80)
        
        for vpr_method in self.config['vpr_methods']:
            for im_method in self.config['im_methods']:
                
                print(f"\n🔧 训练: {vpr_method} + {im_method}")
                
                # 创建模型
                adaptive = AdaptiveReranking(method="logistic")
                
                # 加载训练数据
                train_data_list = [
                    d for d in self.data_splits['train']
                    if d['vpr'] == vpr_method and d['im'] == im_method
                ]
                
                if not train_data_list:
                    print(f"⚠️  没有训练数据，跳过")
                    continue
                
                train_data = self._merge_data(adaptive, train_data_list)
                
                # 加载验证数据
                val_data_list = [
                    d for d in self.data_splits['val']
                    if d['vpr'] == vpr_method and d['im'] == im_method
                ]
                
                if not val_data_list:
                    print(f"⚠️  没有验证数据，跳过")
                    continue
                
                val_data = adaptive.load_data(
                    Path(val_data_list[0]['preds_dir']),
                    Path(val_data_list[0]['inliers_dir'])
                )
                
                # 训练
                lr_results = adaptive.fit_logistic(train_data, val_data)
                
                # 保存模型
                model_dir = Path(f"models/extension_6_1/logistic/{vpr_method}_{im_method}")
                adaptive.save(model_dir)
                
                # 保存结果
                self.results[f"{vpr_method}_{im_method}_logistic"] = {
                    'method': 'logistic',
                    'vpr': vpr_method,
                    'im': im_method,
                    'lr_results': lr_results
                }
        
        print("\n✅ 逻辑回归模型训练完成")
        return True
    
    def step6_evaluate(self):
        """在测试集上评估"""
        print("\n" + "="*80)
        print("步骤6：在测试集上评估")
        print("="*80)
        
        test_results = []
        
        for key, model_info in self.results.items():
            vpr_method = model_info['vpr']
            im_method = model_info['im']
            method = model_info['method']
            
            print(f"\n📊 评估: {vpr_method} + {im_method} ({method})")
            
            # 加载模型
            model_dir = Path(f"models/extension_6_1/{method}/{vpr_method}_{im_method}")
            adaptive = AdaptiveReranking(method=method)
            adaptive.load(model_dir)
            
            # 在每个测试集上评估
            test_data_list = [
                d for d in self.data_splits['test']
                if d['vpr'] == vpr_method and d['im'] == im_method
            ]
            
            for test_data_info in test_data_list:
                test_data = adaptive.load_data(
                    Path(test_data_info['preds_dir']),
                    Path(test_data_info['inliers_dir'])
                )
                
                eval_results = adaptive.evaluate(test_data)
                
                test_results.append({
                    'vpr': vpr_method,
                    'im': im_method,
                    'method': method,
                    'dataset': test_data_info['dataset'],
                    **eval_results
                })
                
                print(f"  {test_data_info['dataset']}:")
                print(f"    R@1: {eval_results['r1_without_rerank']:.2%}")
                print(f"    Re-rank比例: {eval_results['rerank_ratio']:.2%}")
                print(f"    成本节省: {eval_results['cost_saving']:.2%}")
        
        # 保存结果
        results_path = Path("results/extension_6_1/test_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n✅ 测试结果已保存到: {results_path}")
        
        self.test_results = test_results
        return True
    
    def step7_visualize(self):
        """生成可视化结果"""
        print("\n" + "="*80)
        print("步骤7：生成可视化")
        print("="*80)
        
        # 绘制对比图
        self._plot_comparison()
        self._plot_cost_savings()
        
        print("\n✅ 可视化完成")
        return True
    
    def _merge_data(self, adaptive, data_list):
        """合并多个数据集"""
        all_inliers = []
        all_is_correct = []
        all_predictions = []
        all_positives = []
        
        for data_info in data_list:
            data = adaptive.load_data(
                Path(data_info['preds_dir']),
                Path(data_info['inliers_dir'])
            )
            all_inliers.append(data['inliers'])
            all_is_correct.append(data['is_correct'])
            all_predictions.append(data['predictions'])
            all_positives.extend(data['positives_per_query'])
        
        return {
            'inliers': np.concatenate(all_inliers),
            'is_correct': np.concatenate(all_is_correct),
            'predictions': np.concatenate(all_predictions),
            'positives_per_query': all_positives
        }
    
    def _plot_comparison(self):
        """绘制方法对比图"""
        if not hasattr(self, 'test_results'):
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 子图1：R@1对比
        methods = [r['method'] for r in self.test_results]
        r1_scores = [r['r1_without_rerank'] for r in self.test_results]
        
        axes[0].bar(range(len(methods)), r1_scores)
        axes[0].set_xlabel('Method')
        axes[0].set_ylabel('R@1')
        axes[0].set_title('R@1 Comparison')
        axes[0].set_xticks(range(len(methods)))
        axes[0].set_xticklabels(methods, rotation=45, ha='right')
        
        # 子图2：成本节省
        cost_savings = [r['cost_saving'] for r in self.test_results]
        
        axes[1].bar(range(len(methods)), cost_savings, color='green')
        axes[1].set_xlabel('Method')
        axes[1].set_ylabel('Cost Saving')
        axes[1].set_title('Cost Saving Comparison')
        axes[1].set_xticks(range(len(methods)))
        axes[1].set_xticklabels(methods, rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = Path("results/extension_6_1/comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 对比图已保存: {save_path}")
        plt.close()
    
    def _plot_cost_savings(self):
        """绘制成本节省详细分析"""
        # TODO: 实现更详细的成本分析图
        pass
    
    def save_checkpoint(self, completed_steps):
        """保存checkpoint"""
        checkpoint_data = {
            'completed_steps': completed_steps,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"[CHECKPOINT] 已保存: {self.checkpoint_path}")
    
    def load_checkpoint(self):
        """加载checkpoint"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                completed_steps = checkpoint_data.get('completed_steps', [])
                self.results = checkpoint_data.get('results', {})
                print(f"[CHECKPOINT] 从checkpoint恢复: {len(completed_steps)} 个已完成步骤")
                return completed_steps
            except Exception as e:
                print(f"[WARN] 无法加载checkpoint: {e}")
        return []
    
    def run_full_pipeline(self):
        """运行完整流程"""
        print("\n" + "="*80)
        print("Extension 6.1 完整实验流程")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # 加载checkpoint
        completed_steps = self.load_checkpoint()
        
        steps = [
            ("检查前置条件", self.step1_check_prerequisites),
            ("运行Image Matching", self.step2_run_image_matching),
            ("准备数据", self.step3_prepare_data),
            ("训练硬阈值模型", self.step4_train_threshold),
            ("训练逻辑回归模型", self.step5_train_logistic),
            ("测试集评估", self.step6_evaluate),
            ("生成可视化", self.step7_visualize),
        ]
        
        for step_name, step_func in steps:
            # 检查是否已完成
            if step_name in completed_steps:
                print(f"\n{'='*80}")
                print(f"⏭️  跳过（已完成）: {step_name}")
                print(f"{'='*80}")
                continue
            
            print(f"\n{'='*80}")
            print(f"执行: {step_name}")
            print(f"{'='*80}")
            
            success = step_func()
            
            if not success:
                print(f"\n❌ {step_name} 失败，停止流程")
                print(f"[CHECKPOINT] 已保存当前进度，可以重新运行以继续")
                return False
            
            # 标记为完成并保存checkpoint
            completed_steps.append(step_name)
            self.save_checkpoint(completed_steps)
            print(f"[CHECKPOINT] 步骤 '{step_name}' 完成，已保存checkpoint")
        
        print("\n" + "="*80)
        print("✅ Extension 6.1 完整流程完成！")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # 清理checkpoint
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            print(f"[INFO] 已清理checkpoint文件（所有步骤已完成）")
        
        return True


def main():
    """主函数"""
    
    # 配置
    config = {
        'vpr_methods': ['cosplace', 'mixvpr'],  # 选择2个VPR方法
        'im_methods': ['superpoint-lg', 'loftr'],  # 选择2个IM方法
        'datasets': [
            {'name': 'svox_sun', 'type': 'train'},
            {'name': 'svox_night', 'type': 'train'},
            {'name': 'sfxs_val', 'type': 'val'},
            {'name': 'sfxs_test', 'type': 'test'},
            {'name': 'tokyo_xs', 'type': 'test'},
        ]
    }
    
    # 创建pipeline
    pipeline = Extension61Pipeline(config)
    
    # 运行
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()

