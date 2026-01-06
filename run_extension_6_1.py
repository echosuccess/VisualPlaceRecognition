#!/usr/bin/env python3
"""
Extension 6.1 automated experiment script
Full pipeline: data preparation -> training -> evaluation -> visualization
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

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Extension61Pipeline:
    """Extension 6.1 full experiment pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.checkpoint_path = Path("checkpoints/extension_6_1.json")
        
    def step1_check_prerequisites(self):
        """Check prerequisites: VPR results and Image Matching JSON files"""
        print("\n" + "="*80)
        print("Step 1: Check prerequisites")
        print("="*80)
        
        dataset_mapping = {
            'svox_sun': 'svox_sun_test',
            'svox_night': 'svox_night_test',
            'sfxs_test': 'sf_xs_test',
            'tokyo_xs': 'tokyo_xs_test',
            'sfxs_val': 'sfxs_val'
        }
        
        matcher_mapping = {
            'superpoint-lg': 'superpoint-lg',
            'loftr': 'loftr',
            'superglue': 'superglue'
        }
        
        results_dir = Path("results/image_matching")
        if not results_dir.exists():
            results_dir = Path("mydoc/image_matching")
        
        missing_files = []
        
        for vpr_method in self.config['vpr_methods']:
            for im_method in self.config['im_methods']:
                for dataset in self.config['datasets']:
                    dataset_name = dataset_mapping.get(dataset['name'], dataset['name'])
                    matcher_name = matcher_mapping.get(im_method, im_method)
                    
                    preds_dir_l2 = Path(f"logs/baseline/{vpr_method}_l2_{dataset_name}")
                    preds_dir_dot = Path(f"logs/baseline/{vpr_method}_dot_product_{dataset_name}")
                    z_data_files_l2 = list(preds_dir_l2.glob("**/z_data.torch")) if preds_dir_l2.exists() else []
                    z_data_files_dot = list(preds_dir_dot.glob("**/z_data.torch")) if preds_dir_dot.exists() else []
                    
                    if not z_data_files_l2 and not z_data_files_dot:
                        missing_files.append(f"VPR predictions: {vpr_method} + {dataset_name}")
                    
                    json_file_l2 = results_dir / f"{matcher_name}_{vpr_method}_l2_{dataset_name}.json"
                    json_file_dot = results_dir / f"{matcher_name}_{vpr_method}_dot_product_{dataset_name}.json"
                    
                    if not json_file_l2.exists() and not json_file_dot.exists():
                        missing_files.append(f"Image Matching JSON: {matcher_name} + {vpr_method} + {dataset_name}")
        
        if missing_files:
            print("\n[ERROR] Missing files, please run baseline experiments first:")
            for f in missing_files:
                print(f"   - {f}")
            return False
        
        print("\n[OK] All prerequisites satisfied")
        return True
    
    def step2_run_image_matching(self):
        """Run Image Matching to get inliers data (skipped, JSON files already exist)"""
        print("\n" + "="*80)
        print("Step 2: Run Image Matching (if needed)")
        print("="*80)
        
        print("\n[SKIP] Image Matching JSON files already exist, no need to re-run")
        print("   If JSON files don't exist, please run run_image_matching_baseline.py first")
        
        return True
    
    def step3_prepare_data(self):
        """Prepare train/val/test data (read from JSON files)"""
        print("\n" + "="*80)
        print("Step 3: Prepare data")
        print("="*80)
        
        dataset_mapping = {
            'svox_sun': 'svox_sun_test',
            'svox_night': 'svox_night_test',
            'sfxs_test': 'sf_xs_test',
            'tokyo_xs': 'tokyo_xs_test',
            'svox_test': 'svox_sun_test',
            'sfxs_val': 'sfxs_val'
        }
        
        matcher_mapping = {
            'superpoint-lg': 'superpoint-lg',
            'loftr': 'loftr',
            'superglue': 'superglue'
        }
        
        data_splits = {
            'train': [],
            'val': [],
            'test': []
        }
        
        results_dir = Path("results/image_matching")
        if not results_dir.exists():
            results_dir = Path("mydoc/image_matching")
        
        for vpr_method in self.config['vpr_methods']:
            for im_method in self.config['im_methods']:
                matcher_name = matcher_mapping.get(im_method, im_method)
                
                # Training set: SVOX (svox_sun_test, svox_night_test)
                for train_dataset_code in ['svox_sun', 'svox_night']:
                    train_dataset = dataset_mapping.get(train_dataset_code, train_dataset_code)
                    
                    preds_dir_l2 = Path(f"logs/baseline/{vpr_method}_l2_{train_dataset}")
                    preds_dir_dot = Path(f"logs/baseline/{vpr_method}_dot_product_{train_dataset}")
                    preds_dir = preds_dir_l2 if preds_dir_l2.exists() else (preds_dir_dot if preds_dir_dot.exists() else None)
                    
                    json_file_l2 = results_dir / f"{matcher_name}_{vpr_method}_l2_{train_dataset}.json"
                    json_file_dot = results_dir / f"{matcher_name}_{vpr_method}_dot_product_{train_dataset}.json"
                    json_file = json_file_l2 if json_file_l2.exists() else (json_file_dot if json_file_dot.exists() else None)
                    
                    if preds_dir and json_file and json_file.exists():
                        data_splits['train'].append({
                            'vpr': vpr_method,
                            'im': im_method,
                            'dataset': train_dataset_code,
                            'preds_dir': str(preds_dir),
                            'im_result_json': str(json_file)
                        })
                
                # Validation set: SF-XS val
                val_dataset_code = 'sfxs_val'
                val_dataset = dataset_mapping.get(val_dataset_code, val_dataset_code)
                
                preds_dir_l2 = Path(f"logs/baseline/{vpr_method}_l2_{val_dataset}")
                preds_dir = preds_dir_l2 if preds_dir_l2.exists() else None
                
                json_file = results_dir / f"{matcher_name}_{vpr_method}_l2_{val_dataset}.json"
                
                if preds_dir and json_file and json_file.exists():
                    data_splits['val'].append({
                        'vpr': vpr_method,
                        'im': im_method,
                        'dataset': val_dataset_code,
                        'preds_dir': str(preds_dir),
                        'im_result_json': str(json_file)
                    })
                
                # Test set: SF-XS test, Tokyo-XS, SVOX test
                for test_dataset_code in ['sfxs_test', 'tokyo_xs']:
                    test_dataset = dataset_mapping.get(test_dataset_code, test_dataset_code)
                    
                    preds_dir_l2 = Path(f"logs/baseline/{vpr_method}_l2_{test_dataset}")
                    preds_dir_dot = Path(f"logs/baseline/{vpr_method}_dot_product_{test_dataset}")
                    preds_dir = preds_dir_l2 if preds_dir_l2.exists() else (preds_dir_dot if preds_dir_dot.exists() else None)
                    
                    json_file_l2 = results_dir / f"{matcher_name}_{vpr_method}_l2_{test_dataset}.json"
                    json_file_dot = results_dir / f"{matcher_name}_{vpr_method}_dot_product_{test_dataset}.json"
                    json_file = json_file_l2 if json_file_l2.exists() else (json_file_dot if json_file_dot.exists() else None)
                    
                    if preds_dir and json_file and json_file.exists():
                        data_splits['test'].append({
                            'vpr': vpr_method,
                            'im': im_method,
                            'dataset': test_dataset_code,
                            'preds_dir': str(preds_dir),
                            'im_result_json': str(json_file)
                        })
        
        print(f"Training samples: {len(data_splits['train'])}")
        print(f"Validation samples: {len(data_splits['val'])}")
        print(f"Test samples: {len(data_splits['test'])}")
        
        self.data_splits = data_splits
        return True
    
    def step4_train_threshold(self):
        """Train hard threshold model"""
        print("\n" + "="*80)
        print("Step 4: Train hard threshold model")
        print("="*80)
        
        if not hasattr(self, 'data_splits'):
            print("[INFO] data_splits not initialized, re-running step3_prepare_data...")
            if not self.step3_prepare_data():
                print("[ERROR] Failed to prepare data")
                return False
        
        for vpr_method in self.config['vpr_methods']:
            for im_method in self.config['im_methods']:
                
                print(f"\nTraining: {vpr_method} + {im_method}")
                
                adaptive = AdaptiveReranking(method="threshold")
                
                train_data_list = [
                    d for d in self.data_splits['train']
                    if d['vpr'] == vpr_method and d['im'] == im_method
                ]
                
                if not train_data_list:
                    print(f"[WARN] No training data, skipping")
                    continue
                
                train_data = self._merge_data(adaptive, train_data_list)
                
                val_data_list = [
                    d for d in self.data_splits['val']
                    if d['vpr'] == vpr_method and d['im'] == im_method
                ]
                
                if not val_data_list:
                    print(f"[WARN] No validation data, skipping")
                    continue
                
                val_data = adaptive.load_data(
                    Path(val_data_list[0]['preds_dir']),
                    im_result_json=Path(val_data_list[0]['im_result_json'])
                )
                
                threshold_results = adaptive.fit_threshold(train_data, val_data)
                
                model_dir = Path(f"models/extension_6_1/threshold/{vpr_method}_{im_method}")
                adaptive.save(model_dir)
                
                plot_path = Path(f"results/extension_6_1/threshold_analysis_{vpr_method}_{im_method}.png")
                plot_threshold_analysis(threshold_results, plot_path)
                
                self.results[f"{vpr_method}_{im_method}_threshold"] = {
                    'method': 'threshold',
                    'vpr': vpr_method,
                    'im': im_method,
                    'threshold': adaptive.threshold,
                    'threshold_results': threshold_results
                }
        
        print("\n[OK] Hard threshold model training completed")
        return True
    
    def step5_train_logistic(self):
        """Train logistic regression model"""
        print("\n" + "="*80)
        print("Step 5: Train logistic regression model")
        print("="*80)
        
        if not hasattr(self, 'data_splits'):
            print("[INFO] data_splits not initialized, re-running step3_prepare_data...")
            if not self.step3_prepare_data():
                print("[ERROR] Failed to prepare data")
                return False
        
        for vpr_method in self.config['vpr_methods']:
            for im_method in self.config['im_methods']:
                
                print(f"\nTraining: {vpr_method} + {im_method}")
                
                adaptive = AdaptiveReranking(method="logistic")
                
                train_data_list = [
                    d for d in self.data_splits['train']
                    if d['vpr'] == vpr_method and d['im'] == im_method
                ]
                
                if not train_data_list:
                    print(f"[WARN] No training data, skipping")
                    continue
                
                train_data = self._merge_data(adaptive, train_data_list)
                
                val_data_list = [
                    d for d in self.data_splits['val']
                    if d['vpr'] == vpr_method and d['im'] == im_method
                ]
                
                if not val_data_list:
                    print(f"[WARN] No validation data, skipping")
                    continue
                
                val_data = adaptive.load_data(
                    Path(val_data_list[0]['preds_dir']),
                    im_result_json=Path(val_data_list[0]['im_result_json'])
                )
                
                lr_results = adaptive.fit_logistic(train_data, val_data)
                
                model_dir = Path(f"models/extension_6_1/logistic/{vpr_method}_{im_method}")
                adaptive.save(model_dir)
                
                self.results[f"{vpr_method}_{im_method}_logistic"] = {
                    'method': 'logistic',
                    'vpr': vpr_method,
                    'im': im_method,
                    'lr_results': lr_results
                }
        
        print("\n[OK] Logistic regression model training completed")
        return True
    
    def step6_evaluate(self):
        """Evaluate on test sets"""
        print("\n" + "="*80)
        print("Step 6: Evaluate on test sets")
        print("="*80)
        
        if not hasattr(self, 'data_splits'):
            print("[INFO] data_splits not initialized, re-running step3_prepare_data...")
            if not self.step3_prepare_data():
                print("[ERROR] Failed to prepare data")
                return False
        
        test_results = []
        
        for key, model_info in self.results.items():
            vpr_method = model_info['vpr']
            im_method = model_info['im']
            method = model_info['method']
            
            print(f"\nEvaluating: {vpr_method} + {im_method} ({method})")
            
            model_dir = Path(f"models/extension_6_1/{method}/{vpr_method}_{im_method}")
            adaptive = AdaptiveReranking(method=method)
            adaptive.load(model_dir)
            
            test_data_list = [
                d for d in self.data_splits['test']
                if d['vpr'] == vpr_method and d['im'] == im_method
            ]
            
            for test_data_info in test_data_list:
                test_data = adaptive.load_data(
                    Path(test_data_info['preds_dir']),
                    im_result_json=Path(test_data_info['im_result_json'])
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
                print(f"    R@1 (no rerank): {eval_results['r1_without_rerank']:.2%}")
                if 'r1_with_rerank' in eval_results:
                    print(f"    R@1 (with rerank): {eval_results['r1_with_rerank']:.2%}")
                    improvement = eval_results['r1_with_rerank'] - eval_results['r1_without_rerank']
                    print(f"    Improvement: {improvement:+.2%}")
                print(f"    Re-rank ratio: {eval_results['rerank_ratio']:.2%}")
                print(f"    Cost saving: {eval_results['cost_saving']:.2%}")
        
        results_path = Path("results/extension_6_1/test_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n[OK] Test results saved to: {results_path}")
        
        self.test_results = test_results
        return True
    
    def step7_visualize(self):
        """Generate visualization results"""
        print("\n" + "="*80)
        print("Step 7: Generate visualizations")
        print("="*80)
        
        self._plot_comparison()
        self._plot_cost_savings()
        
        print("\n[OK] Visualization completed")
        return True
    
    def _merge_data(self, adaptive, data_list):
        """Merge multiple datasets"""
        all_inliers = []
        all_is_correct = []
        all_predictions = []
        all_positives = []
        
        for data_info in data_list:
            data = adaptive.load_data(
                Path(data_info['preds_dir']),
                im_result_json=Path(data_info['im_result_json'])
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
        """Plot method comparison"""
        if not hasattr(self, 'test_results'):
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Create labels with full experiment configuration
        labels = []
        r1_scores_with_rerank = []
        r1_scores_without_rerank = []
        cost_savings = []
        colors_r1 = []
        colors_cost = []
        
        for r in self.test_results:
            # Create descriptive label: "vpr+im\ndataset (method)"
            label = f"{r['vpr']}+{r['im']}\n{r['dataset']} ({r['method']})"
            labels.append(label)
            
            # Use R@1 with rerank for comparison (shows improvement)
            r1_scores_with_rerank.append(r.get('r1_with_rerank', r.get('r1_without_rerank', 0)))
            r1_scores_without_rerank.append(r.get('r1_without_rerank', 0))
            cost_savings.append(r.get('cost_saving', 0))
            
            # Color coding: blue for threshold, orange for logistic
            if r['method'] == 'threshold':
                colors_r1.append('#1f77b4')  # blue
                colors_cost.append('#2ca02c')  # green
            else:
                colors_r1.append('#ff7f0e')  # orange
                colors_cost.append('#ff7f0e')  # orange
        
        # Left plot: R@1 Comparison (with rerank)
        x_pos = np.arange(len(labels))
        bars1 = axes[0].bar(x_pos, r1_scores_with_rerank, color=colors_r1, alpha=0.7)
        axes[0].set_xlabel('Experiment Configuration', fontsize=11)
        axes[0].set_ylabel('R@1 (with rerank)', fontsize=11)
        axes[0].set_title('R@1 Comparison: Threshold vs Logistic Regression', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, max(r1_scores_with_rerank) * 1.1])
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars1, r1_scores_with_rerank)):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.2f}',
                        ha='center', va='bottom', fontsize=8)
        
        # Right plot: Cost Saving Comparison
        bars2 = axes[1].bar(x_pos, cost_savings, color=colors_cost, alpha=0.7)
        axes[1].set_xlabel('Experiment Configuration', fontsize=11)
        axes[1].set_ylabel('Cost Saving', fontsize=11)
        axes[1].set_title('Cost Saving Comparison: Threshold vs Logistic Regression', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_ylim([0, max(cost_savings) * 1.1 if max(cost_savings) > 0 else 0.1])
        
        # Add value labels on bars
        for i, (bar, saving) in enumerate(zip(bars2, cost_savings)):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{saving:.2f}',
                        ha='center', va='bottom', fontsize=8)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.7, label='Threshold Method'),
            Patch(facecolor='#ff7f0e', alpha=0.7, label='Logistic Regression Method')
        ]
        axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9)
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        save_path = Path("results/extension_6_1/comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Comparison plot saved: {save_path}")
        plt.close()
    
    def _plot_cost_savings(self):
        """Plot detailed cost savings analysis"""
        pass
    
    def save_checkpoint(self, completed_steps):
        """Save checkpoint"""
        checkpoint_data = {
            'completed_steps': completed_steps,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"[CHECKPOINT] Saved: {self.checkpoint_path}")
    
    def load_checkpoint(self):
        """Load checkpoint"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                completed_steps = checkpoint_data.get('completed_steps', [])
                self.results = checkpoint_data.get('results', {})
                print(f"[CHECKPOINT] Resumed from checkpoint: {len(completed_steps)} completed steps")
                return completed_steps
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint: {e}")
        return []
    
    def run_full_pipeline(self):
        """Run full pipeline"""
        print("\n" + "="*80)
        print("Extension 6.1 Full Experiment Pipeline")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        completed_steps = self.load_checkpoint()
        
        steps = [
            ("Check prerequisites", self.step1_check_prerequisites),
            ("Run Image Matching", self.step2_run_image_matching),
            ("Prepare data", self.step3_prepare_data),
            ("Train hard threshold model", self.step4_train_threshold),
            ("Train logistic regression model", self.step5_train_logistic),
            ("Evaluate on test sets", self.step6_evaluate),
            ("Generate visualizations", self.step7_visualize),
        ]
        
        for step_name, step_func in steps:
            if step_name in completed_steps:
                print(f"\n{'='*80}")
                print(f"[SKIP] Already completed: {step_name}")
                print(f"{'='*80}")
                continue
            
            print(f"\n{'='*80}")
            print(f"Executing: {step_name}")
            print(f"{'='*80}")
            
            success = step_func()
            
            if not success:
                print(f"\n[ERROR] {step_name} failed, stopping pipeline")
                print(f"[CHECKPOINT] Current progress saved, can resume by re-running")
                return False
            
            completed_steps.append(step_name)
            self.save_checkpoint(completed_steps)
            print(f"[CHECKPOINT] Step '{step_name}' completed, checkpoint saved")
        
        print("\n" + "="*80)
        print("[OK] Extension 6.1 full pipeline completed!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            print(f"[INFO] Checkpoint file cleaned up (all steps completed)")
        
        return True


def main():
    """Main function"""
    
    config = {
        'vpr_methods': ['cosplace', 'mixvpr'],
        'im_methods': ['superpoint-lg', 'loftr'],
        'datasets': [
            {'name': 'svox_sun', 'type': 'train'},
            {'name': 'svox_night', 'type': 'train'},
            {'name': 'sfxs_val', 'type': 'val'},
            {'name': 'sfxs_test', 'type': 'test'},
            {'name': 'tokyo_xs', 'type': 'test'},
        ]
    }
    
    pipeline = Extension61Pipeline(config)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()

