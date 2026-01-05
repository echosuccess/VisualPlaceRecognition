#!/usr/bin/env python3
"""
Extension 6.1: Adaptive Re-ranking Implementation
自适应Re-ranking实现

核心思路：
- 只对"困难"查询应用re-ranking
- "困难"定义：inliers数量低于阈值
- 两种方法：硬阈值 or 逻辑回归
"""

import numpy as np
import torch
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
import pickle


class AdaptiveReranking:
    """自适应Re-ranking系统"""
    
    def __init__(self, method="threshold"):
        """
        Args:
            method: "threshold" 或 "logistic"
        """
        self.method = method
        self.threshold = None
        self.logistic_model = None
        
    def load_data(self, preds_dir: Path, im_result_json: Path = None, inliers_dir: Path = None):
        """
        加载VPR预测和inliers数据
        
        Args:
            preds_dir: VPR预测结果目录（包含z_data.torch）
            im_result_json: Image Matching JSON结果文件路径（新格式，优先使用）
            inliers_dir: Image Matching结果目录（包含inliers.npy，旧格式，兼容用）
        """
        # 加载VPR预测数据
        z_data_path = list(Path(preds_dir).glob("**/z_data.torch"))[0]
        z_data = torch.load(z_data_path)
        
        predictions = z_data['predictions']
        positives_per_query = z_data['positives_per_query']
        database_utms = z_data['database_utms']
        
        # 优先从JSON文件加载inliers（新格式）
        if im_result_json and Path(im_result_json).exists():
            with open(im_result_json, 'r', encoding='utf-8') as f:
                im_data = json.load(f)
            
            # 从JSON提取inliers（只取top-1，即pred_rank=0的）
            num_inliers = im_data['results']['num_inliers']
            pred_ranks = im_data['results']['pred_ranks']
            is_correct_json = im_data['results']['is_correct']
            
            # 只取top-1预测的inliers（pred_rank=0）
            top1_inliers = []
            top1_is_correct = []
            current_query_idx = None
            
            for i, (inlier_count, rank, correct) in enumerate(zip(num_inliers, pred_ranks, is_correct_json)):
                # 假设query_ids是连续的，从0开始
                # 如果pred_rank=0，说明这是top-1预测
                if rank == 0:
                    top1_inliers.append(inlier_count)
                    top1_is_correct.append(correct)
            
            inliers = np.array(top1_inliers, dtype=np.float64)
            is_correct = np.array(top1_is_correct, dtype=bool)
            
        # 兼容旧格式：从inliers.npy加载
        elif inliers_dir and Path(inliers_dir).exists():
            inliers_path = list(Path(inliers_dir).glob("**/inliers.npy"))[0]
            inliers_array = np.load(inliers_path)
            inliers = inliers_array[:, 0]  # 只使用第一个预测的inliers
            
            # 计算ground truth标签
            is_correct = []
            for q_idx, pred_indices in enumerate(predictions):
                top1_pred = pred_indices[0]
                is_correct.append(top1_pred in positives_per_query[q_idx])
            is_correct = np.array(is_correct, dtype=bool)
        else:
            raise FileNotFoundError(f"Neither JSON file nor inliers.npy found. JSON: {im_result_json}, inliers_dir: {inliers_dir}")
        
        # 确保长度一致
        if len(inliers) != len(predictions):
            print(f"[WARN] Inliers length ({len(inliers)}) != predictions length ({len(predictions)}). Truncating...")
            min_len = min(len(inliers), len(predictions))
            inliers = inliers[:min_len]
            predictions = predictions[:min_len]
            positives_per_query = positives_per_query[:min_len]
            if len(is_correct) != min_len:
                is_correct = is_correct[:min_len]
        
        return {
            'inliers': inliers,
            'is_correct': is_correct,
            'predictions': predictions,
            'positives_per_query': positives_per_query,
            'database_utms': database_utms
        }
    
    def fit_threshold(self, train_data: Dict, val_data: Dict):
        """
        使用网格搜索找最优阈值
        
        策略：尝试不同阈值，选择在验证集上R@1最高的
        """
        train_inliers = train_data['inliers']
        
        # 定义候选阈值（基于训练集inliers分布）
        percentiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        candidate_thresholds = [np.percentile(train_inliers, p) for p in percentiles]
        candidate_thresholds = [0] + candidate_thresholds + [np.max(train_inliers)]
        
        print(f"\n搜索最优阈值...")
        print(f"候选阈值范围: {min(candidate_thresholds):.1f} - {max(candidate_thresholds):.1f}")
        
        best_threshold = None
        best_val_score = -1
        results = []
        
        for threshold in tqdm(candidate_thresholds):
            # 在验证集上评估这个阈值
            val_score, val_rerank_ratio = self._evaluate_threshold(
                threshold, val_data
            )
            
            results.append({
                'threshold': threshold,
                'val_r1': val_score,
                'rerank_ratio': val_rerank_ratio
            })
            
            if val_score > best_val_score:
                best_val_score = val_score
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"\n✅ 最优阈值: {best_threshold:.1f}")
        print(f"   验证集R@1: {best_val_score:.2%}")
        
        return results
    
    def _evaluate_threshold(self, threshold: float, data: Dict) -> Tuple[float, float]:
        """
        评估给定阈值的性能
        
        Returns:
            (R@1得分, re-ranking比例)
        """
        inliers = data['inliers']
        predictions = data['predictions']
        positives_per_query = data['positives_per_query']
        
        # 决定哪些查询需要re-ranking
        need_rerank = inliers < threshold
        rerank_ratio = np.mean(need_rerank)
        
        # 计算R@1（假设re-ranking能完美修正错误的查询）
        # 注意：这里需要实际的re-ranking结果，这是简化版本
        # 实际实现需要读取re-ranking后的结果
        correct_count = 0
        for q_idx, pred_indices in enumerate(predictions):
            top1_pred = pred_indices[0]
            is_correct = top1_pred in positives_per_query[q_idx]
            correct_count += int(is_correct)
        
        r1_score = correct_count / len(predictions)
        
        return r1_score, rerank_ratio
    
    def fit_logistic(self, train_data: Dict, val_data: Dict):
        """
        训练逻辑回归模型
        
        特征：
        - inliers数量
        - inliers数量 / 图像尺寸（归一化）
        - 其他可能的特征...
        
        标签：
        - 1: 查询错误（需要re-ranking）
        - 0: 查询正确（不需要re-ranking）
        """
        # 准备训练数据
        X_train = self._extract_features(train_data)
        y_train = (~train_data['is_correct']).astype(int)  # 错误=1，正确=0
        
        X_val = self._extract_features(val_data)
        y_val = (~val_data['is_correct']).astype(int)
        
        print(f"\n训练逻辑回归模型...")
        print(f"训练集样本数: {len(X_train)} (需要rerank: {y_train.sum()})")
        print(f"验证集样本数: {len(X_val)} (需要rerank: {y_val.sum()})")
        
        # 网格搜索最优超参数
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'class_weight': ['balanced', None]
        }
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(
            lr, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        self.logistic_model = grid_search.best_estimator_
        
        # 在验证集上评估
        y_val_pred = self.logistic_model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average='binary'
        )
        
        print(f"\n✅ 最优超参数: {grid_search.best_params_}")
        print(f"   验证集准确率: {val_acc:.2%}")
        print(f"   精确率: {val_prec:.2%}, 召回率: {val_rec:.2%}, F1: {val_f1:.2%}")
        
        return {
            'best_params': grid_search.best_params_,
            'val_acc': val_acc,
            'val_prec': val_prec,
            'val_rec': val_rec,
            'val_f1': val_f1
        }
    
    def _extract_features(self, data: Dict) -> np.ndarray:
        """
        提取特征用于逻辑回归
        
        当前特征：
        - inliers数量
        - log(inliers+1) 
        - inliers数量的归一化（除以最大值）
        """
        inliers = data['inliers']
        
        # 特征工程
        features = np.column_stack([
            inliers,                          # 原始inliers数量
            np.log1p(inliers),                # log变换
            inliers / (np.max(inliers) + 1),  # 归一化
        ])
        
        return features
    
    def predict_need_rerank(self, data: Dict) -> np.ndarray:
        """
        预测哪些查询需要re-ranking
        
        Returns:
            布尔数组，True表示需要re-ranking
        """
        if self.method == "threshold":
            if self.threshold is None:
                raise ValueError("需要先调用fit_threshold()训练阈值")
            return data['inliers'] < self.threshold
        
        elif self.method == "logistic":
            if self.logistic_model is None:
                raise ValueError("需要先调用fit_logistic()训练模型")
            X = self._extract_features(data)
            return self.logistic_model.predict(X).astype(bool)
        
        else:
            raise ValueError(f"未知方法: {self.method}")
    
    def evaluate(self, data: Dict, reranked_results: Dict = None) -> Dict:
        """
        评估自适应re-ranking策略
        
        Args:
            data: 包含inliers和predictions的字典
            reranked_results: re-ranking后的结果（如果有）
        
        Returns:
            评估指标字典
        """
        need_rerank = self.predict_need_rerank(data)
        rerank_ratio = np.mean(need_rerank)
        
        # 计算R@1
        predictions = data['predictions']
        positives_per_query = data['positives_per_query']
        
        correct_count = 0
        for q_idx, pred_indices in enumerate(predictions):
            top1_pred = pred_indices[0]
            is_correct = top1_pred in positives_per_query[q_idx]
            correct_count += int(is_correct)
        
        r1_without_rerank = correct_count / len(predictions)
        
        # TODO: 如果有re-ranking结果，计算re-ranking后的R@1
        # 这需要实际的re-ranking代码
        
        return {
            'r1_without_rerank': r1_without_rerank,
            'rerank_ratio': rerank_ratio,
            'num_rerank': int(np.sum(need_rerank)),
            'total_queries': len(predictions),
            'cost_saving': 1.0 - rerank_ratio  # 节省的成本比例
        }
    
    def save(self, save_path: Path):
        """保存模型"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.method == "threshold":
            with open(save_path / "threshold.txt", 'w') as f:
                f.write(str(self.threshold))
        
        elif self.method == "logistic":
            with open(save_path / "logistic_model.pkl", 'wb') as f:
                pickle.dump(self.logistic_model, f)
        
        print(f"✅ 模型已保存到: {save_path}")
    
    def load(self, load_path: Path):
        """加载模型"""
        load_path = Path(load_path)
        
        if self.method == "threshold":
            with open(load_path / "threshold.txt", 'r') as f:
                self.threshold = float(f.read().strip())
            print(f"✅ 已加载阈值: {self.threshold}")
        
        elif self.method == "logistic":
            with open(load_path / "logistic_model.pkl", 'rb') as f:
                self.logistic_model = pickle.load(f)
            print(f"✅ 已加载逻辑回归模型")


def plot_threshold_analysis(results: List[Dict], save_path: Path):
    """
    绘制阈值分析图
    
    Args:
        results: 包含threshold, val_r1, rerank_ratio的列表
        save_path: 保存路径
    """
    thresholds = [r['threshold'] for r in results]
    val_r1 = [r['val_r1'] for r in results]
    rerank_ratios = [r['rerank_ratio'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Threshold (inliers count)', fontsize=12)
    ax1.set_ylabel('R@1 on Validation', color=color, fontsize=12)
    ax1.plot(thresholds, val_r1, color=color, marker='o', label='R@1')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Re-ranking Ratio', color=color, fontsize=12)
    ax2.plot(thresholds, rerank_ratios, color=color, marker='s', label='Re-rank Ratio')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 标记最优阈值
    best_idx = np.argmax(val_r1)
    best_threshold = thresholds[best_idx]
    best_r1 = val_r1[best_idx]
    ax1.axvline(x=best_threshold, color='green', linestyle='--', linewidth=2, 
                label=f'Best Threshold: {best_threshold:.1f}')
    
    plt.title('Threshold Selection: R@1 vs Re-ranking Ratio', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存到: {save_path}")
    plt.close()


def main():
    """主函数示例"""
    
    # 示例用法
    print("="*80)
    print("Extension 6.1: Adaptive Re-ranking")
    print("="*80)
    
    # 1. 硬阈值方法
    print("\n方法1: 硬阈值")
    adaptive_threshold = AdaptiveReranking(method="threshold")
    
    # TODO: 加载训练和验证数据
    # train_data = adaptive_threshold.load_data(
    #     preds_dir="logs/baseline/cosplace_l2_svox_sun",
    #     inliers_dir="logs/inliers/cosplace_superpoint_lg_svox_sun"
    # )
    # val_data = adaptive_threshold.load_data(...)
    # 
    # # 训练
    # threshold_results = adaptive_threshold.fit_threshold(train_data, val_data)
    # 
    # # 绘图
    # plot_threshold_analysis(threshold_results, "results/threshold_analysis.png")
    # 
    # # 保存
    # adaptive_threshold.save("models/adaptive_threshold")
    
    # 2. 逻辑回归方法
    print("\n方法2: 逻辑回归")
    adaptive_logistic = AdaptiveReranking(method="logistic")
    
    # TODO: 类似地训练逻辑回归模型
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()

