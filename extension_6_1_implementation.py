#!/usr/bin/env python3
"""
Extension 6.1: Adaptive Re-ranking Implementation

Core idea:
- Apply re-ranking only to "hard" queries
- "Hard" definition: inlier count below threshold
- Two methods: hard threshold or logistic regression
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
    """Adaptive Re-ranking system"""
    
    def __init__(self, method="threshold"):
        """
        Args:
            method: "threshold" or "logistic"
        """
        self.method = method
        self.threshold = None
        self.logistic_model = None
        
    def load_data(self, preds_dir: Path, im_result_json: Path = None, inliers_dir: Path = None):
        """
        Load VPR predictions and inliers data
        
        Args:
            preds_dir: VPR prediction results directory (contains z_data.torch)
            im_result_json: Image Matching JSON result file path (new format, preferred)
            inliers_dir: Image Matching results directory (contains inliers.npy, old format, for compatibility)
        """
        z_data_path = list(Path(preds_dir).glob("**/z_data.torch"))[0]
        z_data = torch.load(z_data_path, map_location='cpu', weights_only=False)
        
        predictions = z_data['predictions']
        positives_per_query = z_data['positives_per_query']
        database_utms = z_data['database_utms']
        
        if im_result_json and Path(im_result_json).exists():
            with open(im_result_json, 'r', encoding='utf-8') as f:
                im_data = json.load(f)
            
            num_inliers = im_data['results']['num_inliers']
            pred_ranks = im_data['results']['pred_ranks']
            query_ids = im_data['results']['query_ids']
            is_correct_json = im_data['results']['is_correct']
            
            num_queries = len(predictions)
            top_k = predictions.shape[1] if len(predictions.shape) > 1 else 1
            
            inliers_matrix = np.zeros((num_queries, top_k), dtype=np.float64)
            top1_inliers = np.zeros(num_queries, dtype=np.float64)
            top1_is_correct = np.zeros(num_queries, dtype=bool)
            
            for i, (q_idx, rank, inlier_count, correct) in enumerate(zip(query_ids, pred_ranks, num_inliers, is_correct_json)):
                if q_idx < num_queries and rank < top_k:
                    inliers_matrix[q_idx, rank] = inlier_count
                    if rank == 0:
                        top1_inliers[q_idx] = inlier_count
                        top1_is_correct[q_idx] = correct
            
            inliers = top1_inliers
            is_correct = top1_is_correct
            
            self.inliers_matrix = inliers_matrix
            
        elif inliers_dir and Path(inliers_dir).exists():
            inliers_path = list(Path(inliers_dir).glob("**/inliers.npy"))[0]
            inliers_array = np.load(inliers_path)
            inliers = inliers_array[:, 0]
            
            is_correct = []
            for q_idx, pred_indices in enumerate(predictions):
                top1_pred = pred_indices[0]
                is_correct.append(top1_pred in positives_per_query[q_idx])
            is_correct = np.array(is_correct, dtype=bool)
            
            top_k = predictions.shape[1] if len(predictions.shape) > 1 else 1
            inliers_matrix = np.zeros((len(predictions), top_k), dtype=np.float64)
            inliers_matrix[:, 0] = inliers
            self.inliers_matrix = inliers_matrix
        else:
            raise FileNotFoundError(f"Neither JSON file nor inliers.npy found. JSON: {im_result_json}, inliers_dir: {inliers_dir}")
        
        if len(inliers) != len(predictions):
            print(f"[WARN] Inliers length ({len(inliers)}) != predictions length ({len(predictions)}). Truncating...")
            min_len = min(len(inliers), len(predictions))
            inliers = inliers[:min_len]
            predictions = predictions[:min_len]
            positives_per_query = positives_per_query[:min_len]
            if len(is_correct) != min_len:
                is_correct = is_correct[:min_len]
            if hasattr(self, 'inliers_matrix'):
                self.inliers_matrix = self.inliers_matrix[:min_len]
        
        return {
            'inliers': inliers,
            'is_correct': is_correct,
            'predictions': predictions,
            'positives_per_query': positives_per_query,
            'database_utms': database_utms
        }
    
    def fit_threshold(self, train_data: Dict, val_data: Dict):
        """
        Find optimal threshold using grid search
        
        Strategy: try different thresholds, select the one with highest R@1 on validation set
        """
        train_inliers = train_data['inliers']
        
        percentiles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
        candidate_thresholds = [np.percentile(train_inliers, p) for p in percentiles]
        candidate_thresholds = [0] + candidate_thresholds + [np.max(train_inliers)]
        
        print(f"\nSearching for optimal threshold...")
        print(f"Candidate threshold range: {min(candidate_thresholds):.1f} - {max(candidate_thresholds):.1f}")
        
        best_threshold = None
        best_val_score = -1
        results = []
        
        for threshold in tqdm(candidate_thresholds):
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
        print(f"\n[OK] Optimal threshold: {best_threshold:.1f}")
        print(f"   Validation R@1: {best_val_score:.2%}")
        
        return results
    
    def _evaluate_threshold(self, threshold: float, data: Dict) -> Tuple[float, float]:
        """
        Evaluate performance of given threshold (actually performs re-ranking)
        
        Returns:
            (R@1 score, re-ranking ratio)
        """
        inliers = data['inliers']
        predictions = data['predictions']
        positives_per_query = data['positives_per_query']
        
        need_rerank = inliers < threshold
        rerank_ratio = np.mean(need_rerank)
        
        correct_count = 0
        for q_idx, pred_indices in enumerate(predictions):
            if need_rerank[q_idx] and hasattr(self, 'inliers_matrix'):
                query_inliers = self.inliers_matrix[q_idx, :len(pred_indices)]
                sorted_indices = np.argsort(query_inliers)[::-1]
                reranked_preds = pred_indices[sorted_indices]
                top1_pred = reranked_preds[0]
            else:
                top1_pred = pred_indices[0]
            
            is_correct = top1_pred in positives_per_query[q_idx]
            correct_count += int(is_correct)
        
        r1_score = correct_count / len(predictions)
        
        return r1_score, rerank_ratio
    
    def fit_logistic(self, train_data: Dict, val_data: Dict):
        """
        Train logistic regression model
        
        Features:
        - inlier count
        - log(inlier count)
        - normalized inlier count
        
        Labels:
        - 1: query is incorrect (needs re-ranking)
        - 0: query is correct (no re-ranking needed)
        """
        X_train = self._extract_features(train_data)
        y_train = (~train_data['is_correct']).astype(int)
        
        X_val = self._extract_features(val_data)
        y_val = (~val_data['is_correct']).astype(int)
        
        print(f"\nTraining logistic regression model...")
        print(f"Training samples: {len(X_train)} (need rerank: {y_train.sum()})")
        print(f"Validation samples: {len(X_val)} (need rerank: {y_val.sum()})")
        
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
        
        y_val_pred = self.logistic_model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average='binary'
        )
        
        print(f"\n[OK] Best hyperparameters: {grid_search.best_params_}")
        print(f"   Validation accuracy: {val_acc:.2%}")
        print(f"   Precision: {val_prec:.2%}, Recall: {val_rec:.2%}, F1: {val_f1:.2%}")
        
        return {
            'best_params': grid_search.best_params_,
            'val_acc': val_acc,
            'val_prec': val_prec,
            'val_rec': val_rec,
            'val_f1': val_f1
        }
    
    def _extract_features(self, data: Dict) -> np.ndarray:
        """
        Extract features for logistic regression
        
        Features:
        - inlier count
        - log(inliers+1)
        - normalized inlier count
        """
        inliers = data['inliers']
        
        features = np.column_stack([
            inliers,
            np.log1p(inliers),
            inliers / (np.max(inliers) + 1),
        ])
        
        return features
    
    def predict_need_rerank(self, data: Dict) -> np.ndarray:
        """
        Predict which queries need re-ranking
        
        Returns:
            Boolean array, True means needs re-ranking
        """
        if self.method == "threshold":
            if self.threshold is None:
                raise ValueError("Must call fit_threshold() first to train threshold")
            return data['inliers'] < self.threshold
        
        elif self.method == "logistic":
            if self.logistic_model is None:
                raise ValueError("Must call fit_logistic() first to train model")
            X = self._extract_features(data)
            return self.logistic_model.predict(X).astype(bool)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def evaluate(self, data: Dict, reranked_results: Dict = None) -> Dict:
        """
        Evaluate adaptive re-ranking strategy (actually performs re-ranking)
        
        Args:
            data: dictionary containing inliers and predictions
            reranked_results: re-ranked results (if any, unused)
        
        Returns:
            evaluation metrics dictionary
        """
        need_rerank = self.predict_need_rerank(data)
        rerank_ratio = np.mean(need_rerank)
        
        predictions = data['predictions']
        positives_per_query = data['positives_per_query']
        
        correct_count_without = 0
        for q_idx, pred_indices in enumerate(predictions):
            top1_pred = pred_indices[0]
            is_correct = top1_pred in positives_per_query[q_idx]
            correct_count_without += int(is_correct)
        
        r1_without_rerank = correct_count_without / len(predictions)
        
        correct_count_with = 0
        for q_idx, pred_indices in enumerate(predictions):
            if need_rerank[q_idx] and hasattr(self, 'inliers_matrix'):
                query_inliers = self.inliers_matrix[q_idx, :len(pred_indices)]
                sorted_indices = np.argsort(query_inliers)[::-1]
                reranked_preds = pred_indices[sorted_indices]
                top1_pred = reranked_preds[0]
            else:
                top1_pred = pred_indices[0]
            
            is_correct = top1_pred in positives_per_query[q_idx]
            correct_count_with += int(is_correct)
        
        r1_with_rerank = correct_count_with / len(predictions)
        
        return {
            'r1_without_rerank': r1_without_rerank,
            'r1_with_rerank': r1_with_rerank,
            'rerank_ratio': rerank_ratio,
            'num_rerank': int(np.sum(need_rerank)),
            'total_queries': len(predictions),
            'cost_saving': 1.0 - rerank_ratio
        }
    
    def save(self, save_path: Path):
        """Save model"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.method == "threshold":
            with open(save_path / "threshold.txt", 'w') as f:
                f.write(str(self.threshold))
        
        elif self.method == "logistic":
            with open(save_path / "logistic_model.pkl", 'wb') as f:
                pickle.dump(self.logistic_model, f)
        
        print(f"[OK] Model saved to: {save_path}")
    
    def load(self, load_path: Path):
        """Load model"""
        load_path = Path(load_path)
        
        if self.method == "threshold":
            with open(load_path / "threshold.txt", 'r') as f:
                self.threshold = float(f.read().strip())
            print(f"[OK] Loaded threshold: {self.threshold}")
        
        elif self.method == "logistic":
            with open(load_path / "logistic_model.pkl", 'rb') as f:
                self.logistic_model = pickle.load(f)
            print(f"[OK] Loaded logistic regression model")


def plot_threshold_analysis(results: List[Dict], save_path: Path):
    """
    Plot threshold analysis
    
    Args:
        results: list containing threshold, val_r1, rerank_ratio
        save_path: save path
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
    print(f"[OK] Plot saved to: {save_path}")
    plt.close()


def main():
    """Example main function"""
    
    print("="*80)
    print("Extension 6.1: Adaptive Re-ranking")
    print("="*80)
    
    print("\nMethod 1: Hard threshold")
    adaptive_threshold = AdaptiveReranking(method="threshold")
    
    print("\nMethod 2: Logistic regression")
    adaptive_logistic = AdaptiveReranking(method="logistic")
    
    print("\n[OK] Done!")


if __name__ == "__main__":
    main()

