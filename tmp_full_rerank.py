# run in colab derictly
import torch, pickle, json, os
import numpy as np
from pathlib import Path

# 确保路径正确
ROOT = Path("/content/drive/MyDrive/Visual-Place-Recognition-Project")

N_LIST = [1, 5, 10, 20]
COMBOS = [
    ("cosplace","superpoint-lg","sf_xs_test"),
    ("cosplace","superpoint-lg","tokyo_xs_test"),
    ("cosplace","loftr","sf_xs_test"),
    ("cosplace","loftr","tokyo_xs_test"),
    ("mixvpr","superpoint-lg","sf_xs_test"),
    ("mixvpr","superpoint-lg","tokyo_xs_test"),
    ("mixvpr","loftr","sf_xs_test"),
    ("mixvpr","loftr","tokyo_xs_test"),
]

def find_z_data(exp_dir):
    if not exp_dir.exists(): return None
    files = list(exp_dir.rglob("z_data.torch"))
    if not files: return None
    return sorted(files, key=lambda x: x.stat().st_mtime)[-1]

def compute_recalls(vpr, matcher, dataset):
    exp_dir = ROOT / f"logs/baseline/{vpr}_dot_product_{dataset}"
    preds_path = find_z_data(exp_dir)
    im_path = ROOT / f"results/image_matching/{matcher}_{vpr}_dot_product_{dataset}.pkl"
    
    if not preds_path or not im_path.exists():
        return None

    # 加载数据
    preds = torch.load(preds_path, map_location="cpu", weights_only=False)
    im = pickle.load(open(im_path, "rb"))["results"]

    K = max(im["pred_ranks"]) + 1
    num_q = max(im["query_ids"]) + 1

    num_inliers = np.zeros((num_q, K), dtype=np.float32)
    for q, r, nin in zip(im["query_ids"], im["pred_ranks"], im["num_inliers"]):
        num_inliers[q, r] = nin

    # --- 修正点：兼容处理 numpy 和 tensor ---
    preds_raw = preds["predictions"]
    if hasattr(preds_raw, "cpu"):
        preds_idx = preds_raw.cpu().numpy()
    else:
        preds_idx = preds_raw # 已经是 numpy 数组
    
    positives = preds["positives_per_query"]
    
    is_corr = np.zeros((num_q, K), dtype=bool)
    min_q = min(num_q, len(positives))
    for q in range(min_q):
        pos = set(positives[q])
        for k in range(K):
            is_corr[q, k] = preds_idx[q, k] in pos

    def recall(mat):
        out = {}
        for n in N_LIST:
            hits = (mat[:min_q, :n].any(axis=1)).sum()
            out[f"R@{n}"] = hits / min_q
        return out

    base = recall(is_corr)
    rerank_corr = np.zeros_like(is_corr)
    for q in range(min_q):
        order = np.argsort(-num_inliers[q] + np.arange(K)*1e-6)
        rerank_corr[q] = is_corr[q, order]
    
    full = recall(rerank_corr)
    return base, full

rows = []
print(f"{'vpr':10} {'matcher':15} {'dataset':15} {'Base R@1':>10} {'Full R@1':>10}")
print("-" * 75)

for vpr, matcher, dataset in COMBOS:
    res = compute_recalls(vpr, matcher, dataset)
    if res:
        base, full = res
        row = {"vpr": vpr, "matcher": matcher, "dataset": dataset}
        row.update({f"base_{k}": v for k,v in base.items()})
        row.update({f"full_{k}": v for k,v in full.items()})
        rows.append(row)
        print(f"{vpr:10} {matcher:15} {dataset:15} {base['R@1']*100:>9.2f}% {full['R@1']*100:>9.2f}%")

out_path = ROOT / "results/extension_6_1/full_rerank_summary.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"\n[SAVED] {out_path}")