"""
evaluator.py — RMSE / MAE / NMAE evaluation utilities

Supports:
  - Per-model metric computation
  - Multi-model comparison table (console + CSV)
  - Full 5-split cross-validation loop (SVD baseline only, no LLM cost)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from config import MIN_RATING, MAX_RATING, RESULTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def nmae(y_true: np.ndarray, y_pred: np.ndarray, scale: float = 4.0) -> float:
    """Normalised MAE = MAE / (max_rating - min_rating)."""
    return mae(y_true, y_pred) / scale


def compute_metrics(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
    u_ids: list[int] | None = None,
    k: int = 10,
    threshold: float = 4.0
) -> dict[str, float]:
    """Return {RMSE, MAE, NMAE, optionally NDCG@K, HR@K} for a prediction pair."""
    yt = np.array(y_true, dtype=float)
    yp = np.clip(np.array(y_pred, dtype=float), MIN_RATING, MAX_RATING)
    
    metrics = {
        "RMSE": rmse(yt, yp),
        "MAE" : mae(yt, yp),
        "NMAE": nmae(yt, yp),
    }

    if u_ids is not None:
        from collections import defaultdict
        user_data = defaultdict(list)
        for u, rt, rp in zip(u_ids, yt, yp):
            user_data[u].append((rt, rp))
            
        ndcg_list = []
        hit_list = []
        
        for u, items in user_data.items():
            if len(items) < 2:
                continue
                
            items.sort(key=lambda x: x[1], reverse=True)
            top_k = items[:k]
            
            # NDCG @ K
            dcg = sum((2**rt - 1) / np.log2(idx + 2) for idx, (rt, rp) in enumerate(top_k))
            ideal_items = sorted(items, key=lambda x: x[0], reverse=True)[:k]
            idcg = sum((2**rt - 1) / np.log2(idx + 2) for idx, (rt, rp) in enumerate(ideal_items))
            
            ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
            
            # Hit Rate @ K (1 if at least one hit, 0 otherwise)
            hits_in_k = sum(1 for rt, rp in top_k if rt >= threshold)
            hit_list.append(1.0 if hits_in_k > 0 else 0.0)
            
        if ndcg_list:
            metrics[f"NDCG@{k}"] = float(np.mean(ndcg_list))
            metrics[f"HR@{k}"] = float(np.mean(hit_list))
        else:
            metrics[f"NDCG@{k}"] = 0.0
            metrics[f"HR@{k}"] = 0.0

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Console reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    """Pretty-print a comparison table of all models.

    Args:
        results: {model_name: {RMSE, MAE, NMAE, etc.}}
    """
    if not results:
        return
        
    width = max(len(k) for k in results) + 2
    metric_keys = list(next(iter(results.values())).keys())
    
    # Try to order them logically
    ordered_keys = ["RMSE", "MAE", "NMAE"]
    ranking_keys = [key for key in metric_keys if key not in ordered_keys]
    ordered_keys += ranking_keys

    header_cols = [f"{'Model':<{width}}"] + [f"{key:>8}" for key in ordered_keys]
    header = "  ".join(header_cols)
    sep    = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for model, m in results.items():
        row_cols = [f"{model:<{width}}"] + [f"{m.get(key, 0.0):>8.4f}" for key in ordered_keys]
        print("  ".join(row_cols))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CSV persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(results: dict[str, dict[str, float]], filename: str = "metrics.csv") -> Path:
    """Save comparison results to results/<filename>."""
    rows = [{"Model": k, **v} for k, v in results.items()]
    df   = pd.DataFrame(rows)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path


def save_predictions_csv(
    y_true: list[float],
    predictions: dict[str, list[float]],
    filename: str = "predictions.csv",
) -> Path:
    """Save a per-pair predictions CSV for further analysis."""
    data = {"y_true": y_true, **predictions}
    df   = pd.DataFrame(data)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# SVD 5-split cross-validation (baseline, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def run_svd_cross_validation(verbose: bool = True) -> dict[str, float]:
    """Train and evaluate SVD on all 5 MovieLens splits.

    Returns averaged metrics across splits.
    """
    # Lazy imports to avoid circular deps
    from data_loader import load_split, build_rating_matrix
    from svd_model    import SVDModel

    all_rmse, all_mae, all_nmae = [], [], []
    all_ndcg, all_hr = [], []

    for split in range(1, 6):
        train_df, test_df = load_split(split)
        R = build_rating_matrix(train_df)

        svd = SVDModel()
        svd.fit(R)

        y_true, y_pred, u_ids = [], [], []
        for row in test_df.itertuples(index=False):
            u, i, r = row.user_id - 1, row.item_id - 1, row.rating
            y_true.append(r)
            y_pred.append(svd.predict(u, i))
            u_ids.append(u)

        m = compute_metrics(y_true, y_pred, u_ids)
        all_rmse.append(m["RMSE"])
        all_mae.append(m["MAE"])
        all_nmae.append(m["NMAE"])
        all_ndcg.append(m.get("NDCG@10", 0.0))
        all_hr.append(m.get("HR@10", 0.0))

        if verbose:
            print(f"  Split u{split}: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  NMAE={m['NMAE']:.4f}  NDCG@10={m.get('NDCG@10', 0.0):.4f}  HR@10={m.get('HR@10', 0.0):.4f}")

    avg = {
        "RMSE": float(np.mean(all_rmse)),
        "MAE" : float(np.mean(all_mae)),
        "NMAE": float(np.mean(all_nmae)),
        "NDCG@10": float(np.mean(all_ndcg)),
        "HR@10": float(np.mean(all_hr)),
    }
    if verbose:
        print(f"  Average  : RMSE={avg['RMSE']:.4f}  MAE={avg['MAE']:.4f}  NMAE={avg['NMAE']:.4f}  NDCG@10={avg['NDCG@10']:.4f}  HR@10={avg['HR@10']:.4f}")

    return avg
