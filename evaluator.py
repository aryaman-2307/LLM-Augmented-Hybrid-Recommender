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
) -> dict[str, float]:
    """Return {RMSE, MAE, NMAE} for a prediction pair."""
    yt = np.array(y_true, dtype=float)
    yp = np.clip(np.array(y_pred, dtype=float), MIN_RATING, MAX_RATING)
    return {
        "RMSE": rmse(yt, yp),
        "MAE" : mae(yt, yp),
        "NMAE": nmae(yt, yp),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Console reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    """Pretty-print a comparison table of all models.

    Args:
        results: {model_name: {RMSE, MAE, NMAE}}
    """
    width = max(len(k) for k in results) + 2
    header = f"{'Model':<{width}} {'RMSE':>8}  {'MAE':>8}  {'NMAE':>8}"
    sep    = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for model, m in results.items():
        print(f"{model:<{width}} {m['RMSE']:>8.4f}  {m['MAE']:>8.4f}  {m['NMAE']:>8.4f}")
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

    for split in range(1, 6):
        train_df, test_df = load_split(split)
        R = build_rating_matrix(train_df)

        svd = SVDModel()
        svd.fit(R)

        y_true, y_pred = [], []
        for row in test_df.itertuples(index=False):
            u, i, r = row.user_id - 1, row.item_id - 1, row.rating
            y_true.append(r)
            y_pred.append(svd.predict(u, i))

        m = compute_metrics(y_true, y_pred)
        all_rmse.append(m["RMSE"])
        all_mae.append(m["MAE"])
        all_nmae.append(m["NMAE"])

        if verbose:
            print(f"  Split u{split}: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  NMAE={m['NMAE']:.4f}")

    avg = {
        "RMSE": float(np.mean(all_rmse)),
        "MAE" : float(np.mean(all_mae)),
        "NMAE": float(np.mean(all_nmae)),
    }
    if verbose:
        print(f"  Average  : RMSE={avg['RMSE']:.4f}  MAE={avg['MAE']:.4f}  NMAE={avg['NMAE']:.4f}")

    return avg
