"""
hybrid_model.py — Hybrid Scoring & Re-ranking Logic

The Hybrid Formula:
    Final_Score = α × CF_Prediction + β × LLM_Semantic_Modifier

Where:
    CF_Prediction     ∈ [1, 5]   — SVD collaborative filtering score
    Semantic_Modifier ∈ [-1, 1]  — LLM alignment signal
    α, β              ∈ ℝ        — tunable weights (default: both 1.0)

Two modes:
  1. Re-ranking:   Sort a fixed candidate list by Final_Score
  2. Rating pred:  Use hybrid score as a rating estimate for MAE/RMSE evaluation
"""

from __future__ import annotations

import numpy as np
from config import ALPHA, BETA, MIN_RATING, MAX_RATING


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring
# ─────────────────────────────────────────────────────────────────────────────

def final_score(
    cf_score: float,
    semantic_modifier: float,
    alpha: float = ALPHA,
    beta: float  = BETA,
) -> float:
    """Compute hybrid score and clip to [MIN_RATING, MAX_RATING]."""
    raw = alpha * cf_score + beta * semantic_modifier
    return float(np.clip(raw, MIN_RATING, MAX_RATING))


# ─────────────────────────────────────────────────────────────────────────────
# Re-ranking (recommendation task)
# ─────────────────────────────────────────────────────────────────────────────

def rerank_candidates(
    candidates: list[dict],
    alpha: float = ALPHA,
    beta: float  = BETA,
) -> list[dict]:
    """Add 'final_score' to each candidate dict and sort descending.

    Each candidate must have 'cf_score' and 'semantic_modifier' keys.
    Returns a new sorted list (does not modify in-place).
    """
    ranked = []
    for c in candidates:
        enriched = dict(c)
        enriched["final_score"] = final_score(
            c["cf_score"], c["semantic_modifier"], alpha, beta
        )
        ranked.append(enriched)

    return sorted(ranked, key=lambda x: x["final_score"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Rating prediction (evaluation task)
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_rating_prediction(
    cf_prediction: float,
    semantic_modifier: float,
    alpha: float = ALPHA,
    beta: float  = BETA,
) -> float:
    """For per-pair rating prediction evaluation (RMSE/MAE).

    The LLM modifier is treated as a direct additive signal on top of the
    CF base prediction, then clipped to [1, 5].
    """
    return float(np.clip(alpha * cf_prediction + beta * semantic_modifier,
                         MIN_RATING, MAX_RATING))


# ─────────────────────────────────────────────────────────────────────────────
# Weight grid search (optional tuning)
# ─────────────────────────────────────────────────────────────────────────────

def tune_weights(
    y_true: list[float],
    y_cf:   list[float],
    y_mod:  list[float],
    alpha_range: tuple[float, float, int] = (0.5, 1.5, 11),
    beta_range:  tuple[float, float, int] = (0.0, 2.0, 11),
) -> dict:
    """Grid-search α and β to minimise MAE on a sample.

    Returns {"best_alpha": float, "best_beta": float, "best_mae": float}.
    """
    yt = np.array(y_true)
    yc = np.array(y_cf)
    ym = np.array(y_mod)

    best_mae   = np.inf
    best_alpha = ALPHA
    best_beta  = BETA

    alphas = np.linspace(*alpha_range)
    betas  = np.linspace(*beta_range)

    for a in alphas:
        for b in betas:
            y_hat = np.clip(a * yc + b * ym, MIN_RATING, MAX_RATING)
            mae   = float(np.mean(np.abs(yt - y_hat)))
            if mae < best_mae:
                best_mae   = mae
                best_alpha = float(a)
                best_beta  = float(b)

    return {"best_alpha": best_alpha, "best_beta": best_beta, "best_mae": best_mae}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated weight tuning (Fix 3A)
# ─────────────────────────────────────────────────────────────────────────────

def tune_weights_cv(
    sample_size: int = 50,
    splits: tuple[int, ...] = (1, 2, 3, 4, 5),
    verbose: bool = True,
) -> dict:
    """Grid-search α and β across multiple MovieLens splits and average.

    This reduces sensitivity to any single train/test partition (Fix 3A).

    Returns {"avg_alpha": float, "avg_beta": float, "avg_mae": float,
             "per_split": list[dict]}.
    """
    # Lazy imports to avoid circular dependencies
    from data_loader import (
        load_split, build_rating_matrix, load_item_metadata,
        get_user_history, get_all_rated_movies,
    )
    from svd_model    import SVDModel
    from llm_reasoner import get_semantic_modifier

    metadata   = load_item_metadata()
    per_split  = []

    for split in splits:
        if verbose:
            print(f"\n  [CV] Processing split u{split}...")

        train_df, test_df = load_split(split)
        R = build_rating_matrix(train_df)

        svd = SVDModel()
        svd.fit(R)

        n_pairs   = min(sample_size, len(test_df))
        test_samp = test_df.sample(n_pairs, random_state=42).reset_index(drop=True)

        y_true, y_cf, y_mod = [], [], []

        for idx, row in enumerate(test_samp.itertuples(index=False)):
            u_0    = row.user_id - 1
            i_0    = row.item_id - 1
            r_true = float(row.rating)

            cf_pred = svd.predict(u_0, i_0)

            loved, disliked = get_user_history(u_0, R, metadata)
            meta   = metadata.get(i_0 + 1, {})
            target = {
                "title" : meta.get("title",  f"Movie {i_0+1}"),
                "genres": meta.get("genres", []),
            }

            llm = get_semantic_modifier(u_0, i_0, loved, disliked, target)
            mod = llm["semantic_modifier"]

            y_true.append(r_true)
            y_cf.append(cf_pred)
            y_mod.append(mod)

            if verbose and ((idx + 1) % 25 == 0 or idx + 1 == n_pairs):
                print(f"    Split u{split}: {idx+1}/{n_pairs}", end="\r")

        if verbose:
            print()

        best = tune_weights(y_true, y_cf, y_mod)
        per_split.append(best)

        if verbose:
            print(f"    Split u{split}: best alpha={best['best_alpha']:.2f}  "
                  f"beta={best['best_beta']:.2f}  MAE={best['best_mae']:.4f}")

    avg_alpha = float(np.mean([s["best_alpha"] for s in per_split]))
    avg_beta  = float(np.mean([s["best_beta"]  for s in per_split]))
    avg_mae   = float(np.mean([s["best_mae"]   for s in per_split]))

    return {
        "avg_alpha": avg_alpha,
        "avg_beta":  avg_beta,
        "avg_mae":   avg_mae,
        "per_split": per_split,
    }
