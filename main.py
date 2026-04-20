"""
main.py -- LLM-Augmented Hybrid Recommendation System
        MovieLens 100K x GPT-4o-mini

MODES
-----
  Demo mode (default -- ~10 LLM calls, < $0.01):
      python main.py
      python main.py --user 42

  Evaluation mode (~200 LLM calls, ~$0.02):
      python main.py --evaluate
      python main.py --evaluate --sample 500 --split 1

  SVD-only cross-validation (no LLM, free):
      python main.py --svd-cv

  Cross-validated weight tuning (Fix 3A):
      python main.py --tune-cv
      python main.py --tune-cv --sample 100

SETUP
-----
  1. pip install -r requirements.txt
  2. Copy .env.example -> .env and add your key:
         OPENAI_API_KEY=sk-...
  3. python main.py
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the package root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config       import ALPHA, BETA, TOP_N_CANDIDATES, OPENAI_API_KEY
from data_loader  import (
    load_split, build_rating_matrix, load_item_metadata,
    get_user_history, get_all_rated_movies,
)
from svd_model    import SVDModel
from llm_reasoner import (
    batch_reason_top_n, batch_reason_top_n_async,
    get_semantic_modifier, generate_taste_profile,
)
from hybrid_model import (
    rerank_candidates, hybrid_rating_prediction, tune_weights, tune_weights_cv,
)
from evaluator    import (
    compute_metrics, print_comparison_table,
    save_metrics_csv, save_predictions_csv,
    run_svd_cross_validation,
)


# ---------------------------------------------------------------------------
# Pretty-print helpers (pure ASCII -- safe on Windows cp1252)
# ---------------------------------------------------------------------------

BANNER = """
+--------------------------------------------------------------+
|   LLM-Augmented Hybrid Recommender System                    |
|   MovieLens 100K  x  GPT-4o-mini  x  SVD                    |
+--------------------------------------------------------------+
"""


def _divider(char: str = "-", width: int = 64) -> None:
    print(char * width)


def _check_api_key() -> None:
    if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        print("WARNING: OPENAI_API_KEY not set or looks invalid.")
        print("  Edit the .env file in llm_hybrid_recsys/ and add:")
        print("  OPENAI_API_KEY=sk-...\n")


# ---------------------------------------------------------------------------
# Mode 1 -- Demo (single user, re-ranking showcase)
#            Now uses async calls + taste profile (Fix 1A + 2A)
# ---------------------------------------------------------------------------

def run_demo(
    user_id_1indexed: int,
    split: int = 1,
    top_n: int = 5,
) -> None:
    """Show hybrid recommendations for one user with full LLM reasoning."""
    _check_api_key()

    print(f"\n[*] Loading MovieLens split u{split}...")
    train_df, _ = load_split(split)
    R            = build_rating_matrix(train_df)
    metadata     = load_item_metadata()

    print("[*] Training SVD model...", end=" ", flush=True)
    svd = SVDModel()
    svd.fit(R)
    print("done.")

    user_id = user_id_1indexed - 1   # 0-indexed from here

    # User history
    loved, disliked = get_user_history(user_id, R, metadata)

    _divider("=")
    print(f"  USER {user_id_1indexed} -- Taste Profile")
    _divider("-")
    print("  [+] Loved movies:")
    if loved:
        for m in loved:
            print(f"      {m['rating']}*  {m['title']}  [{', '.join(m['genres'])}]")
    else:
        print("      (no highly-rated movies found in training set)")

    if disliked:
        print("\n  [-] Disliked movies:")
        for m in disliked:
            print(f"      {m['rating']}*  {m['title']}  [{', '.join(m['genres'])}]")
    _divider("=")

    # Generate taste profile from full history (Fix 2A)
    print("\n[*] Generating taste profile from full history...")
    all_rated = get_all_rated_movies(user_id, R, metadata)
    taste_profile = asyncio.run(generate_taste_profile(user_id, all_rated))
    print(f"  Taste profile:\n{taste_profile}")

    # SVD top-N candidates
    print(f"\n[*] Generating top-{TOP_N_CANDIDATES} SVD candidates...")
    candidates = svd.top_n_recommendations(user_id, R, n=TOP_N_CANDIDATES)

    # Only reason over top_n for demo (cost control)
    demo_candidates = candidates[:top_n]

    # Async LLM calls with taste profile (Fix 1A + 2A)
    print(f"[*] Querying GPT-4o-mini ({len(demo_candidates)} async calls)...")
    reasoned = asyncio.run(
        batch_reason_top_n_async(
            user_id, demo_candidates, loved, disliked, metadata,
            taste_profile=taste_profile,
        )
    )

    # Hybrid re-ranking
    print(f"[*] Applying hybrid scoring (alpha={ALPHA}, beta={BETA})...")
    reranked = rerank_candidates(reasoned)

    # Display results
    _divider("=")
    print(f"  TOP {top_n} HYBRID RECOMMENDATIONS  --  User {user_id_1indexed}")
    _divider("=")

    for rank, rec in enumerate(reranked, 1):
        sign = "+" if rec["semantic_modifier"] >= 0 else ""
        print(f"\n  #{rank:02d}  {rec['title']}")
        print(f"       Genres : {', '.join(rec['genres']) or 'N/A'}")
        print(f"       CF     : {rec['cf_score']:.2f}   "
              f"LLM delta: {sign}{rec['semantic_modifier']:.2f}   "
              f"Final: {rec['final_score']:.2f}")
        print(f"       Reason : {rec['reasoning']}")

    _divider("=")


# ---------------------------------------------------------------------------
# Mode 2 -- Evaluation (sampled test pairs, SVD vs Hybrid)
# ---------------------------------------------------------------------------

def run_evaluation(
    split: int       = 1,
    sample_size: int = 200,
    tune: bool       = False,
) -> None:
    """Compare SVD baseline vs LLM-Hybrid on sampled test ratings."""
    _check_api_key()

    print(f"\n[*] Loading MovieLens split u{split}...")
    train_df, test_df = load_split(split)
    R                  = build_rating_matrix(train_df)
    metadata           = load_item_metadata()

    print("[*] Training SVD model...", end=" ", flush=True)
    svd = SVDModel()
    svd.fit(R)
    print("done.")

    # Sample test pairs grouped by user to support ranking metrics (NDCG, HR)
    items_per_user = 10
    valid_users = test_df.groupby("user_id").filter(lambda x: len(x) >= 2)["user_id"].unique()
    n_users = max(1, sample_size // items_per_user)
    
    np.random.seed(42)
    sampled_users = np.random.choice(valid_users, size=min(n_users, len(valid_users)), replace=False)
    test_samp = test_df[test_df["user_id"].isin(sampled_users)].groupby("user_id").head(items_per_user).reset_index(drop=True)
    n_pairs = len(test_samp)

    print(f"\n[*] Evaluating {n_pairs} test pairs with LLM reasoning...")
    _divider()

    # Pre-generate taste profiles for unique users (Fix 2A)
    unique_users = test_samp["user_id"].unique()
    taste_profiles: dict[int, str] = {}
    print(f"[*] Generating taste profiles for {len(unique_users)} unique users (async)...")
    
    async def _fetch_all_profiles():
        sem = asyncio.Semaphore(5)  # Safely constrain concurrency 
        async def _fetch(u_1ind):
            u_0 = u_1ind - 1
            all_rated = get_all_rated_movies(u_0, R, metadata)
            async with sem:
                prof = await generate_taste_profile(u_0, all_rated)
                return u_0, prof
        return await asyncio.gather(*[_fetch(u) for u in unique_users])
        
    for u_idx, prof in asyncio.run(_fetch_all_profiles()):
        taste_profiles[u_idx] = prof

    y_true, y_svd, y_hybrid = [], [], []
    y_modifiers, u_ids_test  = [], []

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

        llm = get_semantic_modifier(
            u_0, i_0, loved, disliked, target,
            taste_profile=taste_profiles.get(u_0),
        )
        mod = llm["semantic_modifier"]

        hybrid_pred = hybrid_rating_prediction(cf_pred, mod)

        y_true.append(r_true)
        y_svd.append(cf_pred)
        y_hybrid.append(hybrid_pred)
        y_modifiers.append(mod)
        u_ids_test.append(u_0)

        if (idx + 1) % 25 == 0 or idx + 1 == n_pairs:
            print(f"  Progress: {idx+1}/{n_pairs}", end="\r")

    print()  # clear progress line

    # Optional weight tuning
    y_tuned = None
    best    = None
    if tune:
        print("\n[*] Tuning alpha/beta weights...")
        best = tune_weights(y_true, y_svd, y_modifiers)
        print(f"    Best alpha={best['best_alpha']:.2f}  "
              f"beta={best['best_beta']:.2f}  "
              f"MAE={best['best_mae']:.4f}")
        y_tuned = [
            hybrid_rating_prediction(cf, mod, best["best_alpha"], best["best_beta"])
            for cf, mod in zip(y_svd, y_modifiers)
        ]

    # Metrics
    results = {
        "SVD Baseline"                         : compute_metrics(y_true, y_svd, u_ids_test),
        f"LLM-Hybrid (a={ALPHA}, b={BETA})"   : compute_metrics(y_true, y_hybrid, u_ids_test),
    }
    if y_tuned and best:
        ba, bb = best["best_alpha"], best["best_beta"]
        results[f"LLM-Hybrid tuned (a={ba:.2f},b={bb:.2f})"] = compute_metrics(y_true, y_tuned, u_ids_test)

    print_comparison_table(results)

    # Save outputs
    csv_path  = save_metrics_csv(results, f"metrics_split{split}.csv")
    print(f"\n[+] Metrics saved  -> {csv_path}")

    preds = {"y_svd": y_svd, "y_hybrid": y_hybrid}
    if y_tuned:
        preds["y_hybrid_tuned"] = y_tuned
    pred_path = save_predictions_csv(y_true, preds, f"predictions_split{split}.csv")
    print(f"[+] Predictions saved -> {pred_path}")


# ---------------------------------------------------------------------------
# Mode 3 -- SVD cross-validation only (no LLM, free)
# ---------------------------------------------------------------------------

def run_svd_cv() -> None:
    """Evaluate pure SVD across all 5 splits. No API calls."""
    print("\n[*] SVD Cross-Validation (5 splits, no LLM)...")
    _divider()
    avg = run_svd_cross_validation(verbose=True)
    _divider()
    print(f"  Average RMSE : {avg['RMSE']:.4f}")
    print(f"  Average MAE  : {avg['MAE']:.4f}")
    print(f"  Average NMAE : {avg['NMAE']:.4f}")
    _divider()
    save_metrics_csv({"SVD (avg 5-split)": avg}, "svd_cv_results.csv")
    print("[+] Saved -> results/svd_cv_results.csv")


# ---------------------------------------------------------------------------
# Mode 4 -- Cross-validated weight tuning (Fix 3A)
# ---------------------------------------------------------------------------

def run_tune_cv(sample_size: int = 50) -> None:
    """Grid-search alpha/beta across all 5 splits and report averages."""
    _check_api_key()

    print("\n[*] Cross-Validated Weight Tuning (5 splits)...")
    _divider()

    result = tune_weights_cv(sample_size=sample_size, verbose=True)

    _divider("=")
    print(f"  AVERAGED OPTIMAL WEIGHTS (across 5 splits)")
    _divider("-")
    print(f"  alpha = {result['avg_alpha']:.2f}")
    print(f"  beta  = {result['avg_beta']:.2f}")
    print(f"  MAE   = {result['avg_mae']:.4f}")
    _divider("=")

    print("\n  Per-split breakdown:")
    for i, s in enumerate(result["per_split"], 1):
        print(f"    u{i}: alpha={s['best_alpha']:.2f}  "
              f"beta={s['best_beta']:.2f}  MAE={s['best_mae']:.4f}")

    # Save
    save_data = {
        f"Split u{i+1} (a={s['best_alpha']:.2f},b={s['best_beta']:.2f})": {
            "RMSE": 0.0, "MAE": s["best_mae"], "NMAE": s["best_mae"] / 4.0,
        }
        for i, s in enumerate(result["per_split"])
    }
    save_data[f"Average (a={result['avg_alpha']:.2f},b={result['avg_beta']:.2f})"] = {
        "RMSE": 0.0, "MAE": result["avg_mae"], "NMAE": result["avg_mae"] / 4.0,
    }
    csv_path = save_metrics_csv(save_data, "tune_cv_results.csv")
    print(f"\n[+] Saved -> {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-Augmented Hybrid Recommender -- MovieLens 100K x GPT-4o-mini",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--evaluate", action="store_true",
                   help="Evaluation mode: SVD vs Hybrid on sampled test pairs")
    p.add_argument("--svd-cv",   action="store_true",
                   help="SVD 5-split cross-validation only (no LLM)")
    p.add_argument("--tune-cv",  action="store_true",
                   help="Cross-validated alpha/beta tuning across 5 splits (Fix 3A)")
    p.add_argument("--user",     type=int, default=None,
                   help="User ID (1-indexed) for demo mode. Random if omitted.")
    p.add_argument("--split",    type=int, default=1,
                   help="MovieLens split to use (1-5). Default: 1")
    p.add_argument("--sample",   type=int, default=200,
                   help="Number of test pairs for evaluation. Default: 200")
    p.add_argument("--top-n",    type=int, default=10,
                   help="Recommendations to display in demo. Default: 10")
    p.add_argument("--tune",     action="store_true",
                   help="Grid-search optimal alpha/beta during evaluation")
    return p.parse_args()


def main() -> None:
    print(BANNER)
    args = parse_args()

    if args.svd_cv:
        run_svd_cv()

    elif args.tune_cv:
        run_tune_cv(sample_size=args.sample)

    elif args.evaluate:
        run_evaluation(
            split       = args.split,
            sample_size = args.sample,
            tune        = args.tune,
        )

    else:
        user_id = args.user if args.user else random.randint(1, 100)
        run_demo(
            user_id_1indexed = user_id,
            split            = args.split,
            top_n            = args.top_n,
        )


if __name__ == "__main__":
    main()
