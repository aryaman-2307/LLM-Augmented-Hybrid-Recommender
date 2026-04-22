"""
app.py — Flask API for LLM-Augmented Hybrid Recommendation System
           Serves the web frontend and exposes REST endpoints.

Run:
    python app.py
    → http://localhost:5000
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import traceback
from pathlib import Path
from threading import Thread

from flask import Flask, jsonify, request, send_from_directory

# Make sure the package root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ALPHA, BETA, TOP_N_CANDIDATES, N_FACTORS,
    LLM_MODEL, N_USERS, N_ITEMS, OPENAI_API_KEY,
)
from data_loader import (
    load_split, build_rating_matrix, load_item_metadata,
    get_user_history, get_all_rated_movies,
)
from svd_model import SVDModel
from llm_reasoner import (
    batch_reason_top_n_async, generate_taste_profile,
    get_semantic_modifier,
)
from hybrid_model import rerank_candidates, hybrid_rating_prediction
from evaluator import compute_metrics, run_svd_cross_validation


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ── Pre-load data on startup ──────────────────────────────────────────────────

print("[*] Loading MovieLens data and training SVD...")
_metadata = load_item_metadata()
_models: dict[int, dict] = {}  # split -> {svd, R, train_df, test_df}


def _get_split(split: int = 1) -> dict:
    """Load and cache a split's SVD model and rating matrix."""
    if split not in _models:
        train_df, test_df = load_split(split)
        R = build_rating_matrix(train_df)
        svd = SVDModel()
        svd.fit(R)
        _models[split] = {
            "svd": svd,
            "R": R,
            "train_df": train_df,
            "test_df": test_df,
        }
    return _models[split]


# Pre-load split 1
_get_split(1)
print("[+] SVD model ready.")


# ── Static file serving ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── API: System config ───────────────────────────────────────────────────────

@app.route("/api/config")
def get_config():
    return jsonify({
        "alpha": ALPHA,
        "beta": BETA,
        "n_factors": N_FACTORS,
        "top_n_candidates": TOP_N_CANDIDATES,
        "llm_model": LLM_MODEL,
        "n_users": N_USERS,
        "n_items": N_ITEMS,
        "api_key_set": bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-")),
    })


# ── API: Movie search ────────────────────────────────────────────────────────

@app.route("/api/movies/search")
def search_movies():
    q = request.args.get("q", "").strip().lower()
    if not q or len(q) < 2:
        return jsonify([])

    results = []
    for item_id, meta in _metadata.items():
        if q in meta["title"].lower():
            results.append({
                "item_id": item_id,
                "title": meta["title"],
                "genres": meta["genres"],
            })
            if len(results) >= 20:
                break
    return jsonify(results)


# ── API: User taste profile ──────────────────────────────────────────────────

@app.route("/api/user/<int:user_id>")
def get_user(user_id: int):
    """Get user's taste profile and movie history."""
    if user_id < 1 or user_id > N_USERS:
        return jsonify({"error": f"User ID must be 1-{N_USERS}"}), 400

    split = int(request.args.get("split", 1))
    data = _get_split(split)
    R = data["R"]

    u_0 = user_id - 1
    loved, disliked = get_user_history(u_0, R, _metadata)
    all_rated = get_all_rated_movies(u_0, R, _metadata)

    # Count ratings by star level
    rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for m in all_rated:
        r = m["rating"]
        if r in rating_dist:
            rating_dist[r] += 1

    # Genre frequency
    genre_freq: dict[str, int] = {}
    for m in all_rated:
        for g in m["genres"]:
            genre_freq[g] = genre_freq.get(g, 0) + 1
    top_genres = sorted(genre_freq.items(), key=lambda x: x[1], reverse=True)[:8]

    return jsonify({
        "user_id": user_id,
        "total_ratings": len(all_rated),
        "loved": loved[:5],
        "disliked": disliked[:3],
        "rating_distribution": rating_dist,
        "top_genres": [{"genre": g, "count": c} for g, c in top_genres],
        "all_rated_count": len(all_rated),
    })


# ── API: Recommendations ─────────────────────────────────────────────────────

@app.route("/api/recommend", methods=["POST"])
def recommend():
    """Generate hybrid recommendations for a user."""
    body = request.get_json(force=True)
    user_id = body.get("user_id")
    split = body.get("split", 1)
    top_n = body.get("top_n", 10)

    if not user_id:
        user_id = random.randint(1, N_USERS)

    if user_id < 1 or user_id > N_USERS:
        return jsonify({"error": f"User ID must be 1-{N_USERS}"}), 400

    top_n = min(max(1, top_n), 50)

    try:
        data = _get_split(split)
        svd = data["svd"]
        R = data["R"]
        u_0 = user_id - 1

        loved, disliked = get_user_history(u_0, R, _metadata)

        # SVD candidates
        candidates = svd.top_n_recommendations(u_0, R, n=TOP_N_CANDIDATES)
        demo_candidates = candidates[:top_n]

        # Try async LLM reasoning
        try:
            # Generate taste profile
            all_rated = get_all_rated_movies(u_0, R, _metadata)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                taste_profile = loop.run_until_complete(
                    generate_taste_profile(u_0, all_rated)
                )

                reasoned = loop.run_until_complete(
                    batch_reason_top_n_async(
                        u_0, demo_candidates, loved, disliked, _metadata,
                        taste_profile=taste_profile,
                    )
                )
            finally:
                loop.close()

            reranked = rerank_candidates(reasoned)

        except Exception as llm_err:
            print(f"[!] LLM reasoning failed: {llm_err}")
            # Fallback: return SVD-only results
            reranked = []
            for item_id, cf_score in demo_candidates:
                meta = _metadata.get(item_id + 1, {})
                final = float(max(1.0, min(5.0, cf_score)))
                reranked.append({
                    "item_id": item_id,
                    "title": meta.get("title", f"Movie {item_id + 1}"),
                    "genres": meta.get("genres", []),
                    "cf_score": cf_score,
                    "semantic_modifier": 0.0,
                    "reasoning": "LLM unavailable — showing SVD-only prediction.",
                    "final_score": final,
                })
            taste_profile = "LLM unavailable."

        return jsonify({
            "user_id": user_id,
            "split": split,
            "alpha": ALPHA,
            "beta": BETA,
            "taste_profile": taste_profile,
            "loved": loved[:5],
            "disliked": disliked[:3],
            "recommendations": reranked[:top_n],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── API: SVD-only recommendations (no LLM, instant) ────────────────────────

@app.route("/api/recommend-svd", methods=["POST"])
def recommend_svd():
    """Fast SVD-only recommendations (no LLM cost)."""
    body = request.get_json(force=True)
    user_id = body.get("user_id")
    split = body.get("split", 1)
    top_n = body.get("top_n", 10)

    if not user_id:
        user_id = random.randint(1, N_USERS)

    if user_id < 1 or user_id > N_USERS:
        return jsonify({"error": f"User ID must be 1-{N_USERS}"}), 400

    top_n = min(max(1, top_n), 50)

    data = _get_split(split)
    svd = data["svd"]
    R = data["R"]
    u_0 = user_id - 1

    loved, disliked = get_user_history(u_0, R, _metadata)
    candidates = svd.top_n_recommendations(u_0, R, n=top_n)

    results = []
    for item_id, cf_score in candidates:
        meta = _metadata.get(item_id + 1, {})
        results.append({
            "item_id": item_id,
            "title": meta.get("title", f"Movie {item_id + 1}"),
            "genres": meta.get("genres", []),
            "cf_score": cf_score,
            "final_score": float(max(1.0, min(5.0, cf_score))),
        })

    return jsonify({
        "user_id": user_id,
        "split": split,
        "loved": loved[:5],
        "disliked": disliked[:3],
        "recommendations": results,
    })


# ── API: Evaluation ───────────────────────────────────────────────────────────

@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """Compare SVD vs Hybrid on sampled test pairs.

    Accepts an optional ``user_id`` (1-indexed).  When provided the
    evaluation runs exclusively on that user's held-out test ratings so
    each user produces unique metrics.  When omitted, a global multi-user
    evaluation is performed using the user-grouped sampling from main.py.
    """
    body = request.get_json(force=True)
    split = body.get("split", 1)
    sample_size = min(body.get("sample_size", 100), 500)
    user_id = body.get("user_id")          # optional — 1-indexed

    try:
        import numpy as np

        data = _get_split(split)
        svd = data["svd"]
        R = data["R"]
        test_df = data["test_df"]

        # ── Build test sample ────────────────────────────────────────────
        if user_id is not None:
            # ── Per-user evaluation ──────────────────────────────────────
            user_id = int(user_id)
            if user_id < 1 or user_id > N_USERS:
                return jsonify({"error": f"User ID must be 1-{N_USERS}"}), 400

            test_samp = test_df[test_df["user_id"] == user_id].reset_index(drop=True)
            if len(test_samp) == 0:
                return jsonify({
                    "error": f"User {user_id} has no test ratings in split u{split}."
                }), 400
            n_pairs = len(test_samp)
            eval_user_ids = [user_id]
        else:
            # ── Global multi-user sampling (mirrors main.py) ─────────────
            items_per_user = 10
            valid_users = (
                test_df.groupby("user_id")
                .filter(lambda x: len(x) >= 2)["user_id"]
                .unique()
            )
            n_users = max(1, sample_size // items_per_user)

            np.random.seed(42)
            sampled_users = np.random.choice(
                valid_users,
                size=min(n_users, len(valid_users)),
                replace=False,
            )
            test_samp = (
                test_df[test_df["user_id"].isin(sampled_users)]
                .groupby("user_id")
                .head(items_per_user)
                .reset_index(drop=True)
            )
            n_pairs = len(test_samp)
            eval_user_ids = list(sampled_users)

        # ── Taste profiles for evaluated users ───────────────────────────
        unique_users = test_samp["user_id"].unique()
        taste_profiles: dict[int, str] = {}

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _fetch_profiles():
                sem = asyncio.Semaphore(5)
                async def _fetch(u_1ind):
                    u_0 = u_1ind - 1
                    all_rated = get_all_rated_movies(u_0, R, _metadata)
                    async with sem:
                        return u_0, await generate_taste_profile(u_0, all_rated)
                return await asyncio.gather(*[_fetch(u) for u in unique_users])

            for u_idx, prof in loop.run_until_complete(_fetch_profiles()):
                taste_profiles[u_idx] = prof
            loop.close()
        except Exception as tp_err:
            print(f"[!] Taste profile generation failed: {tp_err}")

        # ── Per-pair evaluation ──────────────────────────────────────────
        y_true, y_svd, y_hybrid = [], [], []
        u_ids_test: list[int] = []

        for _, row in enumerate(test_samp.itertuples(index=False)):
            u_0 = row.user_id - 1
            i_0 = row.item_id - 1
            r_true = float(row.rating)

            cf_pred = svd.predict(u_0, i_0)

            loved, disliked = get_user_history(u_0, R, _metadata)
            meta = _metadata.get(i_0 + 1, {})
            target = {
                "title": meta.get("title", f"Movie {i_0 + 1}"),
                "genres": meta.get("genres", []),
            }

            try:
                llm = get_semantic_modifier(
                    u_0, i_0, loved, disliked, target,
                    taste_profile=taste_profiles.get(u_0),
                )
                mod = llm["semantic_modifier"]
            except Exception:
                mod = 0.0

            hybrid_pred = hybrid_rating_prediction(cf_pred, mod)

            y_true.append(r_true)
            y_svd.append(cf_pred)
            y_hybrid.append(hybrid_pred)
            u_ids_test.append(u_0)

        # ── Metrics (with ranking metrics via u_ids) ─────────────────────
        k_val = 3
        svd_metrics = compute_metrics(y_true, y_svd, u_ids_test, k=k_val, threshold=4.5)
        hybrid_metrics = compute_metrics(y_true, y_hybrid, u_ids_test, k=k_val, threshold=4.5)

        return jsonify({
            "split": split,
            "n_pairs": n_pairs,
            "n_users": int(len(unique_users)),
            "user_id": user_id,          # echoed back (None → global)
            "alpha": ALPHA,
            "beta": BETA,
            "svd_baseline": svd_metrics,
            "llm_hybrid": hybrid_metrics,
            "improvement": {
                "rmse": svd_metrics["RMSE"] - hybrid_metrics["RMSE"],
                "mae": svd_metrics["MAE"] - hybrid_metrics["MAE"],
                "nmae": svd_metrics["NMAE"] - hybrid_metrics["NMAE"],
                "ndcg": hybrid_metrics.get(f"NDCG@{k_val}", 0) - svd_metrics.get(f"NDCG@{k_val}", 0),
                "hr": hybrid_metrics.get(f"HR@{k_val}", 0) - svd_metrics.get(f"HR@{k_val}", 0),
                "k": k_val
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── API: SVD Cross-Validation ─────────────────────────────────────────────────

@app.route("/api/svd-cv")
def svd_cv():
    """Run SVD 5-split cross-validation."""
    try:
        results = []
        from data_loader import load_split as _ls, build_rating_matrix as _br
        import numpy as np

        all_rmse, all_mae, all_nmae = [], [], []
        all_ndcg, all_hr = [], []

        for split in range(1, 6):
            data = _get_split(split)
            svd = data["svd"]
            test_df = data["test_df"]

            y_true, y_pred, u_ids = [], [], []
            for row in test_df.itertuples(index=False):
                u, i, r = row.user_id - 1, row.item_id - 1, row.rating
                y_true.append(r)
                y_pred.append(svd.predict(u, i))
                u_ids.append(u)

            m = compute_metrics(y_true, y_pred, u_ids)
            results.append({
                "split": f"u{split}",
                "rmse": m["RMSE"],
                "mae": m["MAE"],
                "nmae": m["NMAE"],
                "ndcg": m.get("NDCG@10", 0.0),
                "hr": m.get("HR@10", 0.0),
            })
            all_rmse.append(m["RMSE"])
            all_mae.append(m["MAE"])
            all_nmae.append(m["NMAE"])
            all_ndcg.append(m.get("NDCG@10", 0.0))
            all_hr.append(m.get("HR@10", 0.0))

        avg = {
            "split": "Average",
            "rmse": float(np.mean(all_rmse)),
            "mae": float(np.mean(all_mae)),
            "nmae": float(np.mean(all_nmae)),
            "ndcg": float(np.mean(all_ndcg)),
            "hr": float(np.mean(all_hr)),
        }
        results.append(avg)

        return jsonify({"results": results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/hybrid-cv")
def hybrid_cv():
    """Run a full 5-split comparison (SVD vs Hybrid) with 50 users per split."""
    try:
        import numpy as np
        results = []
        n_users_per_split = 50
        items_per_user = 10
        k_val = 3
        thresh_val = 4.5

        all_svd_rmse, all_svd_ndcg, all_svd_hr = [], [], []
        all_hyb_rmse, all_hyb_ndcg, all_hyb_hr = [], [], []

        for split_idx in range(1, 6):
            data = _get_split(split_idx)
            svd = data["svd"]
            R = data["R"]
            test_df = data["test_df"]

            # ── Sample Users ─────────────────────────────────────────────
            valid_users = test_df.groupby("user_id").filter(lambda x: len(x) >= 2)["user_id"].unique()
            np.random.seed(42 + split_idx) # unique but deterministic seed per split
            sampled_users = np.random.choice(valid_users, size=min(n_users_per_split, len(valid_users)), replace=False)
            
            test_samp = test_df[test_df["user_id"].isin(sampled_users)].groupby("user_id").head(items_per_user)

            # ── Fetch Profiles (Async) ──────────────────────────────────
            taste_profiles = {}
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                async def _fetch_profiles():
                    sem = asyncio.Semaphore(5)
                    async def _fetch(u_1ind):
                        u_0 = u_1ind - 1
                        all_rated = get_all_rated_movies(u_0, R, _metadata)
                        async with sem: return u_0, await generate_taste_profile(u_0, all_rated)
                    return await asyncio.gather(*[_fetch(u) for u in sampled_users])
                for u_idx, prof in loop.run_until_complete(_fetch_profiles()):
                    taste_profiles[u_idx] = prof
                loop.close()
            except: pass

            # ── Evaluate Pairs (ASYNCHRONOUS BATCHING) ──────────────────
            y_true, y_svd, y_hybrid, u_ids = [], [], [], []
            
            async def _evaluate_pairs():
                sem = asyncio.Semaphore(20) # Process 20 movies at once
                async def _eval_row(row):
                    u_0, i_0, r_true = row.user_id - 1, row.item_id - 1, float(row.rating)
                    cf_pred = svd.predict(u_0, i_0)
                    loved, disliked = get_user_history(u_0, R, _metadata)
                    meta = _metadata.get(i_0 + 1, {})
                    target = {"title": meta.get("title", ""), "genres": meta.get("genres", [])}
                    
                    async with sem:
                        try:
                            # Use an async-friendly wrapper for get_semantic_modifier
                            llm = await loop.run_in_executor(None, get_semantic_modifier, 
                                                           u_0, i_0, loved, disliked, target, 
                                                           taste_profiles.get(u_0))
                            mod = llm["semantic_modifier"]
                        except: mod = 0.0
                    return r_true, cf_pred, hybrid_rating_prediction(cf_pred, mod), u_0

                return await asyncio.gather(*[_eval_row(row) for row in test_samp.itertuples(index=False)])

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results_batch = loop.run_until_complete(_evaluate_pairs())
                for rt, rsvd, rhyb, uid in results_batch:
                    y_true.append(rt); y_svd.append(rsvd); y_hybrid.append(rhyb); u_ids.append(uid)
                loop.close()
            except Exception as batch_err:
                print(f"[!] Batch evaluation failed: {batch_err}")

            m_svd = compute_metrics(y_true, y_svd, u_ids, k=k_val, threshold=thresh_val)
            m_hyb = compute_metrics(y_true, y_hybrid, u_ids, k=k_val, threshold=thresh_val)

            results.append({
                "split": f"u{split_idx}",
                "svd": m_svd,
                "hybrid": m_hyb,
                "improvement": m_svd["RMSE"] - m_hyb["RMSE"]
            })
            
            all_svd_rmse.append(m_svd["RMSE"])
            all_svd_ndcg.append(m_svd.get(f"NDCG@{k_val}", 0))
            all_svd_hr.append(m_svd.get(f"HR@{k_val}", 0))
            all_hyb_rmse.append(m_hyb["RMSE"])
            all_hyb_ndcg.append(m_hyb.get(f"NDCG@{k_val}", 0))
            all_hyb_hr.append(m_hyb.get(f"HR@{k_val}", 0))

        summary = {
            "split": "AVERAGE",
            "svd": {"RMSE": np.mean(all_svd_rmse), f"NDCG@{k_val}": np.mean(all_svd_ndcg), f"HR@{k_val}": np.mean(all_svd_hr)},
            "hybrid": {"RMSE": np.mean(all_hyb_rmse), f"NDCG@{k_val}": np.mean(all_hyb_ndcg), f"HR@{k_val}": np.mean(all_hyb_hr)},
            "k": k_val
        }
        
        return jsonify({"results": results, "summary": summary})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── API: All movies list ─────────────────────────────────────────────────────

@app.route("/api/movies")
def all_movies():
    """Return paginated movie catalog."""
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    start = (page - 1) * per_page

    items = sorted(_metadata.items(), key=lambda x: x[0])
    subset = items[start:start + per_page]

    return jsonify({
        "total": len(_metadata),
        "page": page,
        "per_page": per_page,
        "movies": [
            {"item_id": k, "title": v["title"], "genres": v["genres"]}
            for k, v in subset
        ],
    })


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  LLM-Augmented Hybrid Recommender — Web UI")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)
