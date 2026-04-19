"""
data_loader.py — MovieLens 100K data utilities

Handles:
  - Loading train/test splits (u1–u5)
  - Building the 943×1682 rating matrix
  - Parsing u.item for movie titles and genres
  - Extracting a user's loved/disliked history for LLM prompts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_DIR, N_USERS, N_ITEMS, TOP_K_LOVED, TOP_K_DISLIKED

# ── Genre list (from MovieLens u.genre, positional in u.item) ──────────────────
GENRE_LIST = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]

_COLS = ["user_id", "item_id", "rating", "timestamp"]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_split(split: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) for split u{split}. Columns: user_id, item_id, rating."""
    train = pd.read_csv(DATA_DIR / f"u{split}.base", sep="\t",
                        names=_COLS, encoding="latin-1")
    test  = pd.read_csv(DATA_DIR / f"u{split}.test", sep="\t",
                        names=_COLS, encoding="latin-1")
    return train, test


def load_all_ratings() -> pd.DataFrame:
    """Return complete u.data (all 100 000 interactions)."""
    return pd.read_csv(DATA_DIR / "u.data", sep="\t",
                       names=_COLS, encoding="latin-1")


def build_rating_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert a ratings DataFrame into a dense (N_USERS × N_ITEMS) matrix.
    IDs in the DataFrame are 1-indexed; matrix indices are 0-indexed.
    """
    R = np.zeros((N_USERS, N_ITEMS), dtype=np.float32)
    for row in df.itertuples(index=False):
        R[row.user_id - 1, row.item_id - 1] = row.rating
    return R


# ─────────────────────────────────────────────────────────────────────────────
# Item metadata
# ─────────────────────────────────────────────────────────────────────────────

def load_item_metadata() -> dict[int, dict]:
    """Parse u.item → {item_id (1-indexed): {"title": str, "genres": list[str]}}."""
    items: dict[int, dict] = {}
    item_file = DATA_DIR / "u.item"
    with open(item_file, encoding="latin-1") as fh:
        for line in fh:
            parts = line.strip().split("|")
            if len(parts) < 24:
                continue
            item_id = int(parts[0])
            title   = parts[1].strip()
            flags   = [int(x) for x in parts[5:24]]
            genres  = [g for g, f in zip(GENRE_LIST, flags) if f == 1]
            items[item_id] = {"title": title, "genres": genres}
    return items


# ─────────────────────────────────────────────────────────────────────────────
# User history extraction (for LLM prompt construction)
# ─────────────────────────────────────────────────────────────────────────────

def get_user_history(
    user_id: int,          # 0-indexed
    R: np.ndarray,
    metadata: dict,
    top_loved: int    = TOP_K_LOVED,
    top_disliked: int = TOP_K_DISLIKED,
) -> tuple[list[dict], list[dict]]:
    """Return (loved_movies, disliked_movies) for a user.

    Each element is a dict: {title, genres, rating}.
    user_id is 0-indexed.
    """
    rated_mask  = R[user_id] > 0
    rated_items = np.where(rated_mask)[0]

    if len(rated_items) == 0:
        return [], []

    ratings     = R[user_id, rated_items]
    sorted_desc = np.argsort(ratings)[::-1]   # highest to lowest

    def _build(indices: np.ndarray) -> list[dict]:
        result = []
        for idx in indices:
            item_id = int(rated_items[idx]) + 1          # 1-indexed
            meta    = metadata.get(item_id, {})
            result.append({
                "title" : meta.get("title",  f"Movie {item_id}"),
                "genres": meta.get("genres", []),
                "rating": int(ratings[idx]),
            })
        return result

    # Loved: take up to top_loved from the highest-rated (≥ 3★ preferred)
    loved_indices    = sorted_desc[:top_loved]
    # Disliked: take up to top_disliked from the lowest-rated (filter ≤ 3★)
    disliked_indices = sorted_desc[-top_disliked:] if len(sorted_desc) >= top_disliked else sorted_desc

    loved    = _build(loved_indices)
    disliked = _build(disliked_indices)

    # Only keep genuinely disliked (≤ 2) if available
    disliked = [m for m in disliked if m["rating"] <= 2]

    return loved, disliked


def get_all_rated_movies(
    user_id: int,          # 0-indexed
    R: np.ndarray,
    metadata: dict,
) -> list[dict]:
    """Return *all* rated movies for a user, sorted highest-rating first.

    Each element is a dict: {title, genres, rating}.
    Used for taste-profile summarisation (Fix 2A).
    """
    rated_mask  = R[user_id] > 0
    rated_items = np.where(rated_mask)[0]

    if len(rated_items) == 0:
        return []

    ratings     = R[user_id, rated_items]
    sorted_desc = np.argsort(ratings)[::-1]

    result = []
    for idx in sorted_desc:
        item_id = int(rated_items[idx]) + 1   # 1-indexed
        meta    = metadata.get(item_id, {})
        result.append({
            "title" : meta.get("title",  f"Movie {item_id}"),
            "genres": meta.get("genres", []),
            "rating": int(ratings[idx]),
        })
    return result
