"""
svd_model.py — Bias-corrected SVD Collaborative Filtering

Implementation details:
  - Centers the rating matrix by removing global mean, user bias, and item bias
  - Decomposes the residual sparse matrix with scipy.sparse.linalg.svds (efficient)
  - Latent vectors P (users) and Q (items) are computed as U√Σ and Vt.T√Σ
  - Precomputes the full prediction matrix for O(1) lookups
  - top_n_recommendations() excludes already-rated items
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from config import N_FACTORS, MIN_RATING, MAX_RATING


class SVDModel:
    """Bias-corrected Matrix Factorisation via truncated SVD."""

    def __init__(self, n_factors: int = N_FACTORS):
        self.n_factors     = n_factors
        self.global_mean   = 0.0
        self.user_bias     = None   # shape (n_users,)
        self.item_bias     = None   # shape (n_items,)
        self.P             = None   # user latent matrix  (n_users × k)
        self.Q             = None   # item latent matrix  (n_items × k)
        self._pred_matrix  = None   # precomputed full prediction matrix

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, R: np.ndarray) -> "SVDModel":
        """Fit SVD on training matrix R (n_users × n_items, 0 = not rated)."""
        mask = R > 0
        R_f  = R.astype(float)

        # 1. Global mean
        self.global_mean = float(R_f[mask].mean())

        # 2. User bias (with shrinkage)
        R_c = np.zeros_like(R_f)
        R_c[mask] = R_f[mask] - self.global_mean

        n_users, n_items = R.shape
        self.user_bias   = np.zeros(n_users)
        damping_u = 15.0  # shrinkage for user biases
        for u in range(n_users):
            rated = mask[u]
            n_rated = rated.sum()
            if n_rated > 0:
                self.user_bias[u] = R_c[u, rated].sum() / (n_rated + damping_u)

        # 3. Item bias (after removing user bias, with shrinkage)
        R_ub = R_c.copy()
        for u in range(n_users):
            rated = mask[u]
            R_ub[u, rated] -= self.user_bias[u]

        self.item_bias = np.zeros(n_items)
        damping_i = 25.0  # shrinkage for item biases
        for i in range(n_items):
            rated_col = mask[:, i]
            n_rated = rated_col.sum()
            if n_rated > 0:
                self.item_bias[i] = R_ub[rated_col, i].sum() / (n_rated + damping_i)

        # 4. Residual matrix (no mean, no biases)
        R_res = R_ub.copy()
        for i in range(n_items):
            rated_col = mask[:, i]
            R_res[rated_col, i] -= self.item_bias[i]

        # 5. Truncated SVD on residual (sparse for efficiency)
        k = min(self.n_factors, min(R.shape) - 1)
        U, s, Vt = svds(csr_matrix(R_res), k=k)

        # Sort singular values descending
        order = np.argsort(s)[::-1]
        s, U, Vt = s[order], U[:, order], Vt[order, :]

        self.P = U  * np.sqrt(s)          # (n_users × k)
        self.Q = Vt.T * np.sqrt(s)        # (n_items × k)

        # 6. Precompute full prediction matrix (raw, unclipped — clip at predict time)
        self._pred_matrix = (
            self.P @ self.Q.T
            + self.global_mean
            + self.user_bias.reshape(-1, 1)
            + self.item_bias.reshape(1, -1)
        )

        return self

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for (user_id, item_id) — both 0-indexed."""
        return float(np.clip(self._pred_matrix[user_id, item_id],
                             MIN_RATING, MAX_RATING))

    def predict_batch(
        self, pairs: list[tuple[int, int]]
    ) -> list[float]:
        """Predict a list of (user_id, item_id) pairs efficiently."""
        users = [p[0] for p in pairs]
        items = [p[1] for p in pairs]
        raw   = self._pred_matrix[users, items]
        return list(np.clip(raw, MIN_RATING, MAX_RATING).astype(float))

    # ── Top-N candidacy ────────────────────────────────────────────────────────

    def top_n_recommendations(
        self,
        user_id: int,
        R: np.ndarray,
        n: int = 50,
        exclude_rated: bool = True,
    ) -> list[tuple[int, float]]:
        """Return top-n (item_id_0indexed, cf_score) tuples for user (0-indexed).

        cf_score is the RAW SVD prediction (may exceed 5.0) so that the LLM
        modifier can meaningfully differentiate between items that would all
        clip to 5.0 if rounded prematurely.
        """
        scores = self._pred_matrix[user_id].copy()

        if exclude_rated:
            scores[R[user_id] > 0] = -np.inf

        top_idx = np.argsort(scores)[::-1][:n]
        # Return raw (unclipped) scores so hybrid_model can add LLM signal first
        return [
            (int(idx), float(scores[idx]))
            for idx in top_idx
        ]

    # ── Latent vector access ───────────────────────────────────────────────────

    def latent_vector(self, user_id: int) -> np.ndarray:
        """Return the k-dimensional latent factor vector for a user."""
        return self.P[user_id]
