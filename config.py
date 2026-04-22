"""
config.py — Central configuration for LLM-Augmented Hybrid Recommender
        MovieLens 100K × GPT-4o-mini

Set your OpenAI API key in a .env file (copy from .env.example):
    OPENAI_API_KEY=sk-...
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str      = "gpt-4o-mini"          # reasoning model
EMBED_MODEL: str    = "text-embedding-3-small"  # embedding model (optional)

# ── Dataset paths ─────────────────────────────────────────────────────────────
# ml-100k is in the same folder as this project
BASE_DIR: Path  = Path(__file__).parent
DATA_DIR: Path  = BASE_DIR / "ml-100k"

# ── Dataset constants ─────────────────────────────────────────────────────────
N_USERS: int  = 943
N_ITEMS: int  = 1682

# ── SVD hyperparameters ───────────────────────────────────────────────────────
N_FACTORS: int = 50   # latent factor dimensions

# ── LLM prompting settings ────────────────────────────────────────────────────
TOP_K_LOVED: int      = 5    # top-k loved movies fed into prompt
TOP_K_DISLIKED: int   = 2    # bottom-k disliked movies fed into prompt
TOP_N_CANDIDATES: int = 50   # SVD generates top-50; LLM reasons over them
LLM_CONCURRENCY: int  = 5    # max parallel async LLM calls

# ── Hybrid scoring weights ────────────────────────────────────────────────────
# Final_Score = ALPHA * CF_score + BETA * semantic_modifier
# BETA is kept small (0.3) so the LLM acts as a refinement signal.
# A large BETA would be overwhelmed by [1,5] clipping when CF is already near 5.
ALPHA: float = 1.0   # weight on collaborative filtering score
BETA: float  = 0.15  # weight on LLM semantic modifier (reduced from 0.3 for better stability)

# ── Rating scale ──────────────────────────────────────────────────────────────
MIN_RATING: float = 1.0
MAX_RATING: float = 5.0

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR: Path = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Results output ────────────────────────────────────────────────────────────
RESULTS_DIR: Path = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
