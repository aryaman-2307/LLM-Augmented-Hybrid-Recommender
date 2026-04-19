# LLM-Augmented Hybrid Recommendation System
**MovieLens 100K × GPT-4o-mini × SVD**

A hybrid recommender system that bridges traditional Collaborative Filtering (SVD) with
GPT-4o-mini Chain-of-Thought reasoning to produce explainable, semantically-aware recommendations.

---

## How it Works

```
                    ┌──────────────────────────┐
   MovieLens 100K   │   SVD (Collaborative     │  Top-50 candidates
   Rating Matrix ──>│   Filtering Baseline)    │─────────────────────┐
                    └──────────────────────────┘                     │
                                                                      v
                    ┌─────────────────────────────────────────────────────┐
  User History  ──> │  GPT-4o-mini (Chain-of-Thought JSON Reasoning)     │
  (loved/disliked)  │                                                     │
                    │  Input : "User loved Pulp Fiction, Reservoir Dogs"  │
                    │  Output: { reasoning: "...", semantic_modifier: +0.6}│
                    └─────────────────────────────────────────────────────┘
                                          │
                                          v
                    ┌──────────────────────────────────────────┐
                    │   Hybrid Scoring                         │
                    │   Final = alpha*CF_score + beta*LLM_mod  │
                    │   (default: alpha=1.0, beta=1.0)         │
                    └──────────────────────────────────────────┘
                                          │
                                          v
                    Top-N Re-ranked Recommendations + Explanations
```

### The Hybrid Formula
```
Final_Score = alpha × CF_Prediction + beta × LLM_Semantic_Modifier
```
- `CF_Prediction` ∈ [1, 5] — SVD collaborative filtering score
- `Semantic_Modifier` ∈ [-1, 1] — GPT-4o-mini thematic alignment signal
- Result clipped to [1, 5]

---

## Project Structure

```
llm_hybrid_recsys/
├── config.py          # API keys, paths, all hyperparameters
├── data_loader.py     # MovieLens parser, rating matrix builder, user history
├── svd_model.py       # Bias-corrected SVD (scipy.sparse.linalg.svds, k=50)
├── llm_reasoner.py    # GPT-4o-mini CoT prompts + disk cache
├── hybrid_model.py    # Hybrid scoring, re-ranking, weight tuning
├── evaluator.py       # RMSE / MAE / NMAE, comparison tables, CSV output
├── main.py            # CLI orchestrator (demo / evaluate / svd-cv)
├── requirements.txt
├── .env.example       # Copy to .env and add your API key
├── cache/             # Auto-created: LLM response cache (avoids re-billing)
└── results/           # Auto-created: CSVs from evaluation runs
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API key
```bash
copy .env.example .env
```
Edit `.env` and set:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Verify data path
The system expects MovieLens 100K at:
```
../SimarAhi_2023527_Assignment1/ml-100k/
```
If your `ml-100k` folder is elsewhere, update `DATA_DIR` in `config.py`.

---

## Usage

### Demo Mode (recommended first run)
Shows hybrid recommendations for one user with full LLM reasoning.
~10 API calls, < $0.01 cost.
```bash
python main.py                   # random user
python main.py --user 42         # specific user (1-indexed)
python main.py --user 42 --top-n 5
```

### Evaluation Mode
Compares SVD vs LLM-Hybrid on sampled test pairs. ~200 calls, ~$0.02.
```bash
python main.py --evaluate
python main.py --evaluate --sample 500 --split 2
python main.py --evaluate --sample 200 --tune    # grid-search best alpha/beta
```

### SVD Cross-Validation (no LLM — free)
Establishes the baseline across all 5 MovieLens splits.
```bash
python main.py --svd-cv
```

---

## SVD Baseline Results (verified)

| Split | RMSE   | MAE    | NMAE   |
|-------|--------|--------|--------|
| u1    | 0.9582 | 0.7515 | 0.1879 |
| u2    | 0.9490 | 0.7421 | 0.1855 |
| u3    | 0.9403 | 0.7372 | 0.1843 |
| u4    | 0.9408 | 0.7383 | 0.1846 |
| u5    | 0.9417 | 0.7433 | 0.1858 |
| **Avg** | **0.9460** | **0.7425** | **0.1856** |

---

## Sample Output (Demo Mode)

```
[*] Loading MovieLens split u1...
[*] Training SVD model... done.

================================================================
  USER 42 -- Taste Profile
----------------------------------------------------------------
  [+] Loved movies:
      5*  Pulp Fiction (1994)          [Crime, Drama]
      5*  Silence of the Lambs (1991)  [Drama, Thriller]
      4*  Seven (Se7en) (1995)         [Crime, Drama, Thriller]
================================================================

[*] Generating top-50 SVD candidates...
[*] Querying GPT-4o-mini (10 calls)...
  [##############################] 10/10  ...

[*] Applying hybrid scoring (alpha=1.0, beta=1.0)...

================================================================
  TOP 10 HYBRID RECOMMENDATIONS  --  User 42
================================================================

  #01  Usual Suspects, The (1995)
       Genres : Crime, Drama, Thriller, Mystery
       CF     : 4.35   LLM delta: +0.85   Final: 5.00
       Reason : User consistently gravitates toward crime and
                psychological thrillers with complex narratives;
                The Usual Suspects shares these exact themes.
...
```

---

## The LLM Prompt (CoT JSON)

**System prompt:**
```
You are an expert cinematic recommender system. Identify latent thematic
preferences and evaluate how well a target movie aligns with them.
```

**User prompt template:**
```
User 42 Rating History:

Loved (4-5 stars):
  - Pulp Fiction (1994) (Genres: Crime, Drama) — 5*
  - Silence of the Lambs (1991) (Genres: Drama, Thriller) — 4*

Disliked (1-2 stars):
  - Ace Ventura (1994) (Genres: Comedy) — 1*

Target Movie: Usual Suspects, The (1995) (Genres: Crime, Drama, Thriller)

Task:
1. Think step-by-step about themes the user loved vs. disliked.
2. Compare to Target Movie.
3. Output semantic_modifier in [-1.0, +1.0].

Output as JSON: {"reasoning": "...", "semantic_modifier": float}
```

---

## API Cost Reference

| Task                        | Model              | Calls | Est. Cost |
|-----------------------------|--------------------|-------|-----------|
| Demo (10 recs, 1 user)      | gpt-4o-mini        | 10    | < $0.01   |
| Evaluation (200 pairs)      | gpt-4o-mini        | 200   | ~$0.02    |
| Evaluation (500 pairs)      | gpt-4o-mini        | 500   | ~$0.05    |
| Full test set (~20k pairs)  | gpt-4o-mini        | 20k   | ~$1.00    |

> Results are cached in `cache/reasoning_cache.json` — re-running the same
> user/movie pair **never** makes a second API call.

---

## Configuration (`config.py`)

| Variable          | Default              | Description                        |
|-------------------|----------------------|------------------------------------|
| `N_FACTORS`       | 50                   | SVD latent dimensions              |
| `TOP_N_CANDIDATES`| 50                   | SVD candidates fed to LLM          |
| `TOP_K_LOVED`     | 5                    | Top-rated movies in prompt         |
| `TOP_K_DISLIKED`  | 2                    | Lowest-rated movies in prompt      |
| `ALPHA`           | 1.0                  | Weight on CF prediction            |
| `BETA`            | 1.0                  | Weight on LLM semantic modifier    |
| `LLM_MODEL`       | gpt-4o-mini          | OpenAI chat model                  |
