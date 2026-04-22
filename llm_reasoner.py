"""
llm_reasoner.py — GPT-4o-mini Chain-of-Thought Reasoning Engine

For each (user, candidate_movie) pair this module:
  1. Builds a structured prompt using the user's loved/disliked history
     (or a pre-summarised taste profile for richer context)
  2. Calls GPT-4o-mini with JSON-mode enabled (guaranteed parseable output)
  3. Parses the returned {reasoning, semantic_modifier} JSON object
  4. Caches results to disk so repeated runs never re-bill the same pair

The semantic_modifier is a float in [-1.0, +1.0]:
  - +1.0 = strong thematic alignment with user's taste
  -  0.0 = neutral / uncertain
  - -1.0 = strong thematic mismatch

Usage:
    # Sync (original)
    from llm_reasoner import batch_reason_top_n
    reasoned = batch_reason_top_n(user_id, candidates, loved, disliked, metadata)

    # Async (new — faster)
    import asyncio
    from llm_reasoner import batch_reason_top_n_async, generate_taste_profile
    profile = asyncio.run(generate_taste_profile(user_id, all_rated))
    reasoned = asyncio.run(
        batch_reason_top_n_async(user_id, candidates, loved, disliked, metadata,
                                 taste_profile=profile)
    )
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from openai import OpenAI, AsyncOpenAI
from config import OPENAI_API_KEY, LLM_MODEL, CACHE_DIR, LLM_CONCURRENCY

# ── OpenAI clients ────────────────────────────────────────────────────────────
client       = OpenAI(api_key=OPENAI_API_KEY)
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ── Cache ─────────────────────────────────────────────────────────────────────
_CACHE_FILE = CACHE_DIR / "reasoning_cache.json"


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        with open(_CACHE_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_cache(cache: dict) -> None:
    with open(_CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False)


def _cache_key(user_id: int, item_id: int) -> str:
    return f"u{user_id}_i{item_id}"


def _taste_cache_key(user_id: int) -> str:
    return f"taste_u{user_id}"


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert cinematic recommender system. "
    "Your job is to analyze a user's movie viewing history and evaluate how well a target movie aligns with their latent preferences. "
    "GUIDELINES: "
    "1. Be conservative. 0.0 is the neutral default. "
    "2. Only use modifiers > 0.5 or < -0.5 if there is undeniable thematic evidence. "
    "3. If the genres are a mix of liked and disliked, stay near 0.0. "
    "Always respond with a valid JSON object containing exactly two keys: "
    "\"reasoning\" (string, max 3 sentences) and \"semantic_modifier\" (float in [-1.0, 1.0])."
)

TASTE_SYSTEM_PROMPT = (
    "You are an expert cinematic analyst. Given a user's complete movie rating "
    "history, you distill their taste into a concise profile. "
    "Always respond with a valid JSON object containing exactly one key: "
    "\"taste_profile\" (string, 3-4 bullet points)."
)


def _build_user_prompt(
    user_id: int,
    loved: list[dict],
    disliked: list[dict],
    target: dict,
) -> str:
    def _fmt(movies: list[dict]) -> str:
        if not movies:
            return "  - (none on record)"
        lines = []
        for m in movies:
            genres = ", ".join(m.get("genres", [])) or "Unknown"
            lines.append(f"  - {m['title']} (Genres: {genres}) — {m['rating']}★")
        return "\n".join(lines)

    target_genres = ", ".join(target.get("genres", [])) or "Unknown"

    return f"""User {user_id + 1} Rating History:

Loved (4-5 stars):
{_fmt(loved)}

Disliked (1-2 stars):
{_fmt(disliked)}

Target Movie to Evaluate: {target['title']} (Genres: {target_genres})

Task:
1. Think step-by-step (Chain of Thought) about the underlying themes of the movies the user loved vs. disliked.
2. Compare these themes to the Target Movie's genres and what you know about its plot and style.
3. Predict a "Semantic Modifier" between -1.0 and +1.0 indicating how strongly the user's taste aligns with the Target Movie.
   - Use 0.0 as the conservative default for neutral or mixed alignment.
   - Use small values (+/- 0.1 to 0.3) for minor alignment/mismatch.
   - Use large values (+/- 0.7 to 1.0) ONLY for extreme cases.

Output strictly as a JSON object:
{{
  "reasoning": "string (max 3 sentences explaining the thematic alignment)",
  "semantic_modifier": float
}}"""


def _build_user_prompt_with_profile(
    user_id: int,
    taste_profile: str,
    target: dict,
) -> str:
    """Prompt variant that uses a pre-summarised taste profile (Fix 2A)."""
    target_genres = ", ".join(target.get("genres", [])) or "Unknown"

    return f"""User {user_id + 1} Taste Profile (derived from their full rating history):

{taste_profile}

Target Movie to Evaluate: {target['title']} (Genres: {target_genres})

Task:
1. Think step-by-step about how well the Target Movie matches this user's taste profile.
2. Predict a "Semantic Modifier" between -1.0 and +1.0 indicating alignment.
   - Use 0.0 as the conservative default.
   - Be subtle; only use extreme values (+/- 1.0) if the match is unmistakable.

Output strictly as a JSON object:
{{
  "reasoning": "string (max 3 sentences explaining the thematic alignment)",
  "semantic_modifier": float
}}"""


# ── Taste profile generation (Fix 2A) ────────────────────────────────────────

def _build_taste_profile_prompt(user_id: int, all_rated: list[dict]) -> str:
    """Build a prompt asking the LLM to summarise a user's full history."""
    lines = []
    for m in all_rated:
        genres = ", ".join(m.get("genres", [])) or "Unknown"
        lines.append(f"  - {m['title']} (Genres: {genres}) — {m['rating']}★")
    history = "\n".join(lines)

    return f"""User {user_id + 1} has rated {len(all_rated)} movies. Here is their complete history
(sorted from highest to lowest rating):

{history}

Task:
Summarise this user's cinematic taste in 3-4 concise bullet points. Cover:
  - Preferred genres and themes
  - Narrative style preferences (e.g., complex plots, action-driven, character studies)
  - What they clearly avoid or dislike

Output as JSON:
{{
  "taste_profile": "• bullet 1\\n• bullet 2\\n• bullet 3\\n• bullet 4"
}}"""


async def generate_taste_profile(
    user_id: int,
    all_rated: list[dict],
    use_cache: bool = True,
) -> str:
    """Generate a concise taste profile from a user's full rating history.

    Returns a multiline string of bullet points. Result is cached.
    """
    key   = _taste_cache_key(user_id)
    cache = _load_cache()

    if use_cache and key in cache:
        return cache[key]

    prompt = _build_taste_profile_prompt(user_id, all_rated)

    for attempt in range(3):
        try:
            response = await async_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": TASTE_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=400,
                response_format={"type": "json_object"},
            )

            raw    = response.choices[0].message.content
            result = json.loads(raw)
            profile = str(result.get("taste_profile", "No profile generated."))

            cache[key] = profile
            _save_cache(cache)
            return profile

        except Exception as exc:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"\n  [Taste profile retry {attempt+1}] {exc}. Waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"\n  [Taste profile failed] u{user_id}: {exc}")
                return "No taste profile available."


# ── Core API call (sync — original) ──────────────────────────────────────────

def get_semantic_modifier(
    user_id: int,
    item_id: int,
    loved: list[dict],
    disliked: list[dict],
    target: dict,
    use_cache: bool = True,
    taste_profile: str | None = None,
) -> dict:
    """Return {reasoning: str, semantic_modifier: float} for one (user, item) pair.

    Args:
        user_id, item_id: 0-indexed
        loved: list of {title, genres, rating} — user's top-rated movies
        disliked: list of {title, genres, rating} — user's low-rated movies
        target: {title, genres} — the movie to evaluate
        use_cache: skip API call if result already cached
        taste_profile: if provided, use profile-based prompt (Fix 2A)

    Returns:
        {"reasoning": str, "semantic_modifier": float in [-1, 1]}
    """
    key   = _cache_key(user_id, item_id)
    cache = _load_cache()

    if use_cache and key in cache:
        return cache[key]

    if taste_profile:
        prompt = _build_user_prompt_with_profile(user_id, taste_profile, target)
    else:
        prompt = _build_user_prompt(user_id, loved, disliked, target)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=350,
                response_format={"type": "json_object"},
            )

            raw    = response.choices[0].message.content
            result = json.loads(raw)

            # Validate and clamp modifier
            modifier = float(result.get("semantic_modifier", 0.0))
            modifier = max(-1.0, min(1.0, modifier))

            output = {
                "reasoning":         str(result.get("reasoning", "No reasoning provided.")),
                "semantic_modifier": modifier,
            }

            # Persist to cache
            cache[key] = output
            _save_cache(cache)
            return output

        except Exception as exc:
            if attempt < 2:
                wait = 2 ** attempt
                print(f"\n  [LLM retry {attempt+1}] {exc}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  [LLM failed] u{user_id} i{item_id}: {exc}")
                return {"reasoning": "LLM call failed — using neutral modifier.",
                        "semantic_modifier": 0.0}


# ── Core API call (async — Fix 1A) ───────────────────────────────────────────

async def _get_semantic_modifier_async(
    sem: asyncio.Semaphore,
    user_id: int,
    item_id: int,
    loved: list[dict],
    disliked: list[dict],
    target: dict,
    cache: dict,
    taste_profile: str | None = None,
) -> dict:
    """Async version of get_semantic_modifier with semaphore-based concurrency."""
    key = _cache_key(user_id, item_id)

    if key in cache:
        return cache[key]

    if taste_profile:
        prompt = _build_user_prompt_with_profile(user_id, taste_profile, target)
    else:
        prompt = _build_user_prompt(user_id, loved, disliked, target)

    async with sem:
        for attempt in range(3):
            try:
                response = await async_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=350,
                    response_format={"type": "json_object"},
                )

                raw    = response.choices[0].message.content
                result = json.loads(raw)

                modifier = float(result.get("semantic_modifier", 0.0))
                modifier = max(-1.0, min(1.0, modifier))

                output = {
                    "reasoning":         str(result.get("reasoning", "No reasoning provided.")),
                    "semantic_modifier": modifier,
                }

                cache[key] = output
                return output

            except Exception as exc:
                if attempt < 2:
                    wait = 2 ** attempt
                    print(f"\n  [LLM async retry {attempt+1}] {exc}. Waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"\n  [LLM async failed] u{user_id} i{item_id}: {exc}")
                    return {"reasoning": "LLM call failed — using neutral modifier.",
                            "semantic_modifier": 0.0}


# ── Batch reasoning (sync — original) ────────────────────────────────────────

def batch_reason_top_n(
    user_id: int,
    candidates: list[tuple[int, float]],        # (item_id_0indexed, cf_score)
    loved: list[dict],
    disliked: list[dict],
    metadata: dict,
    verbose: bool = True,
    taste_profile: str | None = None,
) -> list[dict]:
    """Run LLM reasoning over a list of candidate (item_id, cf_score) tuples.

    Returns a list of enriched dicts:
        {item_id, title, genres, cf_score, reasoning, semantic_modifier}
    """
    results = []
    total   = len(candidates)

    for idx, (item_id, cf_score) in enumerate(candidates):
        meta   = metadata.get(item_id + 1, {})
        target = {
            "title" : meta.get("title",  f"Movie {item_id + 1}"),
            "genres": meta.get("genres", []),
        }

        if verbose:
            pct = int((idx + 1) / total * 30)
            bar = "#" * pct + "." * (30 - pct)
            print(f"  [{bar}] {idx+1}/{total}  {target['title'][:40]:<40}", end="\r")

        llm = get_semantic_modifier(
            user_id, item_id, loved, disliked, target,
            taste_profile=taste_profile,
        )

        results.append({
            "item_id"          : item_id,
            "title"            : target["title"],
            "genres"           : target["genres"],
            "cf_score"         : cf_score,
            "reasoning"        : llm["reasoning"],
            "semantic_modifier": llm["semantic_modifier"],
        })

    if verbose:
        print()   # clear the \r progress line

    return results


# ── Batch reasoning (async — Fix 1A) ─────────────────────────────────────────

async def batch_reason_top_n_async(
    user_id: int,
    candidates: list[tuple[int, float]],
    loved: list[dict],
    disliked: list[dict],
    metadata: dict,
    verbose: bool = True,
    taste_profile: str | None = None,
) -> list[dict]:
    """Async version of batch_reason_top_n using concurrent API calls (Fix 1A).

    Uses a semaphore to limit concurrency to LLM_CONCURRENCY.
    Saves cache once after all calls complete.
    """
    sem   = asyncio.Semaphore(LLM_CONCURRENCY)
    cache = _load_cache()

    # Build targets
    targets = []
    for item_id, cf_score in candidates:
        meta = metadata.get(item_id + 1, {})
        targets.append({
            "item_id" : item_id,
            "cf_score": cf_score,
            "title"   : meta.get("title",  f"Movie {item_id + 1}"),
            "genres"  : meta.get("genres", []),
        })

    if verbose:
        print(f"  Launching {len(targets)} async LLM calls (concurrency={LLM_CONCURRENCY})...")

    # Fire all calls concurrently
    tasks = [
        _get_semantic_modifier_async(
            sem, user_id, t["item_id"], loved, disliked,
            {"title": t["title"], "genres": t["genres"]},
            cache, taste_profile,
        )
        for t in targets
    ]
    llm_results = await asyncio.gather(*tasks)

    # Persist cache once
    _save_cache(cache)

    # Assemble output
    results = []
    for t, llm in zip(targets, llm_results):
        results.append({
            "item_id"          : t["item_id"],
            "title"            : t["title"],
            "genres"           : t["genres"],
            "cf_score"         : t["cf_score"],
            "reasoning"        : llm["reasoning"],
            "semantic_modifier": llm["semantic_modifier"],
        })

    if verbose:
        print(f"  All {len(targets)} calls complete.")

    return results
