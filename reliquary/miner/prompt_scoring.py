"""Score-based OpenMath prompt selection for the miner pipeline.

This is the "score-based prompt selection" layer that sits *between* the
per-window slice (``window_prompt_range``) and the cooldown rejection in
``MiningEngine.mine_window``. It ranks the in-slice prompts by how *typical*
their features are of the current cooldown population — the prompts the
validator has already accepted and trained on — on the hypothesis that
feature-similar fresh prompts are likelier to land in-zone for the same
checkpoint.

Self-contained port of the standalone ``score/openmath_score.py`` scorer with
one deliberate change: records come from the miner's already-loaded OpenMath
environment dataset (the first-N OpenMathInstruct-2 shards in the HF cache),
NOT a separate ``train-*-of-00032.parquet`` copy. Reading the same rows the
env indexes guarantees ``prompt_idx`` aligns 1:1 with the validator's universe,
so scored picks never trip ``PROMPT_OUT_OF_RANGE`` / ``BAD_PROMPT_IDX``.

The feature extraction, bin labellers and mixing weights below are copied
verbatim from ``score/openmath_score.py`` so the two scorers stay numerically
identical; only the I/O layer (parquet glob + HTTP state fetch) is replaced.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

# ── mixing weights (mirror score/openmath_score.py) ─────────────────────────
TOTAL_W = {"problem": 0.05, "generated_solution": 0.9, "expected_answer": 0.05}
SOLUTION_W = {"chain_score": 0.9, "count_score": 0.05, "size_score": 0.05}
PROBLEM_W = {"count_score": 0.5, "size_score": 0.5}
ANSWER_W = {"type_score": 0.25, "sign_score": 0.25,
            "count_score": 0.25, "size_score": 0.25}


# ── feature extraction ──────────────────────────────────────────────────────
def number_stats(text: str) -> tuple[int, float]:
    """Return (count_of_numbers, magnitude_of_largest_number)."""
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text or "")]
    return len(nums), (max(nums) if nums else 0.0)


def classify_answer(ans: str | None) -> str:
    s = (ans or "").strip()
    if not s:
        return "empty"
    if re.fullmatch(r"-\d+", s):
        return "negative integer"
    if re.fullmatch(r"\d+", s):
        return "integer"
    if re.fullmatch(r"-?\d*\.\d+", s):
        return "decimal"
    if re.fullmatch(r"-?\d+/\d+", s):
        return "fraction"
    if re.search(r"[a-zA-Z\\^()=,]|\\frac|\\sqrt|pi|π", s):
        return "expression"
    return "other"


def numeric_value(ans: str | None) -> float | None:
    s = (ans or "").strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    if re.fullmatch(r"-?\d+/\d+", s):
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else None
    return None


def value_sign(ans: str | None) -> str:
    val = numeric_value(ans)
    if val is None:
        return "n/a (non-numeric)"
    if val == 0:
        return "zero"
    return "negative" if val < 0 else "positive"


def features_for(rec: dict) -> dict:
    """Raw (un-binned) features used for scoring."""
    prob = rec.get("problem") or ""
    sol = rec.get("generated_solution") or ""
    ans = rec.get("expected_answer")

    p_count, p_max = number_stats(prob)
    s_count, s_max = number_stats(sol)
    a_count, a_max = number_stats(ans or "")

    return {
        "cot_steps": sum(1 for ln in sol.splitlines() if ln.strip()),
        "prob_n_count": p_count,
        "prob_n_max": int(p_max),
        "sol_n_count": s_count,
        "sol_n_max": int(s_max),
        "ans_n_count": a_count,
        "ans_n_max": int(a_max),
        "answer_type": classify_answer(ans),
        "value_sign": value_sign(ans),
    }


# ── bin labellers ───────────────────────────────────────────────────────────
def _cot_bin(n: int) -> str:
    if n <= 30: return str(n)
    if n <= 35: return "31-35"
    if n <= 40: return "36-40"
    if n <= 50: return "41-50"
    return "51+"


def _nc_bin(n: int) -> str:
    if n <= 30: return str(n)
    if n <= 35: return "31-35"
    if n <= 40: return "36-40"
    if n <= 45: return "41-45"
    if n <= 50: return "46-50"
    if n <= 55: return "51-55"
    if n <= 60: return "56-60"
    return "61+"


def _sol_nc_bin(n: int) -> str:
    if n <= 60: return str(n)
    if n <= 65: return "61-65"
    if n <= 70: return "66-70"
    if n <= 75: return "71-75"
    if n <= 80: return "76-80"
    return "81+"


def _nm_bin(n: int) -> str:
    if n == 0: return "0"
    if n < 10: return "1-digit"
    if n < 100: return "2-digit"
    if n < 1000: return "3-digit"
    if n < 10000: return "4-digit"
    if n < 100000: return "5-digit"
    if n < 1000000: return "6-digit"
    return "7+"


def _ident(v: Any) -> str:
    return str(v)


FEATURE_BINS = {
    "cot_steps": ("cot_steps", _cot_bin),
    "prob_n_count": ("prob_n_count", _nc_bin),
    "prob_n_max": ("prob_n_max", _nm_bin),
    "sol_n_count": ("sol_n_count", _sol_nc_bin),
    "sol_n_max": ("sol_n_max", _nm_bin),
    "ans_n_count": ("ans_n_count", _nc_bin),
    "ans_n_max": ("ans_n_max", _nm_bin),
    "answer_type": ("answer_type", _ident),
    "value_sign": ("value_sign", _ident),
}


def bins_for(f: dict) -> dict:
    return {feat: fn(f[raw]) for feat, (raw, fn) in FEATURE_BINS.items()}


# ── env-backed record I/O (replaces score/'s parquet glob) ──────────────────
_COLS = ("problem", "generated_solution", "expected_answer", "problem_source")


def read_records(env, ids: list[int]) -> dict[int, dict]:
    """Return ``{prompt_idx: record}`` for every in-range id, read from the
    env's loaded HF dataset.

    Reads the underlying ``env._dataset`` directly via batched list indexing
    (one Arrow gather) so scoring 5000 in-slice candidates is a single read,
    not 5000 row lookups. Falls back to ``env.get_record`` when a test/env
    exposes that instead of a raw HF dataset. Out-of-range ids are dropped.
    """
    n = len(env)
    valid = [i for i in ids if 0 <= i < n]
    if not valid:
        return {}

    ds = getattr(env, "_dataset", None)
    if ds is not None:
        cols = ds[valid]  # HF Dataset list-index → {col: [values...]}
        out: dict[int, dict] = {}
        for pos, gid in enumerate(valid):
            out[gid] = {c: cols[c][pos] for c in _COLS if c in cols}
        return out

    # Fallback path for envs/tests that expose a per-id record accessor.
    get_record = getattr(env, "get_record", None)
    if get_record is None:
        raise TypeError("env exposes neither _dataset nor get_record")
    return {i: get_record(i) for i in valid}


# ── calibration over the cooldown population ────────────────────────────────
def build_calibration(env, cooldown_ids) -> dict:
    """Bin every cooldown id's features and return per-bin frequency tables.

    ``cal["features"][feature][bin]["pct"]`` is the % of the cooldown
    population in that bin — exactly the per-feature score a fresh prompt in
    the same bin receives. Equivalent to ``score/openmath_score.py``'s
    ``analyze_cooldown.json`` step, built in-memory from the live cooldown set.
    """
    recs = read_records(env, list(cooldown_ids))
    buckets = {feat: defaultdict(int) for feat in FEATURE_BINS}
    n = 0
    for rec in recs.values():
        for feat, label in bins_for(features_for(rec)).items():
            buckets[feat][label] += 1
        n += 1
    features = {
        feat: {label: {"count": c, "pct": (100.0 * c / n) if n else 0.0}
               for label, c in counts.items()}
        for feat, counts in buckets.items()
    }
    return {"n": n, "features": features}


def _pct(cal: dict, feat: str, label: str) -> float:
    return cal["features"].get(feat, {}).get(label, {}).get("pct", 0.0)


def score_record(rec: dict, cal: dict) -> float:
    """Return a 0-100 typicality score for one record against ``cal``."""
    b = bins_for(features_for(rec))

    prob_s = (_pct(cal, "prob_n_count", b["prob_n_count"]) * PROBLEM_W["count_score"]
              + _pct(cal, "prob_n_max", b["prob_n_max"]) * PROBLEM_W["size_score"])
    sol_s = (_pct(cal, "cot_steps", b["cot_steps"]) * SOLUTION_W["chain_score"]
             + _pct(cal, "sol_n_count", b["sol_n_count"]) * SOLUTION_W["count_score"]
             + _pct(cal, "sol_n_max", b["sol_n_max"]) * SOLUTION_W["size_score"])
    ans_s = (_pct(cal, "answer_type", b["answer_type"]) * ANSWER_W["type_score"]
             + _pct(cal, "value_sign", b["value_sign"]) * ANSWER_W["sign_score"]
             + _pct(cal, "ans_n_count", b["ans_n_count"]) * ANSWER_W["count_score"]
             + _pct(cal, "ans_n_max", b["ans_n_max"]) * ANSWER_W["size_score"])

    return (sol_s * TOTAL_W["generated_solution"]
            + prob_s * TOTAL_W["problem"]
            + ans_s * TOTAL_W["expected_answer"])


# ── smart pipeline: candidate ∩ window slice ────────────────────────────────
def eligible_in_slice(candidate_ids, prompt_range) -> list[int]:
    """Candidate ids that fall inside the per-window ``[lo, hi)`` slice.

    The "smart" pipeline intersects a precomputed candidate.json id set with
    the window slice the validator enforces, so only in-range candidates are
    submitted (out-of-range → ``PROMPT_OUT_OF_RANGE``).
    """
    lo, hi = prompt_range
    return [i for i in candidate_ids if lo <= i < hi]


# ── Layer 4: frontier signal ────────────────────────────────────────────────
def is_frontier_signal(signal) -> bool:
    """True if a probe reward signal is *mixed* (contains both a 0 and a 1).

    A uniform signal (all-0 = too hard, all-1 = too easy) predicts σ≈0, so the
    validator would reject the full 8-rollout group with ``OUT_OF_ZONE``. The
    Layer-4 probe uses this to skip such prompts before paying for the full
    group + GRAIL. Example: ``1110`` → True (select); ``0000``/``1111`` → False.
    """
    return len(set(signal)) >= 2


# ── the selection layer ─────────────────────────────────────────────────────
def rank_candidates(
    env,
    prompt_range: tuple[int, int],
    cooldown_ids,
    cal: dict,
    *,
    top: int,
    sources: set[str] | None = None,
) -> list[int]:
    """Score every in-slice, non-cooldown prompt and return the top-``top``
    ``prompt_idx`` values by descending score.

    ``prompt_range`` is the window slice ``[lo, hi)`` from
    ``window_prompt_range``; cooldown rejection is applied here so the new
    layer sits cleanly between the slice (Layer 2) and would-be Layer 3.
    ``sources`` optionally restricts to ``problem_source`` types.
    """
    lo, hi = prompt_range
    cooldown_set = set(cooldown_ids)
    candidates = [i for i in range(lo, hi) if i not in cooldown_set]
    if not candidates:
        return []

    recs = read_records(env, candidates)
    scored: list[tuple[float, int]] = []
    for gid in candidates:
        rec = recs.get(gid)
        if rec is None:
            continue
        if sources is not None and rec.get("problem_source") not in sources:
            continue
        scored.append((score_record(rec, cal), gid))

    # Sort by score desc, tie-break on idx asc for determinism across miners.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [gid for _, gid in scored[:top]]
