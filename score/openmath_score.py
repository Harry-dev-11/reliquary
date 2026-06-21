#!/usr/bin/env python3
"""
openmath_score.py — independent OpenMathInstruct-2 difficulty scorer.

Self-contained: NO imports from the other project files. Everything needed
(state fetching, feature extraction, bin labelling, calibration, mixing) is
inlined here. Only third-party / stdlib modules are used (pyarrow, urllib).

What it does
------------
The score of a prompt is how *typical* its features are of the current
"cooldown" population: for each feature, the score is the % of cooldown IDs
that fall in the same bin. Those per-feature percentages are mixed with fixed
weights into a single 0-100 total.

Pipeline (run automatically, or step-by-step):

  1. FETCH    GET the live state endpoint and save the cooldown prompt IDs
              ->  score/cooldown_id.json

  2. ANALYZE  read every cooldown ID from the local parquet shards, bin its
              features, and save the bin-frequency "feature-score" tables
              ->  score/analyze_cooldown.json

  3. SCORE    given a prompt_id, read that row, bin its features, look up each
              feature's bin % in analyze_cooldown.json, mix -> total score.

  4. RANGE    given an id range [START,END], drop every cooldown id inside it,
              score all the rest, rank by score, and write the candidates
              ->  score/candidate_ids.json

Usage
-----
    python3 score/openmath_score.py 12345          # score one prompt id
    python3 score/openmath_score.py 12 345 6789    # score several
    python3 score/openmath_score.py 12345 --json   # machine-readable
    python3 score/openmath_score.py 12345 --score-only   # just the number

    # range mode — score 30..5030 minus cooldown ids -> candidate_ids.json
    python3 score/openmath_score.py --range 30 5030
    python3 score/openmath_score.py --range 30 5030 --top 8     # keep top 8
    python3 score/openmath_score.py --range 30 5030 --score-only
    # keep only chosen source types, then take the top 8
    python3 score/openmath_score.py --range 30 5030 --top 8 --source math gsm8k

    python3 score/openmath_score.py --fetch        # (re)build cooldown_id.json
    python3 score/openmath_score.py --analyze      # (re)build analyze_cooldown.json
    python3 score/openmath_score.py --refresh 999  # force fetch+analyze, then score 999

Steps 1 and 2 run automatically the first time (when the JSON files are
missing); pass --fetch / --analyze / --refresh to force a rebuild.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict

import pyarrow.parquet as pq

# ── paths & constants ─────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # project root (holds shards)

SHARD_GLOB        = os.path.join(ROOT, "train-*-of-00032.parquet")
COOLDOWN_IDS_FILE = os.path.join(HERE, "cooldown_id.json")
ANALYZE_FILE      = os.path.join(HERE, "analyze_cooldown.json")
CANDIDATE_FILE    = os.path.join(HERE, "candidate_ids.json")
FALLBACK_IDS_FILE = os.path.join(ROOT, "cooldown_openmath.json")   # legacy backup

STATE_URL = "http://86.38.238.30:8080/state?env=openmathinstruct"

COLS = ["problem", "generated_solution", "expected_answer", "problem_source"]

# fixed mixing weights (mirrors the project's weights.json defaults) ─────────────
TOTAL_W    = {"problem": 0.05, "generated_solution": 0.9, "expected_answer": 0.05}
SOLUTION_W = {"chain_score": 0.9, "count_score": 0.05, "size_score": 0.05}
PROBLEM_W  = {"count_score": 0.5, "size_score": 0.5}
ANSWER_W   = {"type_score": 0.25, "sign_score": 0.25,
              "count_score": 0.25, "size_score": 0.25}


# ── 1. feature extraction (inlined from analyze_id.py) ──────────────────────────

def number_stats(text):
    """Return (count_of_numbers, magnitude_of_largest_number)."""
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text or "")]
    return len(nums), (max(nums) if nums else 0.0)


def classify_answer(ans):
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


def numeric_value(ans):
    """Float value of a numeric answer, or None if non-numeric."""
    s = (ans or "").strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    if re.fullmatch(r"-?\d+/\d+", s):
        n, d = s.split("/")
        return float(n) / float(d) if float(d) else None
    return None


def value_sign(ans):
    val = numeric_value(ans)
    if val is None:
        return "n/a (non-numeric)"
    if val == 0:
        return "zero"
    return "negative" if val < 0 else "positive"


def features_for(rec):
    """Raw (un-binned) features used for scoring."""
    prob = rec.get("problem") or ""
    sol = rec.get("generated_solution") or ""
    ans = rec.get("expected_answer")

    p_count, p_max = number_stats(prob)
    s_count, s_max = number_stats(sol)
    a_count, a_max = number_stats(ans or "")

    return {
        "cot_steps":   sum(1 for ln in sol.splitlines() if ln.strip()),
        "prob_n_count": p_count,
        "prob_n_max":   int(p_max),
        "sol_n_count":  s_count,
        "sol_n_max":    int(s_max),
        "ans_n_count":  a_count,
        "ans_n_max":    int(a_max),
        "answer_type":  classify_answer(ans),
        "value_sign":   value_sign(ans),
    }


# ── bin labellers (inlined from score.py) ───────────────────────────────────────

def _cot_bin(n):
    if n <= 30: return str(n)
    if n <= 35: return "31-35"
    if n <= 40: return "36-40"
    if n <= 50: return "41-50"
    return "51+"


def _nc_bin(n):
    if n <= 30: return str(n)
    if n <= 35: return "31-35"
    if n <= 40: return "36-40"
    if n <= 45: return "41-45"
    if n <= 50: return "46-50"
    if n <= 55: return "51-55"
    if n <= 60: return "56-60"
    return "61+"


def _sol_nc_bin(n):
    if n <= 60: return str(n)
    if n <= 65: return "61-65"
    if n <= 70: return "66-70"
    if n <= 75: return "71-75"
    if n <= 80: return "76-80"
    return "81+"


def _nm_bin(n):
    if n == 0:      return "0"
    if n < 10:      return "1-digit"
    if n < 100:     return "2-digit"
    if n < 1000:    return "3-digit"
    if n < 10000:   return "4-digit"
    if n < 100000:  return "5-digit"
    if n < 1000000: return "6-digit"
    return "7+"


def _ident(v):
    return str(v)


# feature key -> (raw key in features_for, bin function)
FEATURE_BINS = {
    "cot_steps":   ("cot_steps",   _cot_bin),
    "prob_n_count": ("prob_n_count", _nc_bin),
    "prob_n_max":   ("prob_n_max",   _nm_bin),
    "sol_n_count":  ("sol_n_count",  _sol_nc_bin),
    "sol_n_max":    ("sol_n_max",    _nm_bin),
    "ans_n_count":  ("ans_n_count",  _nc_bin),
    "ans_n_max":    ("ans_n_max",    _nm_bin),
    "answer_type":  ("answer_type",  _ident),
    "value_sign":   ("value_sign",   _ident),
}


def bins_for(f):
    """Map raw features -> {feature: bin_label}."""
    return {feat: fn(f[raw]) for feat, (raw, fn) in FEATURE_BINS.items()}


# ── shard I/O ───────────────────────────────────────────────────────────────────

_SHARD_CACHE = None


def load_shards():
    """Return (sorted shard files, rows_per_shard). Cached per process."""
    global _SHARD_CACHE
    if _SHARD_CACHE is not None:
        return _SHARD_CACHE
    files = sorted(glob.glob(SHARD_GLOB))
    if not files:
        raise FileNotFoundError(f"no shards match: {SHARD_GLOB}")
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    rps = counts[0]
    if len(set(counts[:-1])) > 1:
        raise ValueError(f"shards have unequal row counts: {counts}")
    _SHARD_CACHE = (files, rps)
    return _SHARD_CACHE


def read_records(ids):
    """Return {global_id: record_dict} for every in-range id."""
    files, rps = load_shards()
    max_rows = rps * len(files)
    by_shard = defaultdict(list)
    for gid in ids:
        if 0 <= gid < max_rows:
            by_shard[gid // rps].append(gid)
    out = {}
    for sh in sorted(by_shard):
        local = [g % rps for g in by_shard[sh]]
        tbl = pq.read_table(files[sh], columns=COLS).take(local)
        for gid, rec in zip(by_shard[sh], tbl.to_pylist()):
            out[gid] = rec
    return out


def max_id():
    files, rps = load_shards()
    return rps * len(files) - 1


# ── step 1: fetch cooldown IDs -> cooldown_id.json ──────────────────────────────

def fetch_state(timeout=30, retries=5):
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(STATE_URL, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"state GET failed after {retries} tries: {last}")


def step_fetch(force=False):
    """Step 1 — fetch live cooldown IDs and write cooldown_id.json.

    Returns the list of IDs. If --force is off and the file already exists it
    is simply loaded. If the live endpoint is unreachable, falls back to an
    existing cooldown_id.json, then to the legacy cooldown_openmath.json.
    """
    if not force and os.path.exists(COOLDOWN_IDS_FILE):
        ids = json.load(open(COOLDOWN_IDS_FILE))
        print(f"[fetch] using cached {COOLDOWN_IDS_FILE} ({len(ids)} IDs)")
        return ids

    try:
        print(f"[fetch] GET {STATE_URL}")
        state = fetch_state()
        ids = state.get("cooldown_prompts")
        if ids is None:
            raise KeyError("'cooldown_prompts' not in state response")
        print(f"[fetch] got {len(ids)} cooldown IDs (state={state.get('state')})")
    except Exception as e:
        print(f"[fetch] live endpoint failed ({e})")
        if os.path.exists(COOLDOWN_IDS_FILE):
            ids = json.load(open(COOLDOWN_IDS_FILE))
            print(f"[fetch] falling back to cached {COOLDOWN_IDS_FILE} ({len(ids)} IDs)")
            return ids
        if os.path.exists(FALLBACK_IDS_FILE):
            ids = json.load(open(FALLBACK_IDS_FILE))
            print(f"[fetch] falling back to {FALLBACK_IDS_FILE} ({len(ids)} IDs)")
        else:
            raise

    with open(COOLDOWN_IDS_FILE, "w") as f:
        json.dump(ids, f)
    print(f"[fetch] wrote {len(ids)} IDs -> {COOLDOWN_IDS_FILE}")
    return ids


# ── step 2: analyze cooldown IDs -> analyze_cooldown.json ───────────────────────

def step_analyze(ids=None, force=False):
    """Step 2 — bin every cooldown ID's features and write the per-bin
    frequency ("feature-score") tables to analyze_cooldown.json.

    Structure:
      {
        "n": <#cooldown IDs binned>,
        "generated_at": <unix ts>,
        "features": {
          "<feature>": { "<bin>": {"count": N, "pct": P}, ... },
          ...
        }
      }
    where pct = 100 * count / n  is exactly the per-feature score a prompt in
    that bin receives.
    """
    if not force and os.path.exists(ANALYZE_FILE):
        cal = json.load(open(ANALYZE_FILE))
        print(f"[analyze] using cached {ANALYZE_FILE} (n={cal.get('n')})")
        return cal

    if ids is None:
        ids = step_fetch()

    print(f"[analyze] binning features for {len(ids)} cooldown IDs…")
    recs = read_records(ids)
    buckets = {feat: defaultdict(int) for feat in FEATURE_BINS}
    n = 0
    for rec in recs.values():
        b = bins_for(features_for(rec))
        for feat, label in b.items():
            buckets[feat][label] += 1
        n += 1

    features = {}
    for feat, counts in buckets.items():
        features[feat] = {
            label: {"count": c, "pct": round(100.0 * c / n, 4) if n else 0.0}
            for label, c in sorted(counts.items(), key=lambda kv: -kv[1])
        }

    cal = {"n": n, "generated_at": int(time.time()), "features": features}
    with open(ANALYZE_FILE, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"[analyze] binned {n} IDs across {len(features)} features "
          f"-> {ANALYZE_FILE}")
    return cal


# ── step 3: score a prompt id ───────────────────────────────────────────────────

def _pct(cal, feat, label):
    """% of cooldown IDs in the same bin (0 if the bin is unseen)."""
    return cal["features"].get(feat, {}).get(label, {}).get("pct", 0.0)


def score_record(rec, cal):
    """Return (total 0-100, detail dict) for one dataset record."""
    f = features_for(rec)
    b = bins_for(f)

    prob_count = _pct(cal, "prob_n_count", b["prob_n_count"])
    prob_size  = _pct(cal, "prob_n_max",   b["prob_n_max"])
    prob_s = prob_count * PROBLEM_W["count_score"] + prob_size * PROBLEM_W["size_score"]

    sol_chain = _pct(cal, "cot_steps",   b["cot_steps"])
    sol_count = _pct(cal, "sol_n_count", b["sol_n_count"])
    sol_size  = _pct(cal, "sol_n_max",   b["sol_n_max"])
    sol_s = (sol_chain * SOLUTION_W["chain_score"]
             + sol_count * SOLUTION_W["count_score"]
             + sol_size * SOLUTION_W["size_score"])

    ans_type  = _pct(cal, "answer_type", b["answer_type"])
    ans_sign  = _pct(cal, "value_sign",  b["value_sign"])
    ans_count = _pct(cal, "ans_n_count", b["ans_n_count"])
    ans_size  = _pct(cal, "ans_n_max",   b["ans_n_max"])
    ans_s = (ans_type * ANSWER_W["type_score"]
             + ans_sign * ANSWER_W["sign_score"]
             + ans_count * ANSWER_W["count_score"]
             + ans_size * ANSWER_W["size_score"])

    total = (sol_s * TOTAL_W["generated_solution"]
             + prob_s * TOTAL_W["problem"]
             + ans_s * TOTAL_W["expected_answer"])

    detail = {
        "generated_solution": {"score": sol_s, "features": {
            "chain_score": sol_chain, "count_score": sol_count, "size_score": sol_size}},
        "problem": {"score": prob_s, "features": {
            "count_score": prob_count, "size_score": prob_size}},
        "expected_answer": {"score": ans_s, "features": {
            "type_score": ans_type, "sign_score": ans_sign,
            "count_score": ans_count, "size_score": ans_size}},
    }
    return total, detail


def score_id(gid, cal):
    """Score a single prompt id. Returns a result dict or None if out of range."""
    recs = read_records([gid])
    if gid not in recs:
        return None
    rec = recs[gid]
    total, detail = score_record(rec, cal)
    return {
        "id": gid,
        "score": round(total, 2),
        "solution_score": round(detail["generated_solution"]["score"], 2),
        "problem_score":  round(detail["problem"]["score"], 2),
        "answer_score":   round(detail["expected_answer"]["score"], 2),
        "source": rec.get("problem_source", "?"),
        "answer": rec.get("expected_answer", "?"),
        "detail": detail,
    }


# ── step 4: score an ID RANGE, excluding cooldown IDs ───────────────────────────

SOURCE_TYPES = ["math", "augmented_math", "gsm8k", "augmented_gsm8k"]


def score_range(n1, n2, cooldown_ids, cal, top=None, sources=None,
                out=CANDIDATE_FILE):
    """Score every id in [n1, n2] that is NOT a cooldown id, rank by score,
    and write the candidates to candidate_ids.json.

    n1, n2        inclusive range bounds (global prompt ids)
    cooldown_ids  the cooldown population to exclude (from step 1)
    cal           the bin-frequency tables (from step 2)
    top           keep only the top-N hardest candidates (None = keep all)
    sources       keep only these problem_source types (None = keep all)
    """
    mx = max_id()
    n1 = max(0, n1)
    n2 = min(mx, n2)
    if n1 > n2:
        raise ValueError(f"empty range after clamping to [0,{mx}]: {n1} > {n2}")

    src_set = set(sources) if sources else None
    cooldown_set = set(cooldown_ids)
    span = n2 - n1 + 1
    candidates = [i for i in range(n1, n2 + 1) if i not in cooldown_set]
    excluded = span - len(candidates)
    print(f"[range] {n1}..{n2}  ({span} ids)  −  {excluded} cooldown  "
          f"=  {len(candidates)} to score"
          + (f"  (sources: {', '.join(sources)})" if src_set else ""))

    recs = read_records(candidates)
    rows = []
    skipped_src = 0
    for gid in candidates:
        rec = recs.get(gid)
        if rec is None:
            continue
        if src_set is not None and rec.get("problem_source") not in src_set:
            skipped_src += 1
            continue
        total, detail = score_record(rec, cal)
        rows.append({
            "id": gid,
            "score": round(total, 2),
            "solution_score": round(detail["generated_solution"]["score"], 2),
            "problem_score":  round(detail["problem"]["score"], 2),
            "answer_score":   round(detail["expected_answer"]["score"], 2),
            "source": rec.get("problem_source", "?"),
            "answer": rec.get("expected_answer", "?"),
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    matched = len(rows)
    if top:
        rows = rows[:top]

    if src_set is not None:
        print(f"[range] {matched} matched source filter "
              f"({skipped_src} skipped by source)")

    result = {
        "range": [n1, n2],
        "total_in_range": span,
        "cooldown_excluded": excluded,
        "source_filter": sorted(src_set) if src_set else "all",
        "matched_source": matched,
        "scored": len(rows),
        "generated_at": int(time.time()),
        "candidates": rows,
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[range] wrote {len(rows)} candidates -> {out}")
    return result


# ── pretty printing ─────────────────────────────────────────────────────────────

_LABELS = {
    "generated_solution": {"chain_score": "CoT steps", "count_score": "number count",
                           "size_score": "largest number"},
    "problem": {"count_score": "number count", "size_score": "largest number"},
    "expected_answer": {"type_score": "answer type", "sign_score": "value sign",
                        "count_score": "number count", "size_score": "largest number"},
}
_GROUP_W = {"generated_solution": SOLUTION_W, "problem": PROBLEM_W,
            "expected_answer": ANSWER_W}


def print_breakdown(res, cal):
    d = res["detail"]
    print("=" * 70)
    print(f"  ID {res['id']}   TOTAL SCORE  {res['score']:.2f} / 100")
    print(f"  source: {res['source']}   answer: {res['answer']}")
    print(f"  (feature score = % of {cal['n']} cooldown IDs in the same bin)")
    print("=" * 70)
    for grp, gmix in TOTAL_W.items():
        gs = d[grp]["score"]
        print(f"\n  [{grp}]  {gs:.2f} × {gmix} → {gs * gmix:.2f} pts")
        for fk, w in _GROUP_W[grp].items():
            v = d[grp]["features"][fk]
            print(f"    {_LABELS[grp][fk]:<16} {v:>6.2f}% × {w:<5g} = {v * w:>6.2f}")
    print(f"\n  TOTAL = {res['score']:.2f}\n")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", type=int,
                    help="prompt id(s) to score")
    ap.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                    help="score every id in [START,END] EXCEPT cooldown ids, "
                         "rank them, and write candidate_ids.json")
    ap.add_argument("--top", type=int, default=8,
                    help="with --range: keep only the top-N hardest candidates "
                         "(default: 8; use --top 0 to keep all)")
    ap.add_argument("--source", nargs="+", choices=SOURCE_TYPES, metavar="TYPE",
                    default=["augmented_math"],
                    help="with --range: keep only these problem_source types "
                         f"({', '.join(SOURCE_TYPES)}; default: augmented_math)")
    ap.add_argument("--out", default=CANDIDATE_FILE,
                    help=f"with --range: output file (default: {CANDIDATE_FILE})")
    ap.add_argument("--fetch", action="store_true",
                    help="step 1: force (re)build cooldown_id.json, then exit "
                         "if no ids given")
    ap.add_argument("--analyze", action="store_true",
                    help="step 2: force (re)build analyze_cooldown.json, then "
                         "exit if no ids given")
    ap.add_argument("--refresh", action="store_true",
                    help="force both step 1 and step 2 before scoring")
    ap.add_argument("--json", action="store_true",
                    help="print machine-readable JSON instead of a breakdown")
    ap.add_argument("--score-only", action="store_true",
                    help="print only the numeric score(s)")
    args = ap.parse_args()

    # steps 1 & 2 (with optional forced rebuild)
    force_fetch   = args.fetch or args.refresh
    force_analyze = args.analyze or args.refresh

    ids = step_fetch(force=force_fetch)
    cal = step_analyze(ids=ids, force=force_analyze)

    # ── range mode: score a whole range minus cooldown ids ──
    if args.range:
        n1, n2 = args.range
        result = score_range(n1, n2, ids, cal, top=args.top,
                             sources=args.source, out=args.out)
        if args.json:
            print(json.dumps(result, indent=2))
        elif args.score_only:
            for r in result["candidates"]:
                print(f"{r['id']}\t{r['score']:.2f}")
        else:
            print(f"\n  top {min(20, len(result['candidates']))} hardest "
                  f"candidates in {n1}..{n2}:")
            for r in result["candidates"][:20]:
                print(f"    id={r['id']:>8}  score={r['score']:>6.2f}  "
                      f"sol={r['solution_score']:>6.2f}  src={r['source']}")
        return

    # if the user only asked to (re)build the caches, stop here
    if not args.ids and (args.fetch or args.analyze or args.refresh):
        return

    if not args.ids:
        ap.error("no prompt id given — e.g. `openmath_score.py 12345`, "
                 "`openmath_score.py --range 30 5030`, "
                 "or --fetch / --analyze to just rebuild caches")

    mx = max_id()
    results = []
    for gid in args.ids:
        if not (0 <= gid <= mx):
            print(f"id {gid} out of range [0,{mx}] — skipped", file=sys.stderr)
            continue
        res = score_id(gid, cal)
        if res is None:
            print(f"id {gid} not found — skipped", file=sys.stderr)
            continue
        results.append(res)

    if args.json:
        out = [{k: v for k, v in r.items() if k != "detail"} for r in results]
        print(json.dumps(out if len(out) != 1 else out[0], indent=2))
    elif args.score_only:
        for r in results:
            print(f"{r['score']:.2f}")
    else:
        for r in results:
            print_breakdown(r, cal)


if __name__ == "__main__":
    main()
