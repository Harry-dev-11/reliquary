#!/usr/bin/env python3
"""Custom standalone rollout pipeline.

Strips the miner down to the bare loop: load the policy model once, take the
prompt ids you hand it from ``openmath.json`` / ``opencode.json``, generate N
rollouts per prompt (default 4), recompute the local reward for each, and record
the wall-clock cost of every stage (model download/load, dataset load, per-prompt
generation, per-rollout reward).

It deliberately drops everything the real miner does that you don't need here:
no validator polling, no drand, no GRAIL proofs, no checkpoint pulls, no chain
submission. The generation params (temperature/top_p/top_k) and the prompt-id ->
text mapping are kept identical to the protocol so the rewards are comparable.

OpenCode rewards are validator-authoritative in production and need the grader
server over a Unix socket. When opencode ids are present this script auto-launches
the grader UNSANDBOXED (RELIQUARY_ALLOW_UNSANDBOXED_GRADER=1) — only run this on
a throwaway/isolated box.

Usage:
    python scripts/run_custom_rollouts.py \
        --checkpoint Qwen/Qwen3.5-4B \
        --openmath openmath.json \
        --opencode opencode.json \
        --n-rollouts 4 \
        --max-new-tokens 1024 \
        --out custom_rollouts_result.json

``--checkpoint`` may be a local path or an HF repo id; if it is an HF repo the
first load includes the download, which is captured in ``timings.model_load_s``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

# Make `import reliquary` work without the caller setting PYTHONPATH: this file
# lives in <repo>/scripts/, so the repo root (its parent) holds the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# OpenCode grading runs generated code; default to the unsandboxed grader so the
# script is self-contained on an isolated box. Override by exporting =0.
os.environ.setdefault("RELIQUARY_ALLOW_UNSANDBOXED_GRADER", "1")

logger = logging.getLogger("custom_rollouts")


def _load_ids(path: str) -> list[int]:
    """Read prompt ids from a json file.

    Accepts either {"id": [...]} (openmath.json) or {"ids": [...]} (opencode.json),
    or a bare list. Returns [] if the file is missing.
    """
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("ids", data.get("id", []))
    else:
        raw = []
    return [int(x) for x in raw]


def _generate_n_rollouts(
    model: Any,
    tokenizer: Any,
    prompt: str,
    n: int,
    max_new_tokens: int,
) -> tuple[list[dict], float]:
    """Generate ``n`` completions for one prompt in a single batched call.

    Mirrors miner.engine._generate_m_rollouts but with a configurable group
    size and no GRAIL bookkeeping. Returns (rollouts, generate_wall_seconds)
    where each rollout is {"tokens", "prompt_length"}.
    """
    import torch

    from reliquary.constants import T_PROTO, TOP_K_PROTO, TOP_P_PROTO
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.shared.modeling import first_eos_index, resolve_eos_token_ids

    prompt_tokens = encode_prompt(tokenizer, prompt)
    prompt_length = len(prompt_tokens)
    eos_ids = resolve_eos_token_ids(model, tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None and eos_ids:
        pad_token_id = min(eos_ids)

    t0 = time.perf_counter()
    with torch.no_grad():
        input_tensor = torch.tensor(
            [prompt_tokens] * n,
            device=getattr(model, "device", "cpu"),
        )
        attention_mask = torch.ones_like(input_tensor)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": T_PROTO,
            "top_p": TOP_P_PROTO,
            "top_k": TOP_K_PROTO,
            "pad_token_id": pad_token_id,
            "attention_mask": attention_mask,
        }
        if eos_ids:
            generate_kwargs["eos_token_id"] = sorted(eos_ids)
        outputs = model.generate(input_tensor, **generate_kwargs)
        # Block on the GPU so the timing reflects real generation cost.
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    generate_s = time.perf_counter() - t0

    rollouts = []
    for i in range(n):
        seq = outputs[i].tolist()
        gen = seq[prompt_length:]
        first_eos = first_eos_index(gen, eos_ids)
        if first_eos is not None:
            gen = gen[: first_eos + 1]
        rollouts.append({"tokens": prompt_tokens + gen, "prompt_length": prompt_length})
    return rollouts, generate_s


def _termination_status(tokens, eos_ids, cap):
    """Local mirror of the validator's verify_termination (minus the p_stop gate,
    which needs the GRAIL proof). natural_eos / cap_truncated are valid; anything
    else is bad_termination — what the validator rejects as BAD_TERMINATION."""
    if not tokens:
        return "bad_termination"
    if int(tokens[-1]) in eos_ids:
        return "natural_eos"
    if len(tokens) >= cap:
        return "cap_truncated"
    return "bad_termination"


def _process_env(
    env_name: str,
    ids: list[int],
    model: Any,
    tokenizer: Any,
    n_rollouts: int,
    max_new_tokens: int,
) -> tuple[list[dict], float]:
    """Run all prompt ids for one environment. Returns (records, env_load_s)."""
    from reliquary.environment import load_environment
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP, MAX_TRUNCATED_PER_SUBMISSION
    from reliquary.shared.modeling import resolve_eos_token_ids

    t0 = time.perf_counter()
    env = load_environment(env_name)  # triggers dataset download/load
    env_load_s = time.perf_counter() - t0
    logger.info("Loaded env %s (%d problems) in %.2fs", env_name, len(env), env_load_s)

    authoritative = bool(getattr(env, "validator_authoritative_reward", False))
    eos_ids = resolve_eos_token_ids(model, tokenizer)
    term_cap = MAX_NEW_TOKENS_PROTOCOL_CAP
    max_trunc = MAX_TRUNCATED_PER_SUBMISSION
    records: list[dict] = []

    for pid in ids:
        problem = env.get_problem(pid)
        rollouts, generate_s = _generate_n_rollouts(
            model, tokenizer, problem["prompt"], n_rollouts, max_new_tokens
        )

        roll_records = []
        rewards = []
        bad_term = 0
        for i, gen in enumerate(rollouts):
            completion_tokens = gen["tokens"][gen["prompt_length"] :]
            completion_text = tokenizer.decode(completion_tokens)
            rt0 = time.perf_counter()
            reward = env.compute_reward(problem, completion_text)
            reward_s = time.perf_counter() - rt0
            rewards.append(reward)
            term = _termination_status(gen["tokens"], eos_ids, term_cap)
            if term == "bad_termination":
                bad_term += 1
            roll_records.append({
                "index": i,
                "reward": reward,
                "completion_tokens": len(completion_tokens),
                "termination": term,
                "reward_s": round(reward_s, 4),
                "completion_preview": completion_text[:200],
            })

        # Validator-side termination verdict: the validator rejects the whole
        # batch (BAD_TERMINATION) when more than MAX_TRUNCATED_PER_SUBMISSION
        # rollouts fail to terminate naturally / at the cap.
        would_reject_term = bad_term > max_trunc
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        rec = {
            "env": env_name,
            "prompt_id": pid,
            "dataset_idx": pid % len(env),
            "problem_id": problem["id"],
            "ground_truth": str(problem.get("ground_truth", ""))[:80],
            "prompt_chars": len(problem["prompt"]),
            "validator_authoritative_reward": authoritative,
            "n_rollouts": n_rollouts,
            "generate_s": round(generate_s, 3),
            "per_rollout_gen_s": round(generate_s / n_rollouts, 3),
            "mean_reward": round(mean_reward, 4),
            "bad_termination": bad_term,
            "term_verdict": "BAD_TERMINATION" if would_reject_term else "ok",
            "rollouts": roll_records,
        }
        records.append(rec)
        if would_reject_term:
            logger.warning(
                "%s id=%d: %d/%d rollouts bad-terminated (> %d) — validator would "
                "reject as BAD_TERMINATION; raise --max-new-tokens",
                env_name, pid, bad_term, n_rollouts, max_trunc,
            )
        logger.info(
            "%s id=%d: mean_reward=%.3f  gen=%.2fs (%.2fs/rollout)  bad_term=%d  rewards=%s",
            env_name, pid, mean_reward, generate_s, generate_s / n_rollouts,
            bad_term, [round(r, 2) for r in rewards],
        )
    return records, env_load_s


def _write_markdown(result: dict, path: str) -> None:
    """Render the results dict as a markdown report with tables."""
    meta = result["meta"]
    t = result["timings"]
    records = result["prompts"]

    lines: list[str] = []
    lines.append("# Custom rollout results\n")

    # --- Run summary ---
    lines.append("## Run\n")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| checkpoint | `{meta['checkpoint']}` |")
    lines.append(f"| device | {meta['device']} |")
    lines.append(f"| rollouts / prompt | {meta['n_rollouts_per_prompt']} |")
    lines.append(f"| max_new_tokens | {meta['max_new_tokens']} |")
    lines.append(f"| temperature | {meta['temperature']} |")
    lines.append(f"| prompts | {meta['num_prompts']} |")
    lines.append(f"| total rollouts | {meta['total_rollouts']} |")
    lines.append("")

    # --- Timings ---
    lines.append("## Timings (seconds)\n")
    lines.append("| stage | seconds |")
    lines.append("|---|--:|")
    lines.append(f"| model load / download | {t['model_load_s']} |")
    for env_name, secs in t.get("env_load_s", {}).items():
        lines.append(f"| dataset load · {env_name} | {secs} |")
    lines.append(f"| total generation | {t['total_generation_s']} |")
    lines.append(f"| mean generation / prompt | {t['mean_generation_s_per_prompt']} |")
    lines.append("")

    # --- Per-env overall reward ---
    by_env: dict[str, list[float]] = {}
    for r in records:
        by_env.setdefault(r["env"], []).append(r["mean_reward"])
    lines.append("## Overall reward by environment\n")
    lines.append("| env | prompts | mean reward |")
    lines.append("|---|--:|--:|")
    for env_name, vals in by_env.items():
        lines.append(f"| {env_name} | {len(vals)} | {sum(vals)/len(vals):.4f} |")
    lines.append("")

    # --- Per-prompt summary ---
    lines.append("## Per-prompt summary\n")
    lines.append("| env | prompt_id | problem_id | mean_reward | rewards | term | gen_s | s/rollout |")
    lines.append("|---|--:|---|--:|---|---|--:|--:|")
    for r in records:
        rewards = ", ".join(f"{ro['reward']:.2f}" for ro in r["rollouts"])
        bt = r.get("bad_termination")
        term_cell = (f"{r.get('term_verdict','-')} ({bt})" if bt is not None
                     else r.get("term_verdict", "-"))
        lines.append(
            f"| {r['env']} | {r['prompt_id']} | `{r['problem_id']}` | "
            f"{r['mean_reward']:.3f} | {rewards} | {term_cell} | {r['generate_s']:.2f} | "
            f"{r['per_rollout_gen_s']:.2f} |"
        )
    lines.append("")

    # --- Per-rollout detail ---
    lines.append("## Per-rollout detail\n")
    lines.append("| env | prompt_id | rollout | reward | tokens | termination | reward_s |")
    lines.append("|---|--:|--:|--:|--:|---|--:|")
    for r in records:
        for ro in r["rollouts"]:
            lines.append(
                f"| {r['env']} | {r['prompt_id']} | {ro['index']} | "
                f"{ro['reward']:.3f} | {ro['completion_tokens']} | "
                f"{ro.get('termination','-')} | {ro['reward_s']} |"
            )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    from reliquary.constants import DEFAULT_BASE_MODEL, GRADER_SOCKET_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_BASE_MODEL,
                        help="Model path or HF repo id (default: %(default)s)")
    parser.add_argument("--openmath", default="openmath.json",
                        help="Path to openmath.json ({'id': [...]})")
    parser.add_argument("--opencode", default="opencode.json",
                        help="Path to opencode.json ({'ids': [...]})")
    parser.add_argument("--env", choices=["openmath", "opencode", "both"],
                        default="both",
                        help="Which environment(s) to run (default: both)")
    parser.add_argument("--openmath-data", default="",
                        help="Local OpenMath dataset path (save_to_disk dir, "
                             "parquet dir, or .parquet file). Overrides the HF download.")
    parser.add_argument("--opencode-data", default="",
                        help="Local OpenCode dataset path (save_to_disk dir). "
                             "Overrides the HF download.")
    parser.add_argument("--n-rollouts", type=int, default=4,
                        help="Rollouts to generate per prompt id (default: 4)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max completion tokens (protocol cap is 8192)")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only process the first N ids per file (smoke test)")
    parser.add_argument("--out", default="custom_rollouts_result.json",
                        help="Where to write the results json")
    parser.add_argument("--md", default="custom_rollouts_result.md",
                        help="Where to write the markdown table report")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Point the environments at local datasets you downloaded, if given. These
    # env vars are read when each Environment is constructed in _process_env.
    if args.openmath_data:
        os.environ["RELIQUARY_OMI_REPO"] = args.openmath_data
        logger.info("OpenMath dataset source: %s (local)", args.openmath_data)
    if args.opencode_data:
        os.environ["RELIQUARY_OCI_REPO"] = args.opencode_data
        logger.info("OpenCode dataset source: %s (local)", args.opencode_data)

    openmath_ids = _load_ids(args.openmath) if args.env in ("openmath", "both") else []
    opencode_ids = _load_ids(args.opencode) if args.env in ("opencode", "both") else []
    if args.limit > 0:
        openmath_ids = openmath_ids[: args.limit]
        opencode_ids = opencode_ids[: args.limit]
    logger.info("openmath ids: %d, opencode ids: %d", len(openmath_ids), len(opencode_ids))

    # --- OpenCode needs the grader server. Auto-launch it (unsandboxed). ---
    if opencode_ids:
        os.environ.setdefault("RELIQUARY_ALLOW_UNSANDBOXED_GRADER", "1")
        from reliquary.cli.main import _ensure_grader_running
        logger.info("OpenCode ids present — ensuring grader at %s", GRADER_SOCKET_PATH)
        _ensure_grader_running()

    # --- Load the policy model once (this is the "download" cost). ---
    import torch
    from reliquary.constants import ATTN_IMPLEMENTATION
    from reliquary.shared.modeling import load_text_generation_model, load_tokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model %s onto %s ...", args.checkpoint, device)

    t0 = time.perf_counter()
    tokenizer = load_tokenizer(args.checkpoint)
    model_kwargs = {"torch_dtype": torch.bfloat16}
    if device.startswith("cuda"):
        model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
    model = load_text_generation_model(args.checkpoint, **model_kwargs).to(device).eval()
    model_load_s = time.perf_counter() - t0
    logger.info("Model+tokenizer loaded/downloaded in %.2fs", model_load_s)

    timings: dict[str, Any] = {"model_load_s": round(model_load_s, 3), "env_load_s": {}}
    all_records: list[dict] = []

    if openmath_ids:
        recs, env_s = _process_env(
            "openmathinstruct", openmath_ids, model, tokenizer,
            args.n_rollouts, args.max_new_tokens,
        )
        all_records.extend(recs)
        timings["env_load_s"]["openmathinstruct"] = round(env_s, 3)

    if opencode_ids:
        recs, env_s = _process_env(
            "opencodeinstruct", opencode_ids, model, tokenizer,
            args.n_rollouts, args.max_new_tokens,
        )
        all_records.extend(recs)
        timings["env_load_s"]["opencodeinstruct"] = round(env_s, 3)

    total_gen_s = sum(r["generate_s"] for r in all_records)
    total_rollouts = sum(r["n_rollouts"] for r in all_records)
    result = {
        "meta": {
            "checkpoint": args.checkpoint,
            "device": device,
            "n_rollouts_per_prompt": args.n_rollouts,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.9,
            "num_prompts": len(all_records),
            "total_rollouts": total_rollouts,
        },
        "timings": {
            **timings,
            "total_generation_s": round(total_gen_s, 3),
            "mean_generation_s_per_prompt": round(total_gen_s / len(all_records), 3) if all_records else 0.0,
        },
        "prompts": all_records,
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    _write_markdown(result, args.md)

    # --- Human-readable summary ---
    print("\n" + "=" * 72)
    print(f"Model: {args.checkpoint}  device={device}")
    print(f"Model load/download: {model_load_s:.2f}s   "
          f"env load: {timings['env_load_s']}")
    print(f"Prompts: {len(all_records)}   rollouts/prompt: {args.n_rollouts}   "
          f"total rollouts: {total_rollouts}")
    print(f"Total generation: {total_gen_s:.2f}s   "
          f"mean/prompt: {result['timings']['mean_generation_s_per_prompt']:.2f}s")
    print("-" * 72)
    print(f"{'env':<18}{'id':>10}{'mean_rew':>10}{'term':>16}{'gen_s':>8}{'s/roll':>8}")
    for r in all_records:
        term = r.get("term_verdict", "ok")
        if r.get("bad_termination"):
            term = f"{term}({r['bad_termination']})"
        print(f"{r['env']:<18}{r['prompt_id']:>10}{r['mean_reward']:>10.3f}"
              f"{term:>16}{r['generate_s']:>8.2f}{r['per_rollout_gen_s']:>8.2f}")
    by_env: dict[str, list[float]] = {}
    for r in all_records:
        by_env.setdefault(r["env"], []).append(r["mean_reward"])
    n_bad = sum(1 for r in all_records if r.get("term_verdict") == "BAD_TERMINATION")
    print("-" * 72)
    for env_name, vals in by_env.items():
        print(f"{env_name}: overall mean reward = {sum(vals) / len(vals):.4f} "
              f"over {len(vals)} prompts")
    print(f"BAD_TERMINATION (would be rejected by validator): {n_bad}/{len(all_records)} prompts")
    print(f"\nWrote full results to {args.out}")
    print(f"Wrote markdown report to {args.md}")
    print("=" * 72)


if __name__ == "__main__":
    main()
