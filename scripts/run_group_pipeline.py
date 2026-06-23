#!/usr/bin/env python3
"""Group-aware rollout selection pipeline.

A smarter selection front-end than run_custom_rollouts.py. Instead of iterating
a flat id list, it walks a list of *key ids* and, for each, uses prompt groups +
cooldown + the live validator window slice to decide what (if anything) to roll
out. The flow per key id:

  1. env = openmath (default) or opencode.
  2. take the next key id from key_id.json.
  3. find which GROUP the key id belongs to (group.json); inside that group keep
     only ids NOT in the validator's LIVE cooldown right now (the cooldown_prompts
     from /state, not a static file) -> candidates -> candidate.json.
  4. fetch the live WINDOW SLICE [lo,hi) from the validator's randomness and keep
     candidates that fall inside it (= "accepted" ids). Select ONE accepted id —
     a different in-slice, non-cooldown member of the same group, never the key
     id itself (the key id only identifies the group). If none are accepted, go
     back to step 2 (next key id).
  5. generate N rollouts for the selected accepted id and score them. The group
     is IN-ZONE (trainable) when the count of reward-1 rollouts is in
     [zone_low, zone_high] (default 2..6 of 8 — not too hard, not too easy).
     OUT_OF_ZONE -> remove the key id from key_id.json -> back to step 2.
  6. (--submit only) for an in-zone group, build GRAIL proofs + merkle + signed
     envelope and POST /submit to the live validator. On a verdict of accepted,
     remove the key id from key_id.json; on failure, keep it and go to step 2.
     Needs a registered wallet + the validator's current checkpoint (loaded into
     a generation model and a proof model) + M_ROLLOUTS=8 rollouts.

This mirrors the protocol's three real constraints (group/cooldown, window slice,
trainable variance) without any GRAIL/submission. It is self-contained: it adds
the repo root to sys.path and defaults the unsandboxed grader on.

Data files (repo root):
  key_id.json          {"ids":[...]} or [...]   -- the key ids to walk
  group.part1.json     [{"group_id":1,"count":N,"ids":[...]}, ...]  (half the groups)
  group.part2.json     [{"group_id":...,"ids":[...]}, ...]          (other half)
                       group.json is split into these two parts so each stays
                       under GitHub's 100MB file limit; they merge at load time.
  candidate.json       written by this script (latest key id's candidates)
Cooldown is read live from the validator /state (cooldown_prompts), NOT from a
file, so it reflects the real cooldown at the moment of the run.

Usage:
  python scripts/run_group_pipeline.py \
      --env openmath \
      --validator-url http://127.0.0.1:8888 \
      --n-rollouts 4 \
      --md group_pipeline_result.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

# import reliquary without the caller setting PYTHONPATH (file lives in scripts/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("RELIQUARY_ALLOW_UNSANDBOXED_GRADER", "1")

# Reuse the batched generator from the simple pipeline.
from scripts.run_custom_rollouts import _generate_n_rollouts  # noqa: E402

logger = logging.getLogger("group_pipeline")

_ENV_FQN = {"openmath": "openmathinstruct", "opencode": "opencodeinstruct"}


def _load_ids(path: str) -> list[int]:
    """Read ids from {"ids":[...]}, {"id":[...]}, or a bare list. [] if missing/empty."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
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


def _key_container(path: str) -> str | None:
    """Detect how key_id.json wraps its ids so we can rewrite in the same shape.

    Returns the dict key ("ids"/"id") or None for a bare list. Defaults to "ids".
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return "ids"
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return None
    if isinstance(data, dict):
        if "ids" in data:
            return "ids"
        if "id" in data:
            return "id"
    return "ids"


def _rewrite_key_ids(path: str, ids: list[int], container_key: str | None) -> None:
    """Persist the remaining key ids back to disk in the original shape."""
    payload: Any = list(ids) if container_key is None else {container_key: list(ids)}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _build_group_index(
    group_paths: list[str],
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Return (id -> group_id, group_id -> ids) merged from one or more group
    files. group.json is split into group.part1.json + group.part2.json to stay
    under GitHub's 100MB file limit; each part is a JSON array of group objects
    and they merge back into the full index."""
    id2group: dict[int, int] = {}
    group_ids: dict[int, list[int]] = {}
    for path in group_paths:
        with open(path) as f:
            groups = json.load(f)
        for g in groups:
            gid = int(g["group_id"])
            ids = [int(x) for x in g.get("ids", [])]
            group_ids[gid] = ids
            for pid in ids:
                id2group[pid] = gid
    return id2group, group_ids


def _termination_status(tokens, prompt_length, eos_ids, cap):
    """Local mirror of the validator's verify_termination (minus the p_stop
    probability gate, which needs the GRAIL proof). A rollout terminates validly
    if its last token is a stop token (natural_eos) or the full sequence reached
    the protocol cap (cap_truncated); anything else is bad_termination — exactly
    what the validator rejects with RejectReason.BAD_TERMINATION."""
    if not tokens:
        return "bad_termination"
    if int(tokens[-1]) in eos_ids:
        return "natural_eos"
    if len(tokens) >= cap:
        return "cap_truncated"
    return "bad_termination"


def _fetch_window_state(
    validator_url: str, env_fqn: str, universe_n: int, retries: int, fallback_randomness: str,
) -> tuple[str, int, tuple[int, int], set[int]]:
    """Fetch the live validator window state and derive [lo,hi).

    Returns (randomness, window_n, (lo, hi), cooldown_ids) where ``cooldown_ids``
    is the validator's REAL per-env cooldown set right now (GrpoBatchState
    .cooldown_prompts) — not a static file. Polls a few times if randomness is
    still empty (validator between OPEN and randomness-set). Falls back to
    ``fallback_randomness`` if given and the validator never populates it.
    """
    import httpx

    from reliquary.shared.prompt_range import window_prompt_range
    from reliquary.constants import PROMPT_RANGE_SIZE

    randomness = ""
    window_n = -1
    cooldown_ids: set[int] = set()
    if validator_url:
        for attempt in range(max(1, retries)):
            try:
                resp = httpx.get(f"{validator_url.rstrip('/')}/state",
                                 params={"env": env_fqn}, timeout=30.0)
                resp.raise_for_status()
                state = resp.json()
                randomness = state.get("randomness", "") or ""
                window_n = int(state.get("window_n", -1))
                cooldown_ids = {int(x) for x in state.get("cooldown_prompts", [])}
                if randomness:
                    break
                logger.info("validator randomness empty (window %d), retry %d/%d",
                            window_n, attempt + 1, retries)
                time.sleep(2.0)
            except Exception as e:  # noqa: BLE001
                logger.warning("failed to fetch /state (%s); retry %d/%d",
                               e, attempt + 1, retries)
                time.sleep(2.0)

    if not randomness:
        if fallback_randomness:
            logger.warning("using --randomness fallback (validator gave none); "
                           "live cooldown unavailable, treating as empty")
            randomness = fallback_randomness
        else:
            raise RuntimeError(
                "no window randomness: validator unreachable/empty and no "
                "--randomness fallback given"
            )

    lo, hi = window_prompt_range(randomness, env_fqn, universe_n, PROMPT_RANGE_SIZE)
    return randomness, window_n, (lo, hi), cooldown_ids


def _select_candidate(
    env_fqn: str, key_id: int, id2group: dict[int, int],
    group_ids: dict[int, list[int]], live_cooldown: set[int],
    lo: int, hi: int, candidate_path: str,
) -> tuple[dict, "int | None"]:
    """Steps 2-4: key id -> group -> candidates (minus live cooldown) ->
    candidate.json -> window-slice intersection -> select ONE other candidate.

    Returns (record, selected_id). selected_id is None when the key id has no
    group or no other candidate landed in the slice (record.status is set).
    Times only the selection work into record["select_s"].
    """
    rec: dict[str, Any] = {"env": env_fqn, "key_id": key_id}
    sel_t0 = time.perf_counter()

    gid = id2group.get(key_id)
    if gid is None:
        rec["status"] = "no_group"
        rec["select_s"] = round(time.perf_counter() - sel_t0, 6)
        return rec, None
    rec["group_id"] = gid

    members = group_ids.get(gid, [])
    candidates = [i for i in members if i not in live_cooldown]
    rec["group_size"] = len(members)
    rec["cooldown_in_group"] = sum(1 for i in members if i in live_cooldown)
    rec["candidates"] = len(candidates)

    with open(candidate_path, "w") as f:
        json.dump({"key_id": key_id, "group_id": gid, "ids": sorted(candidates)}, f)

    # Selection pool = OTHER in-slice, non-cooldown group members (never key id).
    accepted = [i for i in candidates if lo <= i < hi and i != key_id]
    rec["accepted_in_slice"] = len(accepted)
    if not accepted:
        rec["status"] = "no_accepted_in_slice"
        rec["select_s"] = round(time.perf_counter() - sel_t0, 6)
        return rec, None

    selected = min(accepted)
    rec["selected_id"] = selected
    rec["select_s"] = round(time.perf_counter() - sel_t0, 6)
    return rec, selected


async def _run_submit(args, env_fqn: str) -> None:
    """Real-validator submission path (step 6).

    Loads the validator's CURRENT checkpoint into both a generation model and a
    proof model, builds a MiningEngine, and for each in-zone selection produces
    M_ROLLOUTS GRAIL-proven rollouts, signs the envelope, POSTs /submit, and
    reads the accepted/reason verdict. key_id is removed from key_id.json on a
    verdict of accepted (and on OUT_OF_ZONE); kept on submit failure.
    """
    import httpx
    import torch
    import bittensor as bt
    from huggingface_hub import snapshot_download

    from reliquary.constants import (
        ATTN_IMPLEMENTATION, M_ROLLOUTS, PROMPT_RANGE_SIZE,
    )
    from reliquary.environment import load_environment
    import asyncio
    from reliquary.miner.engine import (
        MiningEngine, maybe_pull_checkpoint, _hf_download,
        _compute_merkle_root, _current_drand_round_at_send,
    )
    from reliquary.miner.submitter import (
        SubmissionError, get_window_state_v2, submit_batch_v2,
    )
    from reliquary.protocol.signatures import sign_envelope
    from reliquary.protocol.submission import BatchSubmissionRequest
    from reliquary.shared.modeling import (
        MODEL_SNAPSHOT_ALLOW_PATTERNS, load_text_generation_model, load_tokenizer,
    )
    from reliquary.shared.prompt_range import window_prompt_range

    if not args.validator_url:
        raise SystemExit("--submit requires --validator-url")

    # --- key ids + group index ---
    key_ids = _load_ids(args.key_ids)
    key_container = _key_container(args.key_ids)
    if args.max_key_ids > 0:
        key_ids = key_ids[: args.max_key_ids]
    if not key_ids:
        raise SystemExit(f"{args.key_ids} is empty")
    remaining_key_ids = _load_ids(args.key_ids)

    group_paths = [g for g in args.group if os.path.exists(g)]
    if not group_paths and os.path.exists("group.json"):
        group_paths = ["group.json"]
    if not group_paths:
        raise SystemExit(f"no group file found (looked for {args.group} and group.json)")
    t0 = time.perf_counter()
    id2group, group_ids = _build_group_index(group_paths)
    group_load_s = time.perf_counter() - t0
    logger.info("loaded %d groups (%d ids) from %s in %.2fs",
                len(group_ids), len(id2group), group_paths, group_load_s)

    # --- wallet ---
    wkw = {"name": args.wallet_name, "hotkey": args.hotkey}
    if args.wallet_path:
        wkw["path"] = args.wallet_path
    wallet = bt.Wallet(**wkw)
    logger.info("wallet hotkey: %s", wallet.hotkey.ss58_address)

    # --- live state (randomness + cooldown + checkpoint identity) ---
    async with httpx.AsyncClient(timeout=60) as client:
        state = await get_window_state_v2(args.validator_url, env=env_fqn, client=client)
    if not (state.checkpoint_repo_id and state.checkpoint_revision):
        raise SystemExit("validator has no published checkpoint; cannot build a "
                         "matching GRAIL proof — submit aborted")
    local_hash = state.checkpoint_revision
    local_n = state.checkpoint_n

    # grader for opencode rewards (zone check)
    if args.env == "opencode":
        from reliquary.cli.main import _ensure_grader_running
        _ensure_grader_running()

    # --- load BOTH models from the validator's checkpoint ---
    t0 = time.perf_counter()
    ckpt_path = snapshot_download(repo_id=state.checkpoint_repo_id,
                                  revision=state.checkpoint_revision,
                                  allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS)
    tokenizer = load_tokenizer(ckpt_path)
    proof_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    vllm_model = load_text_generation_model(
        ckpt_path, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION,
    ).to("cuda:0").eval()
    hf_model = load_text_generation_model(
        ckpt_path, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION,
    ).to(proof_device).eval()
    model_load_s = time.perf_counter() - t0
    logger.info("loaded checkpoint %s@%s (gen=cuda:0 proof=%s) in %.2fs",
                state.checkpoint_repo_id, local_hash[:12], proof_device, model_load_s)

    t0 = time.perf_counter()
    env = load_environment(env_fqn)
    env_load_s = time.perf_counter() - t0
    universe_n = len(env)

    engine = MiningEngine(
        vllm_model, hf_model, tokenizer, wallet,
        envs={env_fqn: env}, mix=[(env_fqn, 1)],
        proof_gpu=0 if proof_device == "cuda:0" else 1,
        max_new_tokens=args.max_new_tokens,
        validator_url_override=args.validator_url,
    )

    if id2group and max(id2group) >= universe_n:
        logger.warning("DATASET TOO SMALL: max group id %d >= len(env)=%d — ids "
                       "wrap/never land in-slice.", max(id2group), universe_n)

    zone_high = min(args.zone_high, M_ROLLOUTS)
    records: list[dict] = []
    dead_keys: set[int] = set()  # no_group keys never match — skip after first miss

    async def _process_key(key_id, lo, hi, live_cooldown, window_n, local_hash,
                           randomness, client):
        """Select -> generate -> zone -> (in-zone) submit. Returns the record."""
        rec, selected = _select_candidate(
            env_fqn, key_id, id2group, group_ids, live_cooldown, lo, hi, args.candidate,
        )
        rec["window_n"] = window_n
        if selected is None:
            logger.info("key %d: %s — next", key_id, rec["status"])
            return rec

        problem = env.get_problem(selected)
        gt0 = time.perf_counter()
        generations = engine._generate_m_rollouts(problem, randomness)
        generate_s = time.perf_counter() - gt0
        if len(generations) < M_ROLLOUTS:
            rec["status"] = "gen_short"
            return rec

        rewards = [
            env.compute_reward(problem, tokenizer.decode(g["tokens"][g["prompt_length"]:]))
            for g in generations
        ]
        n_success = sum(1 for r in rewards if r >= args.success_threshold)
        mean = sum(rewards) / len(rewards)
        rec.update({
            "problem_id": problem["id"],
            "generate_s": round(generate_s, 3),
            "rewards": [round(r, 4) for r in rewards],
            "mean_reward": round(mean, 4),
            "n_success": n_success,
            "zone_band": [args.zone_low, zone_high],
        })

        if not (args.zone_low <= n_success <= zone_high):
            rec["status"] = "out_of_zone"
            if key_id in remaining_key_ids:
                remaining_key_ids.remove(key_id)
                _rewrite_key_ids(args.key_ids, remaining_key_ids, key_container)
                rec["removed_from_key_ids"] = True
            logger.info("key %d -> id %d: OUT_OF_ZONE %d/%d — removed, next",
                        key_id, selected, n_success, M_ROLLOUTS)
            return rec

        # in-zone -> build GRAIL submissions + submit
        rollout_subs = [
            engine._build_rollout_submission(g, problem, randomness, env=env)
            for g in generations
        ]
        merkle_root = _compute_merkle_root(rollout_subs)
        current_round = _current_drand_round_at_send()
        nonce = os.urandom(16).hex()
        env_sig = sign_envelope(
            wallet=wallet, miner_hotkey=wallet.hotkey.ss58_address,
            window_start=window_n, prompt_idx=selected, merkle_root=merkle_root,
            checkpoint_hash=local_hash, drand_round=current_round,
            randomness=randomness, nonce=nonce,
        ).hex()
        request = BatchSubmissionRequest(
            miner_hotkey=wallet.hotkey.ss58_address, prompt_idx=selected,
            window_start=window_n, merkle_root=merkle_root, rollouts=rollout_subs,
            checkpoint_hash=local_hash, drand_round=current_round,
            nonce=nonce, envelope_signature=env_sig,
        )
        try:
            resp = await submit_batch_v2(args.validator_url, request, client=client)
            accepted = bool(resp.accepted)
            reason = resp.reason.value if hasattr(resp.reason, "value") else str(resp.reason)
        except SubmissionError as exc:
            accepted, reason = False, f"submit_error:{exc}"

        rec["submitted"] = True
        rec["submit_accepted"] = accepted
        rec["submit_reason"] = reason
        if accepted:
            rec["status"] = "submit_accepted"
            if key_id in remaining_key_ids:
                remaining_key_ids.remove(key_id)
                _rewrite_key_ids(args.key_ids, remaining_key_ids, key_container)
                rec["removed_from_key_ids"] = True
            logger.info("key %d -> id %d: SUBMIT ACCEPTED — removed from %s",
                        key_id, selected, args.key_ids)
        else:
            rec["status"] = "submit_failed"
            logger.info("key %d -> id %d: SUBMIT FAILED (%s) — kept, next",
                        key_id, selected, reason)
        return rec

    def _dump_json(window_n, lo, hi, n_cooldown, randomness):
        """Write the FULL cumulative result.json (json can't be appended)."""
        select_times = [r["select_s"] for r in records if "select_s" in r]
        total_select_s = sum(select_times)
        result = {
            "meta": {
                "checkpoint": f"{state.checkpoint_repo_id}@{local_hash[:12]}",
                "device": "cuda:0", "env": env_fqn, "n_rollouts": M_ROLLOUTS,
                "max_new_tokens": args.max_new_tokens, "window_n": window_n,
                "window_slice": [lo, hi], "live_cooldown_ids": n_cooldown,
                "submit": True, "loop": args.loop,
                "miner_hotkey": wallet.hotkey.ss58_address,
                "randomness": randomness, "key_ids": len(key_ids),
                "windows_processed": len({r.get("window_n") for r in records if "window_n" in r}),
            },
            "timings": {
                "model_load_s": round(model_load_s, 3),
                "env_load_s": round(env_load_s, 3),
                "group_load_s": round(group_load_s, 3),
                "total_select_s": round(total_select_s, 4),
                "mean_select_s": round(total_select_s / len(select_times), 6) if select_times else 0.0,
            },
            "selections": records,
        }
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)

    # markdown: write the run header once, then append a section per window.
    _md_init(args.md, {
        "checkpoint": f"{state.checkpoint_repo_id}@{local_hash[:12]}", "env": env_fqn,
        "miner_hotkey": wallet.hotkey.ss58_address, "n_rollouts": M_ROLLOUTS,
        "max_new_tokens": args.max_new_tokens, "zone_band": [args.zone_low, zone_high],
    }, {
        "model_load_s": round(model_load_s, 3), "env_load_s": round(env_load_s, 3),
        "group_load_s": round(group_load_s, 3),
    })

    # ----------------------------------------------------- window loop
    last_window = -1
    stop = False
    async with httpx.AsyncClient(timeout=120) as client:
        while not stop:
            try:
                st = await get_window_state_v2(args.validator_url, env=env_fqn, client=client)
            except Exception as e:  # noqa: BLE001
                logger.warning("state fetch failed (%s); retrying in %.0fs",
                               e, args.poll_interval)
                await asyncio.sleep(args.poll_interval)
                continue

            # keep proof/gen models in lock-step with the validator checkpoint
            try:
                local_n, local_hash, _ = await maybe_pull_checkpoint(
                    state=st, local_n=local_n, local_hash=local_hash,
                    local_model=engine.hf_model,
                    download_fn=_hf_download, load_fn=engine._load_checkpoint,
                )
            except Exception:
                logger.exception("checkpoint pull failed; keeping current models")

            randomness = st.randomness or args.randomness or ""
            if not randomness or st.window_n == last_window:
                logger.info("waiting for new window (cur=%s, have_randomness=%s, "
                            "remaining_keys=%d)...",
                            st.window_n, bool(randomness), len(remaining_key_ids))
                await asyncio.sleep(args.poll_interval)
                continue

            # --- NEW WINDOW ---
            last_window = st.window_n
            window_n = st.window_n
            live_cooldown = {int(x) for x in st.cooldown_prompts}
            lo, hi = window_prompt_range(randomness, env_fqn, universe_n, PROMPT_RANGE_SIZE)
            keys_this_window = [k for k in remaining_key_ids if k not in dead_keys]
            if args.max_key_ids > 0:
                keys_this_window = keys_this_window[: args.max_key_ids]

            if not keys_this_window:
                logger.info("no live key ids left (remaining are no_group/dead); stopping")
                stop = True
                continue

            logger.info("=== window %d slice [%d,%d) cooldown=%d keys=%d ===",
                        window_n, lo, hi, len(live_cooldown), len(keys_this_window))

            window_records: list[dict] = []
            for key_id in keys_this_window:
                rec = await _process_key(
                    key_id, lo, hi, live_cooldown, window_n, local_hash, randomness, client,
                )
                if rec.get("status") == "no_group":
                    dead_keys.add(key_id)
                records.append(rec)
                window_records.append(rec)

            # append this window's section to the md; rewrite full cumulative json
            _md_append_window(args.md, window_n, lo, hi, len(live_cooldown), window_records)
            _dump_json(window_n, lo, hi, len(live_cooldown), randomness)
            wcounts: dict[str, int] = {}
            for r in window_records:
                wcounts[r["status"]] = wcounts.get(r["status"], 0) + 1
            logger.info("window %d done: %s | appended to %s",
                        window_n, wcounts, args.md)

            if not args.loop:
                stop = True
            elif not remaining_key_ids:
                logger.info("key_id.json empty — all keys resolved; stopping")
                stop = True


def main() -> None:
    from reliquary.constants import ATTN_IMPLEMENTATION, DEFAULT_BASE_MODEL

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["openmath", "opencode"], default="openmath",
                   help="Environment (default: openmath)")
    p.add_argument("--checkpoint", default=DEFAULT_BASE_MODEL,
                   help="Model path or HF repo id (default: %(default)s)")
    p.add_argument("--key-ids", default="key_id.json", help="Key ids to walk")
    p.add_argument("--group", nargs="+",
                   default=["group.part1.json", "group.part2.json"],
                   help="Group membership file(s); group.json is split into two "
                        "parts to stay under GitHub's 100MB limit (default: both parts)")
    p.add_argument("--candidate", default="candidate.json",
                   help="Where to write the current key id's candidate list")
    p.add_argument("--validator-url", default="",
                   help="Validator base URL for the live window randomness "
                        "(e.g. http://127.0.0.1:8888). If omitted, --randomness is required.")
    p.add_argument("--randomness", default="",
                   help="Fallback window randomness hex if the validator has none")
    p.add_argument("--state-retries", type=int, default=5,
                   help="How many times to poll /state for non-empty randomness")
    p.add_argument("--openmath-data", default="", help="Local OpenMath dataset path")
    p.add_argument("--opencode-data", default="", help="Local OpenCode dataset path")
    p.add_argument("--n-rollouts", type=int, default=8,
                   help="Rollouts per selected id (group size; default 8)")
    p.add_argument("--zone-low", type=int, default=2,
                   help="In-zone if #reward-1 rollouts >= this (default 2)")
    p.add_argument("--zone-high", type=int, default=6,
                   help="In-zone if #reward-1 rollouts <= this (default 6)")
    p.add_argument("--success-threshold", type=float, default=1.0,
                   help="A rollout counts as 'reward-1' if reward >= this (default 1.0)")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-key-ids", type=int, default=0,
                   help="If >0, only process the first N key ids")
    p.add_argument("--out", default="group_pipeline_result.json")
    p.add_argument("--md", default="group_pipeline_result.md")
    p.add_argument("--log-level", default="INFO")
    # --- real validator submission (step 6) ---
    p.add_argument("--submit", action="store_true",
                   help="After an in-zone selection, build GRAIL proofs + submit "
                        "to the live validator and read the accepted/reason verdict. "
                        "Removes the key id from key_id.json on a verdict of accepted; "
                        "keeps it on failure. Requires a registered wallet + validator-url.")
    p.add_argument("--wallet-name", default="default", help="Bittensor wallet name")
    p.add_argument("--hotkey", default="default", help="Bittensor hotkey name")
    p.add_argument("--wallet-path", default=os.getenv("BT_WALLET_PATH", ""),
                   help="Optional wallet base path")
    p.add_argument("--loop", action="store_true",
                   help="Keep running: after walking the current window's key ids, "
                        "wait for the next validator window (new randomness/cooldown) "
                        "and walk the remaining key ids again. Stops on Ctrl-C or when "
                        "key_id.json is empty.")
    p.add_argument("--poll-interval", type=float, default=30.0,
                   help="Seconds between /state polls while waiting for a new window")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    env_fqn = _ENV_FQN[args.env]
    if args.openmath_data:
        os.environ["RELIQUARY_OMI_REPO"] = args.openmath_data
    if args.opencode_data:
        os.environ["RELIQUARY_OCI_REPO"] = args.opencode_data

    if args.submit:
        import asyncio
        asyncio.run(_run_submit(args, env_fqn))
        return

    if args.loop:
        logger.warning("--loop only applies in --submit mode; ignoring for the "
                       "offline (non-submit) run.")

    # --- key ids ---
    key_ids = _load_ids(args.key_ids)
    key_container = _key_container(args.key_ids)
    if args.max_key_ids > 0:
        key_ids = key_ids[: args.max_key_ids]
    if not key_ids:
        raise SystemExit(
            f"{args.key_ids} is empty — put the key ids to try in it, "
            'e.g. {"ids":[100555, 236999, ...]}'
        )
    logger.info("key ids to walk: %d", len(key_ids))
    # Mirror the on-disk file; OUT_OF_ZONE key ids are pruned from it as we go.
    remaining_key_ids = _load_ids(args.key_ids)

    # --- group index ---
    # Use whichever group file(s) actually exist: the split parts by default,
    # or a single monolithic group.json if that's what's on disk.
    group_paths = [g for g in args.group if os.path.exists(g)]
    if not group_paths and os.path.exists("group.json"):
        group_paths = ["group.json"]
    if not group_paths:
        raise SystemExit(
            f"no group file found (looked for {args.group} and group.json)"
        )
    t0 = time.perf_counter()
    id2group, group_ids = _build_group_index(group_paths)
    group_load_s = time.perf_counter() - t0
    logger.info("loaded %d groups (%d ids indexed) from %s in %.2fs",
                len(group_ids), len(id2group), group_paths, group_load_s)

    # --- grader (opencode only) ---
    if args.env == "opencode":
        from reliquary.cli.main import _ensure_grader_running
        _ensure_grader_running()

    # --- model ---
    import torch
    from reliquary.shared.modeling import load_text_generation_model, load_tokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    tokenizer = load_tokenizer(args.checkpoint)
    mk = {"torch_dtype": torch.bfloat16}
    if device.startswith("cuda"):
        mk["attn_implementation"] = ATTN_IMPLEMENTATION
    model = load_text_generation_model(args.checkpoint, **mk).to(device).eval()
    model_load_s = time.perf_counter() - t0
    logger.info("model loaded on %s in %.2fs", device, model_load_s)

    # For the offline validator-side termination verdict.
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP, MAX_TRUNCATED_PER_SUBMISSION
    from reliquary.shared.modeling import resolve_eos_token_ids
    eos_ids = resolve_eos_token_ids(model, tokenizer)
    term_cap = MAX_NEW_TOKENS_PROTOCOL_CAP
    max_trunc = MAX_TRUNCATED_PER_SUBMISSION

    # --- env ---
    from reliquary.environment import load_environment
    t0 = time.perf_counter()
    env = load_environment(env_fqn)
    env_load_s = time.perf_counter() - t0
    universe_n = len(env)
    logger.info("env %s loaded (%d problems) in %.2fs", env_fqn, universe_n, env_load_s)

    # Correctness guard: the window slice [lo,hi) and get_problem(idx) both live
    # in [0, universe_n). If group ids exceed it, get_problem wraps (idx % len)
    # to the WRONG problem and those ids can never land in-slice. This means the
    # loaded dataset is smaller than the one group.json was built from (e.g. too
    # few OpenMath shards). It must also match the validator's len(env) for the
    # slice to agree.
    if id2group:
        max_gid = max(id2group)
        if max_gid >= universe_n:
            n_over = sum(1 for i in id2group if i >= universe_n)
            logger.warning(
                "DATASET TOO SMALL: %d group ids (max %d) exceed len(env)=%d. "
                "These wrap to wrong problems and never land in-slice. Load more "
                "shards (RELIQUARY_OMI_SHARDS) or point --openmath-data at the "
                "full dataset, and match the validator's len(env).",
                n_over, max_gid, universe_n,
            )

    if args.zone_low > min(args.zone_high, args.n_rollouts):
        logger.warning(
            "zone_low=%d > effective zone_high=%d (n_rollouts=%d): nothing can "
            "ever be in-zone.", args.zone_low, min(args.zone_high, args.n_rollouts),
            args.n_rollouts,
        )

    # --- live window state: slice + REAL cooldown right now ---
    randomness, window_n, (lo, hi), live_cooldown = _fetch_window_state(
        args.validator_url, env_fqn, universe_n, args.state_retries, args.randomness,
    )
    logger.info("window %d slice [%d, %d) size=%d cooldown_ids=%d (randomness=%s...)",
                window_n, lo, hi, hi - lo, len(live_cooldown), randomness[:12])

    # ------------------------------------------------------------------ walk
    records: list[dict] = []
    for key_id in key_ids:
        rec, selected = _select_candidate(
            env_fqn, key_id, id2group, group_ids, live_cooldown, lo, hi, args.candidate,
        )
        if selected is None:
            records.append(rec)
            logger.info("key %d: %s — next", key_id, rec["status"])
            continue
        logger.info("key %d -> selected %d (group %d) in %.4fs (selection)",
                    key_id, selected, rec["group_id"], rec["select_s"])

        # step 5: generate + score the group
        problem = env.get_problem(selected)
        authoritative = bool(getattr(env, "validator_authoritative_reward", False))
        rollouts, generate_s = _generate_n_rollouts(
            model, tokenizer, problem["prompt"], args.n_rollouts, args.max_new_tokens,
        )
        rewards = []
        roll_recs = []
        bad_term = 0
        for i, gen in enumerate(rollouts):
            completion = tokenizer.decode(gen["tokens"][gen["prompt_length"]:])
            r = env.compute_reward(problem, completion)
            rewards.append(r)
            term = _termination_status(gen["tokens"], gen["prompt_length"], eos_ids, term_cap)
            if term == "bad_termination":
                bad_term += 1
            roll_recs.append({"index": i, "reward": r,
                              "completion_tokens": len(gen["tokens"]) - gen["prompt_length"],
                              "termination": term})
        # Validator-side termination verdict: the validator rejects the whole
        # batch (BAD_TERMINATION) when more than MAX_TRUNCATED_PER_SUBMISSION
        # rollouts fail to terminate naturally / at the cap.
        would_reject_term = bad_term > max_trunc
        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = var ** 0.5
        # Zone = count of reward-1 (success) rollouts within [zone_low, zone_high].
        # e.g. for 8 rollouts, 2..6 correct is in-zone; 0/1 (too hard) or 7/8
        # (too easy) is OUT_OF_ZONE. zone_high is clamped to the group size.
        n_success = sum(1 for r in rewards if r >= args.success_threshold)
        zone_high = min(args.zone_high, args.n_rollouts)

        rec.update({
            "problem_id": problem["id"],
            "validator_authoritative_reward": authoritative,
            "generate_s": round(generate_s, 3),
            "per_rollout_gen_s": round(generate_s / args.n_rollouts, 3),
            "rewards": [round(r, 4) for r in rewards],
            "mean_reward": round(mean, 4),
            "reward_std": round(std, 4),
            "n_success": n_success,
            "zone_band": [args.zone_low, zone_high],
            "bad_termination": bad_term,
            "term_verdict": "BAD_TERMINATION" if would_reject_term else "ok",
            "rollouts": roll_recs,
        })
        if would_reject_term:
            logger.warning(
                "key %d -> id %d: %d/%d rollouts bad-terminated (> %d allowed) — "
                "validator would reject this batch as BAD_TERMINATION; raise "
                "--max-new-tokens", key_id, selected, bad_term, args.n_rollouts, max_trunc,
            )

        if args.zone_low <= n_success <= zone_high:
            rec["status"] = "accepted"
            logger.info("key %d -> id %d: ACCEPTED %d/%d reward-1 (zone %d..%d) "
                        "mean=%.3f gen=%.2fs",
                        key_id, selected, n_success, args.n_rollouts,
                        args.zone_low, zone_high, mean, generate_s)
        else:
            rec["status"] = "out_of_zone"
            # Prune this key id from key_id.json and persist immediately, then
            # move on to the next key id (step 2).
            if key_id in remaining_key_ids:
                remaining_key_ids.remove(key_id)
                _rewrite_key_ids(args.key_ids, remaining_key_ids, key_container)
                rec["removed_from_key_ids"] = True
            logger.info("key %d -> id %d: OUT_OF_ZONE %d/%d reward-1 (need %d..%d) "
                        "— removed from %s, next", key_id, selected, n_success,
                        args.n_rollouts, args.zone_low, zone_high, args.key_ids)
        records.append(rec)

    # ------------------------------------------------------------------ output
    select_times = [r["select_s"] for r in records if "select_s" in r]
    total_select_s = sum(select_times)
    result = {
        "meta": {
            "checkpoint": args.checkpoint, "device": device, "env": env_fqn,
            "n_rollouts": args.n_rollouts, "max_new_tokens": args.max_new_tokens,
            "window_n": window_n, "window_slice": [lo, hi],
            "live_cooldown_ids": len(live_cooldown), "submit": False,
            "randomness": randomness, "key_ids": len(key_ids),
        },
        "timings": {
            "model_load_s": round(model_load_s, 3),
            "env_load_s": round(env_load_s, 3),
            "group_load_s": round(group_load_s, 3),
            "total_select_s": round(total_select_s, 4),
            "mean_select_s": round(total_select_s / len(select_times), 6) if select_times else 0.0,
        },
        "selections": records,
    }
    _emit_outputs(args, result)


def _emit_outputs(args, result: dict) -> None:
    """Write the result json + markdown and print the stdout summary."""
    records = result["selections"]
    meta = result["meta"]
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    _write_markdown(result, args.md)

    by_status: dict[str, int] = {}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print("\n" + "=" * 64)
    print(f"env={meta['env']}  window={meta['window_n']}  "
          f"slice={meta['window_slice']}  device={meta['device']}"
          + ("  [SUBMIT]" if meta.get("submit") else ""))
    print(f"key ids walked: {len(records)}")
    print(f"prompt-selection time: total={result['timings'].get('total_select_s')}s  "
          f"mean={result['timings'].get('mean_select_s')}s/key")
    for st, n in by_status.items():
        print(f"  {st:<22} {n}")
    accepted_recs = [r for r in records if r["status"] in ("accepted", "submit_accepted")]
    if accepted_recs:
        m = sum(r.get("mean_reward", 0.0) for r in accepted_recs) / len(accepted_recs)
        print(f"in-zone selections: {len(accepted_recs)}  mean reward={m:.4f}")
    if meta.get("submit"):
        subok = sum(1 for r in records if r["status"] == "submit_accepted")
        subfail = sum(1 for r in records if r["status"] == "submit_failed")
        print(f"submitted: accepted={subok}  failed={subfail}")
    print(f"\nwrote {args.out} and {args.md}")
    print("=" * 64)


def _md_init(path: str, meta: dict, timings: dict) -> None:
    """Create/truncate the markdown report with a run header. Called once at the
    start of a --loop run; window sections are appended after each window."""
    L = ["# Group pipeline results (live submit)\n", "## Run\n",
         "| field | value |", "|---|---|",
         f"| checkpoint | `{meta['checkpoint']}` |",
         f"| env | {meta['env']} |",
         f"| miner_hotkey | `{meta.get('miner_hotkey','-')}` |",
         f"| rollouts / selection | {meta['n_rollouts']} |",
         f"| max_new_tokens | {meta['max_new_tokens']} |",
         f"| zone band | {meta.get('zone_band','-')} |",
         "", "## Setup timings (seconds)\n",
         "| stage | seconds |", "|---|--:|",
         f"| model load / download | {timings['model_load_s']} |",
         f"| dataset load | {timings['env_load_s']} |",
         f"| group index load | {timings['group_load_s']} |",
         "", "---", ""]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def _md_append_window(path: str, window_n: int, lo: int, hi: int,
                      n_cooldown: int, window_records: list[dict]) -> None:
    """Append one window's section (heading + per-key table + summary) to the
    markdown file, preserving everything already written for earlier windows."""
    L = [f"## Window {window_n} · slice [{lo}, {hi}) · live cooldown {n_cooldown}\n",
         "| key_id | group | cands | in_slice | selected | status | reward-1 | "
         "mean | select_s | gen_s | submit_reason |",
         "|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|---|"]
    for r in window_records:
        nsucc = r.get("n_success")
        band = r.get("zone_band")
        succ = f"{nsucc} ({band[0]}..{band[1]})" if nsucc is not None and band else "-"
        L.append(
            f"| {r['key_id']} | {r.get('group_id','-')} | {r.get('candidates','-')} | "
            f"{r.get('accepted_in_slice','-')} | {r.get('selected_id','-')} | "
            f"{r['status']} | {succ} | {r.get('mean_reward','-')} | "
            f"{r.get('select_s','-')} | {r.get('generate_s','-')} | "
            f"{r.get('submit_reason','-')} |"
        )
    counts: dict[str, int] = {}
    for r in window_records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = " · ".join(f"{k}={v}" for k, v in counts.items()) or "(no keys)"
    L += ["", f"**Window {window_n} outcome:** {summary}", "", "---", ""]
    with open(path, "a") as f:
        f.write("\n".join(L) + "\n")


def _write_markdown(result: dict, path: str) -> None:
    meta = result["meta"]
    t = result["timings"]
    recs = result["selections"]
    L: list[str] = ["# Group pipeline results\n"]

    L.append("## Run\n")
    L.append("| field | value |\n|---|---|")
    L.append(f"| checkpoint | `{meta['checkpoint']}` |")
    L.append(f"| env | {meta['env']} |")
    L.append(f"| device | {meta['device']} |")
    L.append(f"| window_n | {meta['window_n']} |")
    L.append(f"| window slice | [{meta['window_slice'][0]}, {meta['window_slice'][1]}) |")
    L.append(f"| rollouts / selection | {meta['n_rollouts']} |")
    L.append(f"| key ids walked | {len(recs)} |")
    L.append("")

    L.append("## Timings (seconds)\n")
    L.append("| stage | seconds |\n|---|--:|")
    L.append(f"| model load / download | {t['model_load_s']} |")
    L.append(f"| dataset load | {t['env_load_s']} |")
    L.append(f"| group index load | {t['group_load_s']} |")
    L.append(f"| prompt-selection total | {t.get('total_select_s', '-')} |")
    L.append(f"| prompt-selection mean / key | {t.get('mean_select_s', '-')} |")
    L.append("")

    by_status: dict[str, int] = {}
    for r in recs:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    L.append("## Outcome by status\n")
    L.append("| status | count |\n|---|--:|")
    for st, n in by_status.items():
        L.append(f"| {st} | {n} |")
    L.append("")

    submitted_any = any(r.get("submitted") for r in recs)
    L.append("## Per key-id walk\n")
    header = ("| key_id | group | grp_size | cooldown | cands | in_slice | "
              "selected | status | reward-1 | mean | term | select_s | gen_s |")
    sep = "|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|---|--:|--:|"
    if submitted_any:
        header = header + " submit_reason |"
        sep = sep + "---|"
    L.append(header)
    L.append(sep)
    for r in recs:
        nsucc = r.get("n_success")
        band = r.get("zone_band")
        succ_cell = f"{nsucc} ({band[0]}..{band[1]})" if nsucc is not None and band else "-"
        bt = r.get("bad_termination")
        term_cell = (f"{r.get('term_verdict','-')} ({bt})" if bt is not None
                     else r.get("term_verdict", "-"))
        row = (
            f"| {r['key_id']} | {r.get('group_id','-')} | {r.get('group_size','-')} | "
            f"{r.get('cooldown_in_group','-')} | {r.get('candidates','-')} | "
            f"{r.get('accepted_in_slice','-')} | {r.get('selected_id','-')} | "
            f"{r['status']} | {succ_cell} | {r.get('mean_reward','-')} | "
            f"{term_cell} | {r.get('select_s','-')} | {r.get('generate_s','-')} |"
        )
        if submitted_any:
            row = row + f" {r.get('submit_reason','-')} |"
        L.append(row)
    L.append("")

    acc = [r for r in recs if r["status"] == "accepted"]
    if acc:
        L.append("## Accepted selections — per-rollout reward\n")
        L.append("| key_id | selected_id | rollout | reward | tokens |\n|--:|--:|--:|--:|--:|")
        for r in acc:
            for ro in r["rollouts"]:
                L.append(f"| {r['key_id']} | {r['selected_id']} | {ro['index']} | "
                         f"{ro['reward']:.3f} | {ro['completion_tokens']} |")
        L.append("")

    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
