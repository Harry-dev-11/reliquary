"""Miner engine — vLLM generation + HuggingFace GRAIL proof construction.

Protocol v2: free prompt selection (uniform random with cooldown skip),
M rollouts per prompt at fixed temperature T_PROTO, local reward computation,
Merkle root commitment, HTTP batch submission to validator.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

import random as _random

from reliquary.constants import (
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
    PROMPT_RANGE_SIZE,
    T_PROTO,
    TOP_K_PROTO,
    TOP_P_PROTO,
    UPLOAD_BUFFER,
    WINDOW_LENGTH,
)
from reliquary.shared.prompt_range import window_prompt_range
from reliquary.miner.timing import get_recorder
from reliquary.infrastructure import chain
from reliquary.protocol.signatures import sign_envelope
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
)

if TYPE_CHECKING:
    from reliquary.environment.base import Environment

logger = logging.getLogger(__name__)

# Layer-4 frontier probe: number of cheap rollouts generated to test whether a
# scored prompt sits on the learning frontier before committing to a full
# group + GRAIL. A uniform reward vector (all-0 / all-1) over these predicts
# OUT_OF_ZONE, so the prompt is skipped.
PROBE_ROLLOUTS = 4

# Layer-2 scored candidate pool: how many top-ranked prompts to keep per window
# for the probe to filter. Larger than the 8/window submission cap because the
# Layer-4 probe rejects uniform prompts, so the pool must over-provision to
# still yield up to 8 frontier submissions.
SCORED_CANDIDATE_POOL = 16


async def maybe_pull_checkpoint(
    state,
    local_n: int,
    local_hash: str,
    local_model,
    *,
    download_fn,
    load_fn,
):
    """If remote checkpoint_n > local, download via HF and load.

    state.checkpoint_repo_id + state.checkpoint_revision identify the
    HF snapshot. download_fn/load_fn still injected for testability.

    Returns ``(new_local_n, new_local_hash, new_model)``. If no update is
    needed (remote ≤ local, or remote has no repo/revision yet), returns
    inputs unchanged.
    """
    if state.checkpoint_n <= local_n:
        return local_n, local_hash, local_model
    if state.checkpoint_repo_id is None or state.checkpoint_revision is None:
        return local_n, local_hash, local_model
    local_path = await download_fn(state.checkpoint_repo_id, state.checkpoint_revision)
    new_model = load_fn(local_path)
    return state.checkpoint_n, state.checkpoint_revision, new_model


async def _hf_download(repo_id: str, revision: str) -> str:
    """Download a snapshot into the local HF cache and return the model folder path."""
    import asyncio
    from huggingface_hub import snapshot_download
    from reliquary.shared.modeling import MODEL_SNAPSHOT_ALLOW_PATTERNS

    t0 = time.perf_counter()
    path = await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
    )
    get_recorder().record_one_time(
        f"model download ({repo_id}@{revision[:12]})", time.perf_counter() - t0,
    )
    return path


def pick_prompt_idx(
    env,
    cooldown_prompts: set[int],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
    prompt_range: tuple[int, int] | None = None,
) -> int:
    """Pick a random prompt index that isn't currently in cooldown.

    When ``prompt_range`` is given, sampling is confined to ``[lo, hi)`` —
    the per-window slice the validator enforces. The reference miner uses
    uniform-random selection with rejection sampling against the cooldown
    set; more sophisticated strategies are left to miner operators.

    Raises ``RuntimeError`` if no eligible prompt can be found.
    """
    rng = rng or _random
    n = len(env)
    lo, hi = (0, n) if prompt_range is None else prompt_range
    lo = max(0, lo)
    hi = min(n, hi)
    span = hi - lo
    if span <= 0:
        raise RuntimeError("no eligible prompt — empty range")
    cd_in_span = sum(1 for c in cooldown_prompts if lo <= c < hi)
    if cd_in_span < span / 2:
        for _ in range(max_attempts):
            idx = lo + rng.randrange(span)
            if idx not in cooldown_prompts:
                return idx
        raise RuntimeError("no eligible prompt found after max attempts")
    eligible = [i for i in range(lo, hi) if i not in cooldown_prompts]
    if not eligible:
        raise RuntimeError("no eligible prompt — range fully in cooldown")
    return rng.choice(eligible)


def pick_env_and_prompt(
    envs: dict,
    mix: list[tuple[str, int]],
    cooldown_per_env: dict[str, set[int]],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 1000,
    randomness: str | None = None,
) -> tuple[str, int]:
    """Sample env per `mix` weights, then a prompt within that env.

    When ``randomness`` is given, each env's prompt is drawn only from that
    window's slice (``window_prompt_range``), matching the validator. Falls
    through to the next env (re-sampling with the chosen env masked) if the
    chosen env's slice is fully in cooldown.
    """
    rng = rng or _random
    names = [n for n, _ in mix]
    weights = [w for _, w in mix]
    if not names:
        raise RuntimeError("pick_env_and_prompt: empty mix")

    available = list(names)
    while available:
        avail_weights = [weights[names.index(n)] for n in available]
        env_name = rng.choices(available, weights=avail_weights)[0]
        env = envs[env_name]
        prompt_range = None
        if randomness:
            env_label = getattr(env, "name", env_name)
            prompt_range = window_prompt_range(
                randomness, env_label, len(env), PROMPT_RANGE_SIZE,
            )
        try:
            idx = pick_prompt_idx(
                env, cooldown_per_env.get(env_name, set()),
                rng=rng, max_attempts=max_attempts, prompt_range=prompt_range,
            )
            return env_name, idx
        except RuntimeError:
            available.remove(env_name)

    raise RuntimeError("pick_env_and_prompt: all envs fully in cooldown")


def _compute_merkle_root(rollouts) -> str:
    """Compute Merkle root over rollout leaves — returns 64-char hex.

    Uses canonical JSON (sort_keys=True, compact separators) for dict/list
    serialisation so the root is deterministic across Python
    implementations and refactor-stable against dict-construction-order
    changes.
    """
    import hashlib
    import json

    leaves = []
    for i, r in enumerate(rollouts):
        h = hashlib.sha256()
        h.update(i.to_bytes(8, "big"))
        h.update(json.dumps(r.tokens, separators=(",", ":")).encode())
        h.update(json.dumps(r.reward).encode())
        h.update(json.dumps(r.commit, sort_keys=True, separators=(",", ":")).encode())
        leaves.append(h.digest())

    while len(leaves) > 1:
        new = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            new.append(hashlib.sha256(left + right).digest())
        leaves = new
    return leaves[0].hex()


def _rollout_token_hash(tokens) -> bytes:
    """SHA256 over *tokens* as 4-byte big-endian unsigned ints.

    MUST stay byte-identical to
    ``reliquary.validator.dedup.compute_rollout_hash`` so the miner's local
    distinct-rollout guard predicts the validator's HASH_DUPLICATE check
    exactly. The validator hashes ``rollout.commit["tokens"]`` — the full
    prompt+completion sequence — which equals the generation dict's
    ``"tokens"`` here.
    """
    h = hashlib.sha256()
    for t in tokens:
        h.update(int(t).to_bytes(4, "big", signed=False))
    return h.digest()


def _current_drand_round_at_send() -> int:
    """Drand quicknet round currently in progress at wall-clock now.

    Called just before POSTing /submit so the attached round matches what
    the validator sees at receipt (modulo the 1-round tolerance). Uses
    chain params cached at process start; one drand period of skew is
    tolerated by the validator.
    """
    from reliquary.infrastructure.chain import compute_current_drand_round
    from reliquary.infrastructure.drand import get_current_chain

    ci = get_current_chain()
    return compute_current_drand_round(time.time(), ci["genesis_time"], ci["period"])


class MiningEngine:
    """Two-GPU mining: vLLM (GPU 0) for generation, HF (GPU 1) for proofs."""

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        wallet,
        env: "Environment | None" = None,
        *,
        envs: "dict[str, Environment] | None" = None,
        mix: "list[tuple[str, int]] | None" = None,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
        validator_url_override: str | None = None,
        selection_mode: str = "score",
        candidate_path: str | None = None,
    ) -> None:
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.wallet = wallet
        self.vllm_gpu = vllm_gpu
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self.validator_url_override = validator_url_override

        if envs is not None and mix is not None:
            self.envs = envs
            self.mix = mix
            self.env = next(iter(envs.values()))  # legacy fallback
        else:
            assert env is not None, "must pass either env or envs+mix"
            self.envs = {env.name: env}
            self.mix = [(env.name, 1)]
            self.env = env
        self._cooldown_per_env: dict[str, set[int]] = {n: set() for n in self.envs}

        # Score-based prompt selection (OpenMath-only). The env-mix layer is
        # removed: the miner ranks the per-window slice by feature-typicality
        # against the live cooldown population and submits the top picks.
        self._score_env = self.envs.get("openmathinstruct", self.env)
        self._scored_window_key: tuple | None = None
        self._scored_queue: list[int] = []
        self._rng = _random.Random()

        # Per-process memory of rollout-token hashes we've already submitted,
        # so a restart-in-process / re-pick never re-sends content the
        # validator would reject with HASH_DUPLICATE. Bounded LRU.
        self._recent_rollout_hashes: "OrderedDict[bytes, None]" = OrderedDict()
        self._recent_rollout_cap = 4096

        # --- "smart" pipeline state (selection_mode == "smart") ---------------
        # Differs from the score pipeline only in prompt selection: pick from a
        # precomputed candidate.json (per-env id lists) intersected with the
        # per-window slice; submit OpenMath, fall back to OpenCode on
        # batch_filled, wait for the next window if both fill. No frontier probe.
        self.selection_mode = selection_mode
        self._smart_order = ["openmathinstruct", "opencodeinstruct"]
        self._smart_candidates: dict[str, set[int]] = {}
        self._smart_prev_cooldown: dict[str, set[int]] = {}
        self._smart_window_n: int | None = None
        self._smart_current_env = self._smart_order[0]
        self._smart_filled: set[str] = set()
        if selection_mode == "smart":
            self._load_smart_candidates(candidate_path)

        # Lazy imports for heavy deps — keep module import cheap.
        from reliquary.shared.hf_compat import resolve_hidden_size
        from reliquary.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def mine_window(
        self,
        subtensor,
        window_start: int = 0,  # v2.0 param kept for CLI compat; ignored
        use_drand: bool = True,
    ) -> list:
        """v2.1: poll state, pull checkpoint on n-change, submit when OPEN.

        Returns the list of BatchSubmissionResponse objects collected
        across the loop. The loop exits only on external cancellation
        (asyncio.CancelledError) or if env becomes fully cooldown'd.
        """
        import httpx

        from reliquary.constants import M_ROLLOUTS, POLL_INTERVAL_SECONDS
        from reliquary.miner.submitter import (
            SubmissionError, discover_validator_url,
            get_window_state_v2, submit_batch_v2,
        )
        from reliquary.protocol.submission import (
            BatchSubmissionRequest, WindowState,
        )

        # Resolve validator URL (once).
        if self.validator_url_override:
            url = self.validator_url_override
        else:
            metagraph = await chain.get_metagraph(subtensor, chain.NETUID)
            url = discover_validator_url(metagraph)

        # v2.3: randomness is fetched per-window from /state instead of
        # recomputed locally. The validator aligns window OPEN to a drand
        # boundary and binds randomness to the round publishing at that
        # boundary — a value that didn't exist a few seconds earlier, so
        # nothing to pre-fetch. The miner just reads what /state reports.
        results = []
        local_n = 0
        local_hash = ""

        async with httpx.AsyncClient(timeout=30) as client:
            while True:
                try:
                    state = await get_window_state_v2(url, client=client)
                except SubmissionError:
                    # /state may return 503 between windows; wait briefly.
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue
                except Exception as e:
                    logger.debug("state fetch failed: %s", e)
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # Pull new checkpoint if needed (works at any state).
                try:
                    local_n, local_hash, self.hf_model = await maybe_pull_checkpoint(
                        state=state, local_n=local_n, local_hash=local_hash,
                        local_model=self.hf_model,
                        download_fn=_hf_download,
                        load_fn=self._load_checkpoint,
                    )
                except Exception:
                    logger.exception("checkpoint pull failed; keeping local")

                if state.state != WindowState.OPEN:
                    await asyncio.sleep(1)
                    continue

                # v2.3: trust the validator's per-window randomness rather
                # than recomputing locally. Empty string means the validator
                # hasn't yet finished _set_window_randomness — wait briefly.
                randomness = state.randomness
                if not randomness:
                    await asyncio.sleep(0.1)
                    continue

                recorder = get_recorder()
                if self.selection_mode == "smart":
                    # SMART pipeline: candidate.json ∩ per-window slice, submit
                    # OpenMath then fall back to OpenCode on batch_filled. No
                    # frontier probe — submit directly.
                    pick = await self._smart_pick(randomness, state, url, client)
                    if pick is None:
                        logger.info(
                            "smart: no eligible env/candidate for window %d; "
                            "waiting", state.window_n,
                        )
                        await asyncio.sleep(5)
                        continue
                    env, env_name, prompt_idx, problem = pick
                    _t_gen = time.perf_counter()
                    generations = self._generate_m_rollouts(problem, randomness)
                    recorder.record_generation(
                        state.window_n, time.perf_counter() - _t_gen, M_ROLLOUTS,
                    )
                    if len(generations) < M_ROLLOUTS:
                        logger.warning(
                            "generated %d/%d for prompt %d; skipping",
                            len(generations), M_ROLLOUTS, prompt_idx,
                        )
                        continue
                else:
                    # SCORE pipeline (OpenMath-only): fetch this env's own
                    # cooldown set (``prompt_idx`` is per-env), falling back to
                    # the flat base set on a fetch error rather than stall.
                    env = self._score_env
                    env_name = getattr(env, "name", "openmathinstruct")
                    try:
                        env_state = await get_window_state_v2(
                            url, env=env_name, client=client,
                        )
                        cooldown_prompts = set(env_state.cooldown_prompts)
                    except Exception:
                        cooldown_prompts = set(state.cooldown_prompts)
                    self._cooldown_per_env[env_name] = cooldown_prompts

                    # Layers 1-3 (slice → score → cooldown) + Layer 4 (frontier
                    # probe): pick a scored candidate whose cheap 4-rollout probe
                    # has a MIXED reward signal, so it won't fail OUT_OF_ZONE.
                    _t_gen = time.perf_counter()
                    probe_result = await self._probe_for_valid_prompt(
                        env, randomness, cooldown_prompts, state.window_n,
                    )
                    if probe_result is None:
                        logger.info(
                            "no frontier prompt in window %d "
                            "(candidates uniform/exhausted); waiting",
                            state.window_n,
                        )
                        await asyncio.sleep(5)
                        continue
                    prompt_idx, problem, probe_rollouts = probe_result

                    # Top up the probe sample to a full M_ROLLOUTS group, reusing
                    # the probe rollouts (all are identical T_PROTO samples).
                    need = M_ROLLOUTS - len(probe_rollouts)
                    extra = (
                        self._generate_m_rollouts(problem, randomness, n=need)
                        if need > 0 else []
                    )
                    generations = (probe_rollouts + extra)[:M_ROLLOUTS]
                    recorder.record_generation(
                        state.window_n, time.perf_counter() - _t_gen, M_ROLLOUTS,
                    )
                    if len(generations) < M_ROLLOUTS:
                        logger.warning(
                            "generated %d/%d for prompt %d; skipping",
                            len(generations), M_ROLLOUTS, prompt_idx,
                        )
                        continue

                # Window-change guard #1: generation took tens of seconds. If
                # the window advanced, the captured window_n/randomness are now
                # stale — discard before spending GRAIL rather than POST a
                # submission the validator will reject.
                if await self._window_changed(url, client, state):
                    logger.info(
                        "window advanced during generation (was %d); "
                        "discarding %d rollouts for prompt %d",
                        state.window_n, len(generations), prompt_idx,
                    )
                    continue

                # Distinct-rollout guard: the validator rejects the WHOLE group
                # with HASH_DUPLICATE if any two rollouts share token content,
                # or if any rollout matches one already accepted in its
                # retention window. Predict both here and skip before spending
                # GRAIL. A group with duplicate completions is also low-σ
                # (likely OUT_OF_ZONE), so skipping it is doubly correct.
                gen_hashes = [_rollout_token_hash(g["tokens"]) for g in generations]
                if len(set(gen_hashes)) < len(gen_hashes):
                    logger.info(
                        "prompt %d: rollouts not all distinct "
                        "(%d unique of %d); skipping (would be hash_duplicate)",
                        prompt_idx, len(set(gen_hashes)), len(gen_hashes),
                    )
                    continue
                if any(h in self._recent_rollout_hashes for h in gen_hashes):
                    logger.info(
                        "prompt %d: produced already-submitted rollouts; "
                        "skipping (would be hash_duplicate)", prompt_idx,
                    )
                    continue

                # Build each rollout's GRAIL proof one by one, timing each.
                rollout_submissions = []
                for _i, gen in enumerate(generations, start=1):
                    _t_grail = time.perf_counter()
                    sub = self._build_rollout_submission(
                        gen, problem, randomness, env=env,
                    )
                    recorder.record_grail(
                        state.window_n, _i, len(generations),
                        time.perf_counter() - _t_grail,
                    )
                    rollout_submissions.append(sub)
                merkle_root = _compute_merkle_root(rollout_submissions)

                # Window-change guard #2: last-line defense for a window change
                # during the GRAIL pass. Cheaper to re-poll than to POST a
                # stale submission and burn the validator's WINDOW_MISMATCH path.
                if await self._window_changed(url, client, state):
                    logger.info(
                        "window advanced during GRAIL build (was %d); "
                        "dropping submission for prompt %d",
                        state.window_n, prompt_idx,
                    )
                    continue

                # v2.3 design A': fetch the drand round just before the POST.
                # The attached round determines the submission's chronological
                # slot at seal time. Miss this and the validator rejects with
                # STALE_ROUND or FUTURE_ROUND.
                current_round = _current_drand_round_at_send()
                # Envelope signature (introduced 2026-05 to close the
                # /submit hotkey-spoof DoS). Sign the canonical binding
                # over every field the validator routes on, including
                # the validator-published window randomness — that ties
                # the signature to this exact validator's view of the
                # window so a captured signature cannot be replayed
                # against a different validator instance or a future
                # window. The nonce is per-submission fresh randomness.
                import os as _os
                _nonce = _os.urandom(16).hex()
                _envelope_sig = sign_envelope(
                    wallet=self.wallet,
                    miner_hotkey=self.wallet.hotkey.ss58_address,
                    window_start=state.window_n,
                    prompt_idx=prompt_idx,
                    merkle_root=merkle_root,
                    checkpoint_hash=local_hash,
                    drand_round=current_round,
                    randomness=state.randomness or "",
                    nonce=_nonce,
                ).hex()
                request = BatchSubmissionRequest(
                    miner_hotkey=self.wallet.hotkey.ss58_address,
                    prompt_idx=prompt_idx,
                    window_start=state.window_n,
                    merkle_root=merkle_root,
                    rollouts=rollout_submissions,
                    checkpoint_hash=local_hash,
                    drand_round=current_round,
                    nonce=_nonce,
                    envelope_signature=_envelope_sig,
                )
                try:
                    resp = await submit_batch_v2(url, request, client=client)
                    logger.info(
                        "submitted window=%d prompt=%d accepted=%s reason=%s",
                        state.window_n, prompt_idx, resp.accepted,
                        resp.reason.value if hasattr(resp.reason, "value") else resp.reason,
                    )
                    results.append(resp)
                    recorder.mark_window_submitted()
                    # Remember what we sent so we never re-POST identical
                    # content in this process (→ HASH_DUPLICATE).
                    for _h in gen_hashes:
                        self._remember_rollout_hash(_h)
                    # SMART: on batch_filled, switch OpenMath→OpenCode (or wait).
                    if self.selection_mode == "smart":
                        self._smart_after_submit(env_name, resp)
                except SubmissionError as exc:
                    logger.error("submit failed: %s", exc)

                # Refresh the timing report after every attempt so the
                # operator always has a current result.md while mining.
                recorder.write_result_md()

        return results

    def _load_smart_candidates(self, candidate_path: str | None) -> None:
        """Load per-env candidate id lists from candidate.json for smart mode."""
        import json
        import os

        tried = [
            candidate_path,
            os.environ.get("RELIQUARY_CANDIDATE_JSON"),
            "candidate.json",
            os.path.join("reliquary", "candidate.json"),
        ]
        for p in tried:
            if p and os.path.exists(p):
                try:
                    data = json.load(open(p))
                except Exception:
                    logger.exception("smart: failed to read candidate file %s", p)
                    continue
                for env_name in self._smart_order:
                    grp = data.get(env_name) or {}
                    ids = grp.get("candidate_ids", []) or []
                    self._smart_candidates[env_name] = {int(i) for i in ids}
                logger.info(
                    "smart: loaded candidates from %s: %s", p,
                    {e: len(self._smart_candidates.get(e, set())) for e in self._smart_order},
                )
                return
        logger.error(
            "smart: candidate.json not found (tried %s); no candidates loaded",
            [t for t in tried if t],
        )
        for env_name in self._smart_order:
            self._smart_candidates.setdefault(env_name, set())

    async def _smart_pick(self, randomness: str, state, url, client):
        """Smart selection: candidate.json ∩ per-window slice, with OpenMath→
        OpenCode batch_filled fallback.

        Returns ``(env, env_name, prompt_idx, problem)`` or ``None`` when every
        active env is filled / has no eligible candidate this window. Consumes
        the chosen id so it isn't re-picked, and prunes ids that newly entered
        cooldown since the previous window (the "remove last window's cooldown"
        step).
        """
        from reliquary.miner.submitter import get_window_state_v2

        if self._smart_window_n != state.window_n:
            self._smart_window_n = state.window_n
            self._smart_current_env = self._smart_order[0]
            self._smart_filled = set()

        order = [e for e in self._smart_order if e in self.envs]
        active = [e for e in order if e not in self._smart_filled]
        if not active:
            return None
        if self._smart_current_env not in active:
            self._smart_current_env = active[0]
        env_name = self._smart_current_env
        env = self.envs[env_name]

        # Cooldown for this env; prune candidates that newly entered cooldown.
        try:
            es = await get_window_state_v2(url, env=env_name, client=client)
            cooldown = set(es.cooldown_prompts)
        except Exception:
            cooldown = set(state.cooldown_prompts)
        # Prune candidates that newly entered cooldown SINCE the previous
        # window. The first time we see an env we only record the baseline —
        # pruning against the full cooldown there would wipe candidate.json
        # (its ids are, by construction, the older cooldown population).
        prev = self._smart_prev_cooldown.get(env_name)
        if prev is not None:
            newly_cooled = cooldown - prev
            if newly_cooled:
                before = len(self._smart_candidates[env_name])
                self._smart_candidates[env_name] -= newly_cooled
                pruned = before - len(self._smart_candidates[env_name])
                if pruned:
                    logger.info(
                        "smart: pruned %d newly-cooled candidate(s) from %s",
                        pruned, env_name,
                    )
        self._smart_prev_cooldown[env_name] = cooldown

        # Eligible = candidate ids inside this window's slice.
        from reliquary.miner.prompt_scoring import eligible_in_slice
        lo, hi = window_prompt_range(
            randomness, env_name, len(env), PROMPT_RANGE_SIZE,
        )
        eligible = eligible_in_slice(self._smart_candidates[env_name], (lo, hi))
        if not eligible:
            logger.info(
                "smart: env=%s no eligible candidate in slice [%d,%d); skipping env",
                env_name, lo, hi,
            )
            self._smart_filled.add(env_name)
            return await self._smart_pick(randomness, state, url, client)

        idx = self._rng.choice(eligible)
        self._smart_candidates[env_name].discard(idx)  # consume
        problem = env.get_problem(idx)
        logger.info(
            "smart pick env=%s prompt=%d (eligible=%d in [%d,%d))",
            env_name, idx, len(eligible), lo, hi,
        )
        return env, env_name, idx, problem

    def _smart_after_submit(self, env_name: str, resp) -> None:
        """On batch_filled, mark the env filled and switch to the next one."""
        reason = resp.reason.value if hasattr(resp.reason, "value") else resp.reason
        if reason == "batch_filled":
            self._smart_filled.add(env_name)
            active = [
                e for e in self._smart_order
                if e in self.envs and e not in self._smart_filled
            ]
            self._smart_current_env = active[0] if active else env_name
            logger.info(
                "smart: %s batch_filled; %s",
                env_name,
                f"switching to {self._smart_current_env}" if active
                else "both envs filled, waiting for next window",
            )

    async def _probe_for_valid_prompt(
        self, env, randomness: str, cooldown_prompts: set[int], window_n: int,
    ):
        """Layer 4: cheap learning-frontier pre-screen.

        Pops scored candidates one at a time (Layers 1-3); for each, generates
        ``PROBE_ROLLOUTS`` quick rollouts and computes their binary reward
        vector. A *uniform* vector (all-0 = too hard, all-1 = too easy) means
        σ≈0 → the validator would reject the full group with ``OUT_OF_ZONE``,
        so the prompt is skipped. The first candidate with a *mixed* vector is
        a frontier prompt: return ``(prompt_idx, problem, probe_rollouts)``
        immediately, reusing the probe rollouts as part of the final group.
        Returns ``None`` when the window's candidates are exhausted with no
        frontier hit.
        """
        from reliquary.miner.prompt_scoring import is_frontier_signal

        while True:
            prompt_idx = self._next_scored_prompt(
                randomness, cooldown_prompts, window_n,
            )
            if prompt_idx is None:
                return None
            problem = env.get_problem(prompt_idx)
            probe = self._generate_m_rollouts(problem, randomness, n=PROBE_ROLLOUTS)
            signal = []
            for r in probe:
                text = self.tokenizer.decode(r["tokens"][r["prompt_length"]:])
                signal.append(1 if env.compute_reward(problem, text) > 0.5 else 0)
            sig_str = "".join(str(s) for s in signal)
            if not is_frontier_signal(signal):
                logger.info(
                    "Layer4 probe prompt=%d signal=%s uniform; skip (out_of_zone)",
                    prompt_idx, sig_str,
                )
                continue
            logger.info(
                "Layer4 probe prompt=%d signal=%s frontier; select",
                prompt_idx, sig_str,
            )
            return prompt_idx, problem, probe

    def _remember_rollout_hash(self, h: bytes) -> None:
        """Record a submitted rollout hash in the bounded LRU."""
        self._recent_rollout_hashes[h] = None
        self._recent_rollout_hashes.move_to_end(h)
        while len(self._recent_rollout_hashes) > self._recent_rollout_cap:
            self._recent_rollout_hashes.popitem(last=False)

    async def _window_changed(self, url, client, prior_state) -> bool:
        """True if the active window sealed/advanced since ``prior_state``.

        Generation + GRAIL take tens of seconds, during which the validator
        may seal the window and open the next one — binding the in-flight
        submission to a stale ``window_n`` / ``randomness`` that the validator
        will reject (``WINDOW_MISMATCH`` / ``WRONG_RANDOMNESS``). Re-poll
        ``/state`` so the caller can discard the work early instead of
        spending GRAIL on a doomed submission. Best-effort: a failed re-poll
        returns False (can't confirm a change → proceed and let the POST
        decide) rather than throwing away possibly-good work.
        """
        from reliquary.miner.submitter import get_window_state_v2
        from reliquary.protocol.submission import WindowState

        try:
            now = await get_window_state_v2(url, client=client)
        except Exception:
            return False
        return (
            now.state != WindowState.OPEN
            or now.window_n != prior_state.window_n
            or now.randomness != prior_state.randomness
        )

    def _next_scored_prompt(
        self, randomness: str, cooldown_prompts: set[int], window_n: int,
    ) -> int | None:
        """Score-based prompt selection — the layer between the window slice
        and cooldown rejection.

        Rebuilds a ranked top-``SCORED_CANDIDATE_POOL`` queue
        once per ``(window_n, randomness)``: derives the same ``[lo, hi)``
        slice the validator enforces, builds a calibration over the current
        cooldown population, then ranks the in-slice non-cooldown prompts by
        feature-typicality (``prompt_scoring.rank_candidates``). Subsequent
        calls in the same window pop the next-best idx, skipping any that
        entered cooldown since the queue was built. Returns ``None`` when the
        window's scored candidates are exhausted.
        """
        from reliquary.miner.prompt_scoring import (
            build_calibration, rank_candidates,
        )

        env = self._score_env
        env_label = getattr(env, "name", "openmathinstruct")
        key = (window_n, randomness)
        if self._scored_window_key != key:
            prompt_range = window_prompt_range(
                randomness, env_label, len(env), PROMPT_RANGE_SIZE,
            )
            try:
                cal = build_calibration(env, cooldown_prompts)
                if cal["n"] == 0:
                    # No cooldown population to score against yet (cold start).
                    # Scoring is degenerate here — every prompt ties at 0 and
                    # the lowest indices win, so all miners running this code
                    # collide on the same picks. Use the reference uniform
                    # picker until the cooldown set has signal.
                    self._scored_queue = self._uniform_queue(
                        prompt_range, cooldown_prompts,
                    )
                    logger.info(
                        "scored selection: window=%d cold start, uniform "
                        "fallback (%d picks)", window_n, len(self._scored_queue),
                    )
                else:
                    self._scored_queue = rank_candidates(
                        env, prompt_range, cooldown_prompts, cal,
                        top=SCORED_CANDIDATE_POOL,
                    )
                    logger.info(
                        "scored selection: window=%d ranked %d candidates "
                        "(cal n=%d)", window_n,
                        len(self._scored_queue), cal["n"],
                    )
            except Exception:
                # A dataset read hiccup must not kill the mine loop — degrade
                # to uniform selection for this window rather than propagate.
                logger.exception(
                    "scored selection failed; uniform fallback for window %d",
                    window_n,
                )
                self._scored_queue = self._uniform_queue(
                    prompt_range, cooldown_prompts,
                )
            self._scored_window_key = key

        while self._scored_queue:
            idx = self._scored_queue.pop(0)
            if idx not in cooldown_prompts:
                return idx
        return None

    def _uniform_queue(
        self, prompt_range: tuple[int, int], cooldown_prompts: set[int],
        k: int = SCORED_CANDIDATE_POOL,
    ) -> list[int]:
        """Up to ``k`` distinct uniform-random in-slice, non-cooldown picks.

        The reference selection strategy, used as the cold-start and
        error-recovery fallback for the scored layer.
        """
        picks: list[int] = []
        seen: set[int] = set()
        for _ in range(k):
            try:
                idx = pick_prompt_idx(
                    self._score_env, cooldown_prompts | seen,
                    rng=self._rng, prompt_range=prompt_range,
                )
            except RuntimeError:
                break
            seen.add(idx)
            picks.append(idx)
        return picks

    def _load_checkpoint(self, local_path: str):
        """Reload both hf_model and vllm_model from *local_path*.

        vllm_model is the fast-generation copy on ``self.vllm_gpu``;
        hf_model is the GRAIL-proof copy on ``self.proof_gpu``. The shared
        loader picks CausalLM for legacy text checkpoints and conditional
        text-only loading for Qwen3.5.
        """
        import torch

        from reliquary.constants import ATTN_IMPLEMENTATION
        from reliquary.shared.modeling import load_text_generation_model

        if getattr(self, "_loaded_checkpoint_path", None) == local_path:
            logger.debug("_load_checkpoint: already loaded from %s", local_path)
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)

        # 1. Reload hf_model (for GRAIL proofs) on the proof GPU.
        try:
            new_hf = load_text_generation_model(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.proof_gpu}").eval()
        except Exception:
            logger.exception(
                "Failed to reload hf_model from %s; keeping old model",
                local_path,
            )
            return self.hf_model

        old_hf = self.hf_model
        self.hf_model = new_hf
        del old_hf
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 2. Reload vllm_model on the generation GPU.
        try:
            new_gen = load_text_generation_model(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.vllm_gpu}").eval()
        except Exception:
            logger.exception(
                "Failed to reload vllm_model from %s; miner generation is "
                "BROKEN until the next successful pull. hf_model was swapped "
                "so GRAIL proofs will be inconsistent.",
                local_path,
            )
            self.vllm_model = None
            self._loaded_checkpoint_path = None
            return self.hf_model

        old_gen = self.vllm_model
        self.vllm_model = new_gen
        del old_gen
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self._loaded_checkpoint_path = local_path
        logger.info("Checkpoint %s loaded into both models", local_path)
        return self.hf_model

    def _generate_m_rollouts(self, problem, randomness, n: int = M_ROLLOUTS) -> list[dict]:
        """Generate *n* completions at T_PROTO in one batched call (default M).

        One .generate() with batch shape (n, prompt_len) is ~5-7×
        faster on GPU than n serial calls — the matmul tiling
        utilizes far more of the GPU's compute. Each row samples
        independently (do_sample=True), so GRPO-group semantics are
        preserved. Each output row is truncated at its first post-prompt
        EOS so trailing batch-padding (which HF pads with pad_token_id =
        eos_token_id) is not carried downstream — otherwise the validator's
        GRAIL forward pass would see extra EOS tokens the miner didn't
        "generate" in the usual sense.

        ``n`` is parametrized so the Layer-4 frontier probe can generate a
        cheap 4-rollout sample before committing to a full group.
        """
        import torch

        from reliquary.protocol.tokens import encode_prompt
        from reliquary.shared.modeling import first_eos_index, resolve_eos_token_ids

        prompt_tokens = encode_prompt(self.tokenizer, problem["prompt"])
        prompt_length = len(prompt_tokens)
        eos_ids = resolve_eos_token_ids(self.vllm_model, self.tokenizer)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None and eos_ids:
            pad_token_id = min(eos_ids)

        with torch.no_grad():
            input_tensor = torch.tensor(
                [prompt_tokens] * n,
                device=getattr(self.vllm_model, "device", "cpu"),
            )
            attention_mask = torch.ones_like(input_tensor)
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": True,
                "temperature": T_PROTO,
                "top_p": TOP_P_PROTO,
                "top_k": TOP_K_PROTO,
                "pad_token_id": pad_token_id,
                "attention_mask": attention_mask,
            }
            if eos_ids:
                generate_kwargs["eos_token_id"] = sorted(eos_ids)
            outputs = self.vllm_model.generate(input_tensor, **generate_kwargs)
        rollouts = []
        for i in range(n):
            seq = outputs[i].tolist()
            gen = seq[prompt_length:]
            first_eos = first_eos_index(gen, eos_ids)
            if first_eos is not None:
                gen = gen[: first_eos + 1]
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
            })
        return rollouts

    def _build_rollout_submission(self, generation, problem, randomness, *, env=None):
        """Build a RolloutSubmission: completion + claimed reward + GRAIL commit."""
        active_env = env if env is not None else self.env
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        if getattr(active_env, "validator_authoritative_reward", False):
            reward = 0.0
        else:
            reward = active_env.compute_reward(problem, completion_text)

        commit = self._build_grail_commit(generation, randomness)
        return RolloutSubmission(
            tokens=all_tokens,
            reward=reward,
            commit=commit,
            env_name=active_env.name,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _compute_randomness(
        self, subtensor, window_start: int, use_drand: bool
    ) -> str:
        """Derive window randomness from the drand beacon (v2.3+: drand-only).

        Matches the validator's ``service._derive_randomness``: block_hash is
        no longer mixed in, so the miner does not need a substrate roundtrip
        for the GRAIL seed. The legacy ``use_drand=False`` path remains for
        offline tests and uses block_hash as a single-source seed.
        """
        if use_drand:
            from reliquary.infrastructure.drand import get_beacon, get_current_chain

            chain_info = get_current_chain()
            drand_round = chain.compute_drand_round_for_window(
                window_start, chain_info["genesis_time"], chain_info["period"]
            )
            beacon = get_beacon(round_id=str(drand_round), use_drand=True)
            return chain.compute_window_randomness(
                None, beacon["randomness"], drand_round=beacon["round"]
            )
        block_hash = await chain.get_block_hash(subtensor, window_start)
        return chain.compute_window_randomness(block_hash)

    def _build_grail_commit(self, generation: dict, randomness: str) -> dict:
        """Construct a GRAIL proof commit dict from a generation dict.

        Reproduces the proof construction:
          - HF forward pass for hidden_states + logits
          - Commitment batch via GRAILVerifier
          - log-softmax token log-probs
          - Signature via sign_commit_binding
        """
        import torch

        from reliquary.constants import GRAIL_PROOF_VERSION
        from reliquary.protocol.signatures import sign_commit_binding
        from reliquary.shared.forward import forward_single_layer

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        # HF forward pass on proof GPU
        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{self.proof_gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]  # [seq_len, hidden_dim]

        # Build commitments
        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        # fp32 log_softmax to match the validator and reduce tail-token drift.
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs: list[float] = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        # Sign
        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")
        signature = sign_commit_binding(
            all_tokens, randomness, model_name, LAYER_INDEX,
            commitments, self.wallet,
        )

        return {
            "tokens": all_tokens,
            "commitments": commitments,
            "proof_version": GRAIL_PROOF_VERSION,
            "model": {"name": model_name, "layer_index": LAYER_INDEX},
            "signature": signature.hex(),
            "beacon": {"randomness": randomness},
            "rollout": {
                "prompt_length": prompt_length,
                "completion_length": len(all_tokens) - prompt_length,
                "success": True,
                "total_reward": 0.0,
                "advantage": 0.0,
                "token_logprobs": token_logprobs,
            },
        }
