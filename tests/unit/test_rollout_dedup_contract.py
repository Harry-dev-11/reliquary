"""Contract: the miner's local distinct-rollout guard must predict the
validator's HASH_DUPLICATE check exactly.

The miner hashes each rollout's tokens with the same scheme the validator
uses (``reliquary.validator.dedup.compute_rollout_hash``) so it can skip a
group before submitting it instead of eating a HASH_DUPLICATE rejection.
This test pins the scheme and the within-group / cross-submission logic.
"""

import hashlib

from reliquary.validator.dedup import compute_rollout_hash


def _miner_hash(tokens):
    """Mirror of reliquary.miner.engine._rollout_token_hash (kept in sync)."""
    h = hashlib.sha256()
    for t in tokens:
        h.update(int(t).to_bytes(4, "big", signed=False))
    return h.digest()


def test_hash_scheme_matches_validator():
    for toks in ([1, 2, 3], [151643, 0, 99, 42], list(range(100))):
        assert _miner_hash(toks) == compute_rollout_hash(toks)


def test_within_group_duplicate_detected():
    # Two identical rollouts in a group → validator rejects whole group.
    groups = [[1, 2, 3], [1, 2, 3], [4, 5, 6]]
    hashes = [_miner_hash(g) for g in groups]
    assert len(set(hashes)) < len(hashes)  # guard fires → miner skips


def test_all_distinct_passes():
    groups = [[1, 2, 3], [1, 2, 4], [4, 5, 6]]
    hashes = [_miner_hash(g) for g in groups]
    assert len(set(hashes)) == len(hashes)  # guard does not fire


def test_cross_submission_duplicate_detected():
    recent = {_miner_hash([7, 8, 9])}
    new_group = [[7, 8, 9], [1, 1, 1]]
    new_hashes = [_miner_hash(g) for g in new_group]
    assert any(h in recent for h in new_hashes)  # guard fires → miner skips
