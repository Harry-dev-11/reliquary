"""Score-based prompt selection layer (reliquary/miner/prompt_scoring.py)."""

from reliquary.miner.prompt_scoring import (
    build_calibration,
    rank_candidates,
    read_records,
)


class _FakeDataset:
    """Mimics the slice of the HF Dataset API the scorer touches: ``ds[list]``
    returns a column-major dict ``{col: [values...]}``."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, ids):
        cols = self._rows[0].keys()
        return {c: [self._rows[i][c] for i in ids] for c in cols}


class _FakeEnv:
    name = "openmathinstruct"

    def __init__(self, rows):
        self._dataset = _FakeDataset(rows)

    def __len__(self):
        return len(self._dataset)


def _row(problem, solution, answer, source="augmented_math"):
    return {
        "problem": problem,
        "generated_solution": solution,
        "expected_answer": answer,
        "problem_source": source,
    }


def _make_env(n=20):
    # Two families: "simple" short integer-answer rows and "complex" long
    # multi-step rows, so the calibration can distinguish them.
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append(_row("What is 2+2?", "Step 1.\nStep 2.", "4"))
        else:
            rows.append(_row(
                "A long multi step problem with 12 and 345 and 6789.",
                "\n".join(f"Line {k} compute {k * 7}." for k in range(40)),
                "-3/4",
            ))
    return _FakeEnv(rows)


def test_read_records_drops_out_of_range():
    env = _make_env(10)
    recs = read_records(env, [0, 3, 9, 10, -1, 999])
    assert set(recs) == {0, 3, 9}
    assert recs[0]["expected_answer"] == "4"


def test_rank_excludes_cooldown_and_respects_range():
    env = _make_env(20)
    cooldown = {0, 2, 4}
    cal = build_calibration(env, cooldown)
    ranked = rank_candidates(env, (0, 10), cooldown, cal, top=8)
    assert all(0 <= i < 10 for i in ranked)
    assert not (set(ranked) & cooldown)
    assert len(ranked) <= 8


def test_rank_favors_features_typical_of_cooldown():
    """Calibration built from the 'complex' family should rank complex rows
    above simple rows."""
    env = _make_env(40)
    complex_ids = [i for i in range(40) if i % 2 == 1]
    cal = build_calibration(env, complex_ids[:10])  # cooldown = complex rows
    # Score the even (simple) vs odd (complex) rows in a fresh slice.
    ranked = rank_candidates(env, (20, 40), cooldown_ids=set(), cal=cal, top=4)
    # Top picks should be odd (complex) indices — most typical of cooldown.
    assert all(i % 2 == 1 for i in ranked), ranked


def test_rank_deterministic_tiebreak():
    env = _make_env(20)
    cal = build_calibration(env, {1, 3, 5})
    a = rank_candidates(env, (0, 20), {1, 3, 5}, cal, top=8)
    b = rank_candidates(env, (0, 20), {1, 3, 5}, cal, top=8)
    assert a == b  # stable ordering across calls


def test_empty_cooldown_calibration_is_zero():
    """Cold start: no cooldown population → n=0 and every prompt scores 0.

    This is the degenerate case the engine guards against with a uniform
    fallback (see test_engine_cold_start_uniform_fallback)."""
    from reliquary.miner.prompt_scoring import score_record
    env = _make_env(10)
    cal = build_calibration(env, set())
    assert cal["n"] == 0
    rec = read_records(env, [0])[0]
    assert score_record(rec, cal) == 0.0


def test_source_filter():
    rows = [_row("p", "s", "1", source="gsm8k" if i < 5 else "augmented_math")
            for i in range(10)]
    env = _FakeEnv(rows)
    cal = build_calibration(env, set(range(10)))
    ranked = rank_candidates(
        env, (0, 10), set(), cal, top=8, sources={"augmented_math"},
    )
    assert all(i >= 5 for i in ranked)
