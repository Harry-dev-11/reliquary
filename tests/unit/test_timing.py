"""Miner timing recorder + result.md report writer."""

from reliquary.miner.timing import TimingRecorder, _stats


def test_stats_basic():
    s = _stats([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4
    assert s["min"] == 1.0
    assert s["max"] == 4.0
    assert s["total"] == 10.0
    assert abs(s["avg"] - 2.5) < 1e-9


def test_stats_empty_is_none():
    assert _stats([]) is None


def test_records_and_writes_report(tmp_path):
    path = tmp_path / "result.md"
    rec = TimingRecorder(result_path=str(path))

    rec.record_one_time("model download (boot seed)", 42.5)
    rec.record_one_time("dataset load", 7.25)
    rec.record_generation(window_n=100, seconds=80.0, n_rollouts=8)
    for i in range(1, 9):
        rec.record_grail(window_n=100, idx=i, total=8, seconds=2.0 + i * 0.1)
    rec.mark_window_submitted()
    rec.write_result_md()

    text = path.read_text()
    assert "# Miner Timing Report" in text
    assert "model download (boot seed)" in text
    assert "42.50" in text
    assert "## Rollout generation (batched)" in text
    assert "## GRAIL proof (per rollout)" in text
    # per-rollout generation = 80/8 = 10.00
    assert "10.00" in text
    # recent-window row carries the window number
    assert "| 100 |" in text


def test_report_handles_no_data(tmp_path):
    path = tmp_path / "result.md"
    rec = TimingRecorder(result_path=str(path))
    rec.write_result_md()
    text = path.read_text()
    assert "none recorded yet" in text
    assert "no generations recorded yet" in text


def test_write_is_atomic_and_overwrites(tmp_path):
    path = tmp_path / "result.md"
    rec = TimingRecorder(result_path=str(path))
    rec.record_generation(1, 10.0, 8)
    rec.write_result_md()
    first = path.read_text()
    rec.record_generation(2, 20.0, 8)
    rec.write_result_md()
    second = path.read_text()
    assert second != first
    assert "| 2 |" in second
    assert not (tmp_path / "result.md.tmp").exists()  # tmp cleaned up
