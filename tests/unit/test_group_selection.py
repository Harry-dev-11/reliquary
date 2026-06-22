"""Unit tests for the miner's opt-in group/key-id prompt selection.

Pure-Python (no torch/transformers), so these run in the normal unit suite.
"""

import json

import pytest

from reliquary.miner.group_selection import (
    GroupSelector,
    build_group_index,
    resolve_group_paths,
)


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


@pytest.fixture
def groups_file(tmp_path):
    # group 1 = {10,11,12,13}, group 2 = {20,21}
    return _write(tmp_path / "group.json", [
        {"group_id": 1, "ids": [10, 11, 12, 13]},
        {"group_id": 2, "ids": [20, 21]},
    ])


def test_build_group_index_merges_parts(tmp_path):
    p1 = _write(tmp_path / "g1.json", [{"group_id": 1, "ids": [10, 11]}])
    p2 = _write(tmp_path / "g2.json", [{"group_id": 2, "ids": [20]}])
    id2group, group_ids = build_group_index([p1, p2])
    assert id2group == {10: 1, 11: 1, 20: 2}
    assert group_ids == {1: [10, 11], 2: [20]}


def test_resolve_group_paths_prefers_given(tmp_path):
    p = _write(tmp_path / "custom.json", [])
    assert resolve_group_paths([p]) == [p]
    assert resolve_group_paths(["/does/not/exist.json"]) == []


def _selector(tmp_path, groups_file, key_ids, **kw):
    kp = _write(tmp_path / "key_id.json", {"ids": key_ids})
    cand = str(tmp_path / "candidate.json")
    return GroupSelector(kp, [groups_file], "openmathinstruct",
                         candidate_path=cand, **kw), kp


def test_selects_other_in_slice_candidate(tmp_path, groups_file, monkeypatch):
    # Force a deterministic slice covering [10, 14) so all of group 1 is in-slice.
    monkeypatch.setattr(
        "reliquary.miner.group_selection.window_prompt_range",
        lambda *a, **k: (10, 14),
    )
    gs, _ = _selector(tmp_path, groups_file, [10])
    sel = gs.next_selection("rand", universe_n=1000, live_cooldown=set())
    assert sel is not None
    key_id, selected, gid = sel
    assert key_id == 10 and gid == 1
    assert selected != key_id            # never the key id itself
    assert selected in (11, 12, 13)      # another in-slice group member


def test_cooldown_excluded_from_candidates(tmp_path, groups_file, monkeypatch):
    monkeypatch.setattr(
        "reliquary.miner.group_selection.window_prompt_range",
        lambda *a, **k: (10, 14),
    )
    gs, _ = _selector(tmp_path, groups_file, [10])
    # cooldown removes 11 and 12 → only 13 eligible
    sel = gs.next_selection("rand", 1000, {11, 12})
    assert sel[1] == 13


def test_no_candidate_in_slice_skips_window(tmp_path, groups_file, monkeypatch):
    # Slice excludes the whole group → no selection this window.
    monkeypatch.setattr(
        "reliquary.miner.group_selection.window_prompt_range",
        lambda *a, **k: (100, 200),
    )
    gs, _ = _selector(tmp_path, groups_file, [10])
    assert gs.next_selection("rand", 1000, set()) is None


def test_no_group_key_marked_dead(tmp_path, groups_file, monkeypatch):
    monkeypatch.setattr(
        "reliquary.miner.group_selection.window_prompt_range",
        lambda *a, **k: (0, 1000),
    )
    gs, _ = _selector(tmp_path, groups_file, [99999])  # not in any group
    assert gs.next_selection("rand", 1000, set()) is None
    assert gs.exhausted()


def test_prune_rewrites_file_and_advances(tmp_path, groups_file, monkeypatch):
    monkeypatch.setattr(
        "reliquary.miner.group_selection.window_prompt_range",
        lambda *a, **k: (10, 22),
    )
    gs, kp = _selector(tmp_path, groups_file, [10, 20])
    first = gs.next_selection("rand", 1000, set())
    assert first[0] == 10
    gs.prune(10)
    assert json.loads(open(kp).read()) == {"ids": [20]}   # shape preserved
    nxt = gs.next_selection("rand", 1000, set())
    assert nxt[0] == 20


def test_in_zone_band(tmp_path, groups_file):
    gs, _ = _selector(tmp_path, groups_file, [10], zone_low=2, zone_high=6)
    assert [gs.in_zone(n) for n in range(9)] == [
        False, False, True, True, True, True, True, False, False,
    ]


def test_empty_key_ids_raises(tmp_path, groups_file):
    kp = _write(tmp_path / "key_id.json", {"ids": []})
    with pytest.raises(ValueError):
        GroupSelector(kp, [groups_file], "openmathinstruct")
