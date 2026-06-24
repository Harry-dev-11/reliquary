"""Tests for the multi-env prompt selection logic in the miner engine."""

import random
import pytest


class _FakeEnv:
    """Minimal Environment stub: name + len + get_problem stub."""
    def __init__(self, name: str, size: int):
        self.name = name
        self._size = size
    def __len__(self):
        return self._size
    def get_problem(self, idx):
        return {"prompt": f"{self.name}-{idx}", "ground_truth": "", "id": "x" * 16}


def test_pick_env_and_prompt_returns_env_and_idx():
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {
        "openmathinstruct": _FakeEnv("openmathinstruct", 100),
        "opencodeinstruct": _FakeEnv("opencodeinstruct", 50),
    }
    mix = [("openmathinstruct", 8), ("opencodeinstruct", 8)]
    cooldown = {name: set() for name in envs}
    rng = random.Random(0)
    env_name, idx = pick_env_and_prompt(envs, mix, cooldown, rng=rng)
    assert env_name in {"openmathinstruct", "opencodeinstruct"}
    assert 0 <= idx < len(envs[env_name])


def test_pick_env_and_prompt_respects_weights():
    """With weights 1:9, the rare env should be chosen far less often."""
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {"a": _FakeEnv("a", 10), "b": _FakeEnv("b", 10)}
    mix = [("a", 1), ("b", 9)]
    cooldown = {"a": set(), "b": set()}
    rng = random.Random(42)
    counts = {"a": 0, "b": 0}
    for _ in range(1000):
        env_name, _ = pick_env_and_prompt(envs, mix, cooldown, rng=rng)
        counts[env_name] += 1
    # 1:9 weights → roughly 100:900. Allow generous slack.
    assert 70 < counts["a"] < 150
    assert 850 < counts["b"] < 950


def test_pick_env_and_prompt_skips_env_in_full_cooldown():
    """If one env is fully in cooldown, sampling still works on the other."""
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {"a": _FakeEnv("a", 5), "b": _FakeEnv("b", 5)}
    mix = [("a", 1), ("b", 1)]
    cooldown = {"a": set(range(5)), "b": set()}  # 'a' fully blocked
    rng = random.Random(0)
    for _ in range(50):
        env_name, idx = pick_env_and_prompt(envs, mix, cooldown, rng=rng)
        assert env_name == "b"  # never 'a'


def test_pick_env_and_prompt_falls_back_when_candidates_are_ineligible():
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {"a": _FakeEnv("a", 100)}
    mix = [("a", 1)]
    cooldown = {"a": {20, 21, 22}}
    rng = random.Random(0)
    env_name, idx = pick_env_and_prompt(
        envs,
        mix,
        cooldown,
        candidate_indices_per_env={"a": (20, 21, 22)},
        rng=rng,
    )
    assert env_name == "a"
    assert 0 <= idx < 100
    assert idx not in cooldown["a"]


def test_pick_env_and_prompt_prefers_eligible_candidates():
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {"a": _FakeEnv("a", 100)}
    mix = [("a", 1)]
    cooldown = {"a": {10}}
    rng = random.Random(0)
    env_name, idx = pick_env_and_prompt(
        envs,
        mix,
        cooldown,
        candidate_indices_per_env={"a": (10, 25, 90)},
        rng=rng,
    )
    assert env_name == "a"
    assert idx in {25, 90}


def test_pick_env_and_prompt_prioritizes_opencode_then_openmath_candidates():
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {
        "openmathinstruct": _FakeEnv("openmathinstruct", 100),
        "opencodeinstruct": _FakeEnv("opencodeinstruct", 100),
    }
    mix = [("openmathinstruct", 1), ("opencodeinstruct", 1)]
    cooldown = {"openmathinstruct": set(), "opencodeinstruct": set()}
    rng = random.Random(0)
    env_name, idx = pick_env_and_prompt(
        envs,
        mix,
        cooldown,
        candidate_indices_per_env={
            "openmathinstruct": (11, 12),
            "opencodeinstruct": (21, 22),
        },
        rng=rng,
    )
    assert env_name == "opencodeinstruct"
    assert idx in {21, 22}


def test_pick_env_and_prompt_uses_openmath_candidates_when_opencode_has_none():
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {
        "openmathinstruct": _FakeEnv("openmathinstruct", 100),
        "opencodeinstruct": _FakeEnv("opencodeinstruct", 100),
    }
    mix = [("openmathinstruct", 1), ("opencodeinstruct", 1)]
    cooldown = {"openmathinstruct": set(), "opencodeinstruct": {21, 22}}
    rng = random.Random(0)
    env_name, idx = pick_env_and_prompt(
        envs,
        mix,
        cooldown,
        candidate_indices_per_env={
            "openmathinstruct": (11, 12),
            "opencodeinstruct": (21, 22),
        },
        rng=rng,
    )
    assert env_name == "openmathinstruct"
    assert idx in {11, 12}


def test_pick_env_and_prompt_skips_blocked_envs():
    from reliquary.miner.engine import pick_env_and_prompt
    envs = {
        "openmathinstruct": _FakeEnv("openmathinstruct", 100),
        "opencodeinstruct": _FakeEnv("opencodeinstruct", 100),
    }
    mix = [("openmathinstruct", 1), ("opencodeinstruct", 1)]
    cooldown = {"openmathinstruct": set(), "opencodeinstruct": set()}
    rng = random.Random(0)
    env_name, idx = pick_env_and_prompt(
        envs,
        mix,
        cooldown,
        candidate_indices_per_env={
            "openmathinstruct": (11, 12),
            "opencodeinstruct": (21, 22),
        },
        blocked_envs={"opencodeinstruct"},
        rng=rng,
    )
    assert env_name == "openmathinstruct"
    assert idx in {11, 12}
