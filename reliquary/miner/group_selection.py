"""Group-based prompt selection for the miner.

An opt-in alternative to the stock uniform-random ``pick_env_and_prompt``. The
miner walks a list of *key ids* (``key_id.json``); each key id identifies a
prompt *group* (``group.part*.json``). Within that group it keeps the members
that are NOT in the validator's live cooldown, intersects them with the current
window slice, and rolls out one OTHER in-slice candidate (never the key id
itself). Out-of-zone groups (see the engine's zone gate) and successfully
submitted ones prune their key id from ``key_id.json``.

This module owns only the selection state + file IO; the engine owns generation,
the zone gate, GRAIL, and submission. Selection is confined to a single env
(``env_name``) — the env whose index space the groups were built from.
"""

from __future__ import annotations

import json
import logging
import os

from reliquary.constants import PROMPT_RANGE_SIZE
from reliquary.shared.prompt_range import window_prompt_range

logger = logging.getLogger(__name__)


def _load_ids(path: str) -> list[int]:
    """Read ids from {"ids":[...]}, {"id":[...]}, or a bare list. [] if missing."""
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


def _key_container(path: str) -> "str | None":
    """Detect how key_id.json wraps its ids so rewrites keep the same shape."""
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


def build_group_index(
    group_paths: list[str],
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Merge one or more group files into (id -> group_id, group_id -> ids).

    group.json is split into group.part1.json + group.part2.json to stay under
    GitHub's 100MB limit; both parts merge into the same index.
    """
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


def resolve_group_paths(group_paths: "list[str] | None") -> list[str]:
    """Pick whichever group files exist: the given paths, the default split
    parts, or a single monolithic group.json. Returns [] if none found."""
    if group_paths:
        existing = [p for p in group_paths if os.path.exists(p)]
        if existing:
            return existing
    for default in (["group.part1.json", "group.part2.json"], ["group.json"]):
        if all(os.path.exists(p) for p in default):
            return default
    return []


class GroupSelector:
    """Stateful key-id walker for the miner loop.

    One ``next_selection`` call returns the next (key_id, selected_idx, group_id)
    whose group has an in-slice, non-cooldown candidate for the current window;
    the miner generates + zone-gates + submits it, then calls ``prune`` on
    out-of-zone or accepted. State persists across loop iterations and resets its
    per-window skip set when the window randomness changes.
    """

    def __init__(
        self,
        key_ids_path: str,
        group_paths: list[str],
        env_name: str,
        *,
        zone_low: int = 2,
        zone_high: int = 6,
        success_threshold: float = 1.0,
        candidate_path: "str | None" = "candidate.json",
    ) -> None:
        self.key_ids_path = key_ids_path
        self.env_name = env_name
        self.zone_low = zone_low
        self.zone_high = zone_high
        self.success_threshold = success_threshold
        self.candidate_path = candidate_path

        self._container = _key_container(key_ids_path)
        self.remaining: list[int] = _load_ids(key_ids_path)
        if not self.remaining:
            raise ValueError(f"{key_ids_path} is empty — no key ids to walk")

        self.id2group, self.group_ids = build_group_index(group_paths)
        logger.info(
            "GroupSelector: %d key ids, %d groups (%d ids indexed) for env %s",
            len(self.remaining), len(self.group_ids), len(self.id2group), env_name,
        )

        self._dead: set[int] = set()           # no_group key ids — never match
        self._skip_window: set[int] = set()     # no candidate in THIS window
        self._cur_randomness: str = ""

    def in_zone(self, n_success: int) -> bool:
        """True when the reward-1 count is in [zone_low, min(zone_high, group)]."""
        return self.zone_low <= n_success <= self.zone_high

    def next_selection(
        self, randomness: str, universe_n: int, live_cooldown: set[int],
    ) -> "tuple[int, int, int] | None":
        """Return (key_id, selected_idx, group_id) for the next walkable key id,
        or None if no remaining key id lands in this window's slice."""
        if randomness != self._cur_randomness:
            self._cur_randomness = randomness
            self._skip_window = set()  # new window → re-try everything

        lo, hi = window_prompt_range(randomness, self.env_name, universe_n, PROMPT_RANGE_SIZE)

        for key_id in list(self.remaining):
            if key_id in self._dead or key_id in self._skip_window:
                continue
            gid = self.id2group.get(key_id)
            if gid is None:
                self._dead.add(key_id)
                continue
            members = self.group_ids.get(gid, [])
            candidates = [i for i in members if i not in live_cooldown]
            if self.candidate_path:
                with open(self.candidate_path, "w") as f:
                    json.dump({"key_id": key_id, "group_id": gid,
                               "ids": sorted(candidates)}, f)
            # selection pool = OTHER in-slice, non-cooldown members (never key id)
            accepted = [i for i in candidates if lo <= i < hi and i != key_id]
            if not accepted:
                self._skip_window.add(key_id)
                continue
            return key_id, min(accepted), gid
        return None

    def prune(self, key_id: int) -> None:
        """Remove a key id from key_id.json (persisted) and from the walk."""
        if key_id in self.remaining:
            self.remaining.remove(key_id)
            payload = (list(self.remaining) if self._container is None
                       else {self._container: list(self.remaining)})
            with open(self.key_ids_path, "w") as f:
                json.dump(payload, f, indent=2)
        self._skip_window.discard(key_id)

    def exhausted(self) -> bool:
        """True when no walkable key ids remain (all pruned or dead)."""
        return all(k in self._dead for k in self.remaining)
