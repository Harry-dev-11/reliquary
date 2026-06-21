"""Wall-clock timing instrumentation for the miner pipeline.

Records and logs how long the expensive stages take — model download, dataset
load, per-rollout generation, and per-rollout GRAIL proof — and writes a
rolling Markdown report (``result.md``) the operator can read at any time
while the miner runs.

A single process-wide recorder is shared by the CLI boot path and the
``MiningEngine`` via :func:`get_recorder`. The report path defaults to
``./result.md`` and is overridable with ``RELIQUARY_TIMING_RESULT``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

DEFAULT_RESULT_PATH = os.environ.get("RELIQUARY_TIMING_RESULT", "result.md")


def _stats(xs: list[float]) -> dict | None:
    """avg / min / max / p50 / p95 / total over a list of seconds."""
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return {
        "n": n,
        "avg": sum(s) / n,
        "min": s[0],
        "max": s[-1],
        "p50": s[n // 2],
        "p95": s[min(n - 1, int(round(n * 0.95)) - 1) if n > 1 else 0],
        "total": sum(s),
    }


class TimingRecorder:
    """Thread-safe collector of stage durations + Markdown report writer."""

    def __init__(self, result_path: str = DEFAULT_RESULT_PATH) -> None:
        self.result_path = result_path
        self._lock = threading.Lock()
        self.one_time: dict[str, float] = {}        # label -> seconds
        self.gen_batches: list[tuple] = []          # (window_n, seconds, n)
        self.grail_times: list[tuple] = []          # (window_n, seconds)
        self.windows_submitted = 0
        self.started_at = time.time()

    # -- recording -----------------------------------------------------------
    def record_one_time(self, label: str, seconds: float) -> None:
        """A one-off boot cost (model download, dataset load, model load)."""
        with self._lock:
            self.one_time[label] = seconds
        logger.info("[timing] %s: %.2fs", label, seconds)

    def record_generation(self, window_n, seconds: float, n_rollouts: int) -> None:
        """One batched generation of ``n_rollouts`` completions."""
        with self._lock:
            self.gen_batches.append((window_n, seconds, n_rollouts))
        per = seconds / n_rollouts if n_rollouts else 0.0
        logger.info(
            "[timing] generation window=%s: %d rollouts in %.2fs (%.2fs/rollout)",
            window_n, n_rollouts, seconds, per,
        )

    def record_grail(self, window_n, idx: int, total: int, seconds: float) -> None:
        """One rollout's GRAIL proof construction."""
        with self._lock:
            self.grail_times.append((window_n, seconds))
        logger.info(
            "[timing] grail window=%s rollout %d/%d: %.2fs",
            window_n, idx, total, seconds,
        )

    def mark_window_submitted(self) -> None:
        with self._lock:
            self.windows_submitted += 1

    # -- reporting -----------------------------------------------------------
    def _render(self) -> str:
        with self._lock:
            one_time = dict(self.one_time)
            gen_batches = list(self.gen_batches)
            grail_times = list(self.grail_times)
            windows = self.windows_submitted
            started = self.started_at

        now = time.time()
        lines: list[str] = []
        lines.append("# Miner Timing Report")
        lines.append("")
        lines.append(
            f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}"
            f" — uptime {now - started:.0f}s — {windows} window(s) submitted_"
        )
        lines.append("")

        # One-time costs
        lines.append("## One-time costs")
        lines.append("")
        if one_time:
            lines.append("| Stage | Seconds |")
            lines.append("|---|---:|")
            for label, secs in one_time.items():
                lines.append(f"| {label} | {secs:.2f} |")
        else:
            lines.append("_none recorded yet_")
        lines.append("")

        # Generation (per batch of M rollouts)
        gen_secs = [g[1] for g in gen_batches]
        per_rollout = [g[1] / g[2] for g in gen_batches if g[2]]
        lines.append("## Rollout generation (batched)")
        lines.append("")
        gs = _stats(gen_secs)
        if gs:
            prs = _stats(per_rollout)
            lines.append("| Metric | Batch (s) | Per-rollout (s) |")
            lines.append("|---|---:|---:|")
            lines.append(f"| count | {gs['n']} | {gs['n']} |")
            lines.append(f"| avg | {gs['avg']:.2f} | {prs['avg']:.2f} |")
            lines.append(f"| p50 | {gs['p50']:.2f} | {prs['p50']:.2f} |")
            lines.append(f"| p95 | {gs['p95']:.2f} | {prs['p95']:.2f} |")
            lines.append(f"| min | {gs['min']:.2f} | {prs['min']:.2f} |")
            lines.append(f"| max | {gs['max']:.2f} | {prs['max']:.2f} |")
            lines.append(f"| total | {gs['total']:.2f} | — |")
        else:
            lines.append("_no generations recorded yet_")
        lines.append("")

        # GRAIL proof (per rollout)
        grail_secs = [g[1] for g in grail_times]
        lines.append("## GRAIL proof (per rollout)")
        lines.append("")
        hs = _stats(grail_secs)
        if hs:
            lines.append("| Metric | Seconds |")
            lines.append("|---|---:|")
            lines.append(f"| count | {hs['n']} |")
            lines.append(f"| avg | {hs['avg']:.2f} |")
            lines.append(f"| p50 | {hs['p50']:.2f} |")
            lines.append(f"| p95 | {hs['p95']:.2f} |")
            lines.append(f"| min | {hs['min']:.2f} |")
            lines.append(f"| max | {hs['max']:.2f} |")
            lines.append(f"| total | {hs['total']:.2f} |")
        else:
            lines.append("_no GRAIL proofs recorded yet_")
        lines.append("")

        # Recent per-window breakdown (last 20)
        grail_by_win: dict = defaultdict(list)
        for w, s in grail_times:
            grail_by_win[w].append(s)
        lines.append("## Recent windows (last 20)")
        lines.append("")
        lines.append(
            "| window | gen batch (s) | gen/rollout (s) | grail avg (s) | grail total (s) |"
        )
        lines.append("|---:|---:|---:|---:|---:|")
        for (w, secs, n) in gen_batches[-20:]:
            gl = grail_by_win.get(w, [])
            g_avg = sum(gl) / len(gl) if gl else 0.0
            g_tot = sum(gl)
            per = secs / n if n else 0.0
            lines.append(
                f"| {w} | {secs:.2f} | {per:.2f} | {g_avg:.2f} | {g_tot:.2f} |"
            )
        lines.append("")
        return "\n".join(lines)

    def write_result_md(self) -> None:
        """Atomically (re)write the Markdown report. Never raises."""
        try:
            text = self._render()
            tmp = f"{self.result_path}.tmp"
            with open(tmp, "w") as f:
                f.write(text)
            os.replace(tmp, self.result_path)
        except Exception:
            logger.exception("failed to write timing report to %s", self.result_path)


_RECORDER: TimingRecorder | None = None


def get_recorder() -> TimingRecorder:
    """Return the process-wide timing recorder, creating it on first use."""
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = TimingRecorder()
    return _RECORDER
