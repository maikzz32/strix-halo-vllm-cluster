#!/usr/bin/env python3
"""Patch 75: report vLLM's built-in per-step timing (target forward vs drafter).

Why: patch 74 measured the draft step itself at 1.93 ms (forward 1.22 + sample 0.70),
but the throughput numbers attribute 4.98 ms per step. Iteration 46.73 ms minus the
31.80 ms target model leaves 14.93 ms for three draft steps, of which only 5.79 ms are
actually spent inside `_generate_draft`. So ~9.14 ms per iteration sit somewhere else.

The prime suspect is verification: with k=3 the target model runs a forward over FOUR
tokens instead of one, which for a mixture-of-experts means 40 expert invocations
instead of 10. The decomposition "iteration = target + k x draft" silently charges that
growth to the draft.

vLLM already has exactly the right instrumentation and does not use it in normal
operation: `StepTimingCollector` (v1/worker/gpu/async_utils.py) records
forward_start/forward_end around the target forward and drafter_start/drafter_end
around the whole draft loop -- but `_collecting` is only ever true inside
`collect()`, which the model runner enters solely for adaptive verification's cost
curves (model_runner.py:948). The record calls themselves are already in the hot path
at model_runner.py:1734 / 777 / 782 / 825.

This patch therefore adds no measurement points. It only
  * turns `_collecting` on when VLLM_GFX1X_STEP_TIMING=N (N>0), and
  * drains the queue every N steps, logging mean forward_ms and drafter_ms.

Draining matters: with `_collecting` on and nothing consuming it, `_timed` would grow
without bound (drafter_end appends one entry per step, async_utils.py:111).

As in patch 74, no synchronize() in the hot path -- entries are harvested only once
their end event reports ready via query(), the rest carry over to the next call.

Gate: VLLM_GFX1X_STEP_TIMING=0 (default off) / N = log every N steps.
Note: async_utils.py has no module-level logger, so the report grabs one lazily.

Usage: python3 75_step_timing_report.py --src <site-packages> [--check]
Exit codes: 0 applied/ok, 1 check failed, 42 anchor moved -> re-audit.
Written against vLLM v0.29.0rc1 (33898f832c).
"""
import argparse
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 75_step_timing_report"
REL = "vllm/v1/worker/gpu/async_utils.py"

OLD_INIT = '''    def __init__(self):
        self._collecting = False
        self._step: StepTimingEvents | None = None
        self._batch = (False, 0, 0)
        self._timed: list[tuple[StepTimingEvents, tuple[bool, int, int]]] = []
'''

NEW_INIT = '''    def __init__(self):
        self._collecting = False
        self._step: StepTimingEvents | None = None
        self._batch = (False, 0, 0)
        self._timed: list[tuple[StepTimingEvents, tuple[bool, int, int]]] = []
        # ''' + MARKER + '''
        # The record calls are already in the hot path; only `_collecting` gates them,
        # and it is normally true just inside collect(). Turning it on here reuses the
        # existing instrumentation to split target forward from draft loop.
        import os as _gfx_os

        try:
            self._gfx_every = int(_gfx_os.environ.get("VLLM_GFX1X_STEP_TIMING", "0"))
        except ValueError:
            self._gfx_every = 0
        self._gfx_acc = [0.0, 0.0, 0, 0]   # forward_ms, drafter_ms, tokens, n
        self._gfx_steps = 0
        if self._gfx_every > 0:
            self._collecting = True

    def _gfx_report(self) -> None:
        """''' + MARKER + '''

        Drain finished entries and log a mean every N steps. Without draining,
        `_timed` would grow unbounded once `_collecting` is on outside collect().
        """
        if self._gfx_every <= 0:
            return
        still = []
        for events, batch in self._timed:
            if events.drafter_end.query() and events.forward_start.query():
                self._gfx_acc[0] += events.forward_start.elapsed_time(events.forward_end)
                self._gfx_acc[1] += events.drafter_start.elapsed_time(events.drafter_end)
                self._gfx_acc[2] += batch[1]
                self._gfx_acc[3] += 1
            else:
                still.append((events, batch))
        self._timed = still
        self._gfx_steps += 1
        if self._gfx_steps % self._gfx_every or self._gfx_acc[3] == 0:
            return
        fwd, drf, toks, n = self._gfx_acc
        # async_utils.py hat keinen Modul-Logger -> lazy holen (einmal je Report).
        from vllm.logger import init_logger

        _log = init_logger(__name__)
        _log.info(
            "''' + MARKER + '''| steps=%d | target_forward=%.3fms drafter=%.3fms "
            "sum=%.3fms | target_tokens/step=%.2f",
            self._gfx_steps,
            fwd / n,
            drf / n,
            (fwd + drf) / n,
            toks / n,
        )
        self._gfx_acc = [0.0, 0.0, 0, 0]
'''

OLD_END = '''            self._step.drafter_end.record()
            self._timed.append((self._step, self._batch))
            self._step = None
'''

NEW_END = '''            self._step.drafter_end.record()
            self._timed.append((self._step, self._batch))
            self._step = None
            # ''' + MARKER + '''
            self._gfx_report()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/usr/local/lib64/python3.12/site-packages")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    f = Path(a.src) / REL
    if not f.is_file():
        print(f"ERROR: {REL} not found under {a.src}", file=sys.stderr)
        return 42
    s = f.read_text()
    if a.check:
        ok = MARKER in s
        print(("OK: patch 75 present in " if ok else "FAIL: patch 75 marker not found in ") + str(f))
        return 0 if ok else 1
    if MARKER in s:
        print(f"SKIP: patch 75 already applied to {f}")
        return 0
    if "class StepTimingCollector" not in s:
        print(f"ERROR: {f} has no StepTimingCollector; re-audit patch 75", file=sys.stderr)
        return 42
    for name, old in (("__init__", OLD_INIT), ("drafter_end", OLD_END)):
        if s.count(old) != 1:
            print(
                f"ERROR: {name} anchor found {s.count(old)} times in {f}; re-audit patch 75",
                file=sys.stderr,
            )
            return 42
    s = s.replace(OLD_INIT, NEW_INIT, 1).replace(OLD_END, NEW_END, 1)
    f.write_text(s)
    print(f"OK: patch 75 applied to {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
