#!/usr/bin/env python3
"""Patch 76: time the TARGET model forward in the production path.

Why this exists even though vLLM has a StepTimingCollector: that collector is only
wired up completely inside `_dummy_run` (model_runner.py:777/782/825, i.e. graph
capture). The production path has `record_batch` and `forward_start` (1731/1734) but
NO `forward_end` and no drafter markers, so `_timed` never fills and patch 75 -- which
merely switches `_collecting` on -- reports nothing. Verified on the cluster: zero
output over a full benchmark.

What we already know (measured, not assumed):
  * iteration at k=3 is 46.73 ms, at k=0 it is 31.80 ms
  * patch 74 timed the draft step directly: 1.93 ms (forward 1.22 + sample 0.70)
  * so three draft steps are 5.79 ms, leaving 40.94 ms -- 9.14 ms MORE than the
    31.80 ms that the target model needs on its own

The open question is where those 9.14 ms sit. The prime suspect is verification: at
k=3 the target forward runs over FOUR tokens instead of one, which for a
mixture-of-experts means 40 expert invocations instead of 10. If that is right,
`target_forward` here will come out ~9 ms above the k=0 value. If it comes out equal,
the time is in the surrounding bookkeeping (rejection sampling, scheduler) and the
search continues there.

Method as in patch 74: HIP events, no synchronize() in the hot path, results harvested
only once `query()` reports them ready.

Gate: VLLM_GFX1X_TARGET_TIMING=0 (default off) / N = log every N forwards.

Usage: python3 76_target_forward_timing.py --src <site-packages> [--check]
Exit codes: 0 applied/ok, 1 check failed, 42 anchor moved -> re-audit.
Written against vLLM v0.29.0rc1 (33898f832c).
"""
import argparse
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 76_target_forward_timing"
REL = "vllm/v1/worker/gpu/model_runner.py"

# --- start: right where the (incomplete) built-in marker already sits ----------
OLD_START = '''        self.step_timing.forward_start()
'''
NEW_START = '''        self.step_timing.forward_start()
        # ''' + MARKER + '''
        # The built-in collector is only completed inside _dummy_run, so time the
        # target forward separately here.
        _gfx_tt = self._gfx_target_timer()
        _gfx_tt_tok = _gfx_tt.begin(input_batch.num_tokens) if _gfx_tt is not None else None
'''

# --- end: immediately after the forward, before the output is unpacked --------
OLD_END = '''        if self.is_last_pp_rank:
            if self.use_aux_hidden_state_outputs:
                assert isinstance(model_output, tuple)
                hidden_states, aux_hidden_states = model_output
'''
NEW_END = '''        # ''' + MARKER + '''
        if _gfx_tt is not None:
            _gfx_tt.end(_gfx_tt_tok)

        if self.is_last_pp_rank:
            if self.use_aux_hidden_state_outputs:
                assert isinstance(model_output, tuple)
                hidden_states, aux_hidden_states = model_output
'''

# --- the timer helper, added to the class -------------------------------------
OLD_HELPER = '''    def _remove_request(self, req_id: str) -> bool:
'''
NEW_HELPER = '''    def _gfx_target_timer(self):
        """''' + MARKER + '''

        Times the target-model forward and logs a mean every N calls, split by the
        number of tokens in the batch -- that is the whole point here, because at
        k=3 the verify forward carries four tokens instead of one.
        """
        t = getattr(self, "_gfx_tt", None)
        if t is not None:
            return t if t is not False else None
        import os as _os

        try:
            every = int(_os.environ.get("VLLM_GFX1X_TARGET_TIMING", "0"))
        except ValueError:
            every = 0
        if every <= 0:
            self._gfx_tt = False
            return None

        class _TargetTimer:
            def __init__(self, every_n):
                self.every_n = every_n
                self.pending = []
                self.acc = {}      # num_tokens -> [total_ms, count]
                self.calls = 0

            def begin(self, num_tokens):
                import torch as _t

                e = _t.cuda.Event(enable_timing=True)
                e.record()
                return (num_tokens, e)

            def end(self, token):
                import torch as _t

                if token is None:
                    return
                num_tokens, start = token
                e = _t.cuda.Event(enable_timing=True)
                e.record()
                self.pending.append((num_tokens, start, e))
                self.calls += 1
                self._harvest()
                if self.calls % self.every_n == 0:
                    self._report()

            def _harvest(self):
                still = []
                for n, s, e in self.pending:
                    if e.query() and s.query():
                        slot = self.acc.setdefault(n, [0.0, 0])
                        slot[0] += s.elapsed_time(e)
                        slot[1] += 1
                    else:
                        still.append((n, s, e))
                self.pending = still

            def _report(self):
                if not self.acc:
                    return
                parts = [
                    f"{n}tok={ms / c:.3f}ms(n={c})"
                    for n, (ms, c) in sorted(self.acc.items())
                ]
                logger.info(
                    "''' + MARKER + '''| forwards=%d | target forward by batch size: %s",
                    self.calls,
                    " ".join(parts),
                )
                self.acc = {}

        self._gfx_tt = _TargetTimer(every)
        logger.info(
            "''' + MARKER + ''': target forward timing on, logging every %d forwards.",
            every,
        )
        return self._gfx_tt

    def _remove_request(self, req_id: str) -> bool:
'''

STEPS = (
    ("forward start", OLD_START, NEW_START),
    ("forward end", OLD_END, NEW_END),
    ("timer helper", OLD_HELPER, NEW_HELPER),
)


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
        print(("OK: patch 76 present in " if ok else "FAIL: patch 76 marker not found in ") + str(f))
        return 0 if ok else 1
    if MARKER in s:
        print(f"SKIP: patch 76 already applied to {f}")
        return 0
    for name, old, _ in STEPS:
        if s.count(old) != 1:
            print(
                f"ERROR: {name} anchor found {s.count(old)} times in {f}; re-audit patch 76",
                file=sys.stderr,
            )
            return 42
    for _, old, new in STEPS:
        s = s.replace(old, new, 1)
    f.write_text(s)
    print(f"OK: patch 76 applied to {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
