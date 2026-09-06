#!/usr/bin/env python3
"""Patch 74: measure where the MTP draft step actually spends its 5.08 ms.

Motivation (all measured on the gfx1151 cluster, 2026-09-06):
iteration = 31.80 ms (48-layer target model) + k x 5.08 ms per draft step. One draft
step is a SINGLE layer and thus costs 11x a target layer. Three explanations were
tested and each was refuted by measurement:
  * Python metadata rebuild between steps -> patch 73 (fused path) changed nothing
  * kernel launch latency -> all k steps as ONE graph replay changed nothing
  * MoE weight bandwidth -> draft head BF16 5.03 GB -> INT4 1.455 GB changed nothing
    (the MoE path reads sparse: ~10 of 512 experts, ~98 MB)
So the time is somewhere else, and guessing has failed three times.

Both ready-made profilers are unusable here: the torch profiler dies on export, and
rocprofv3 hangs (verified twice, exit 124 on a trivial matmul; a leftover run from a
previous session had been hanging for 18h). Hence in-code timing.

What it does: wraps the three phases of `_generate_draft` in HIP events
(torch.cuda.Event) -- `_run_model` (the draft forward), `sample_draft`, and
`update_draft_inputs` -- accumulates them, and logs a table every N draft steps.

Design constraints that matter:
  * NO torch.cuda.synchronize() in the hot path -- that would serialise exactly what
    we are trying to measure. Events are recorded asynchronously and only read back
    once they report ready via `query()`.
  * Only meaningful while the draft decode runs EAGER. With VLLM_GFX1X_SPEC_CUDAGRAPH
    = prefill (the production setting) patch 65 gives the decode cudagraph manager
    NONE, so `_generate_draft` is called per step and events record normally. Under
    SPEC_CG=1 the steps run inside a captured graph and `run_fullgraph` is used
    instead -- `_generate_draft` is then not called at all and this patch reports
    nothing, which is the honest outcome rather than a wrong number.

Gate: VLLM_GFX1X_DRAFT_TIMING=0 (default) / N>0 = log every N draft steps.
Off by default: this is a diagnostic, not a production feature.

Usage: python3 74_draft_phase_timing.py --src <site-packages> [--check]
Exit codes: 0 applied/ok, 1 check failed, 42 anchor moved -> re-audit.
Written against vLLM v0.29.0rc1 (33898f832c).
"""
import argparse
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 74_draft_phase_timing"
REL = "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"

# --- 1. the timer helper, inserted before the class that uses it ---------------
OLD_HEAD = '''    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._prepare_eplb_forward(num_reqs)
'''

NEW_HEAD = '''    def _gfx_draft_timer(self):
        """''' + MARKER + '''

        Lazily build the phase timer. Returns None when the gate is off, so the
        hot path costs one attribute lookup in that case.
        """
        t = getattr(self, "_gfx_timer", None)
        if t is not None:
            return t if t is not False else None
        import os as _os

        try:
            every = int(_os.environ.get("VLLM_GFX1X_DRAFT_TIMING", "0"))
        except ValueError:
            every = 0
        if every <= 0:
            self._gfx_timer = False
            return None

        class _PhaseTimer:
            """Records HIP events per phase and reads them back without blocking.

            elapsed_time() would synchronise, so completed pairs are harvested only
            once query() says both ends are done. Pending pairs simply carry over to
            the next call.
            """

            def __init__(self, every_n):
                self.every_n = every_n
                self.pending = []          # (phase, start_event, end_event)
                self.acc = {}              # phase -> [total_ms, count]
                self.steps = 0

            def begin(self, phase):
                import torch as _t

                e = _t.cuda.Event(enable_timing=True)
                e.record()
                return (phase, e)

            def end(self, token):
                import torch as _t

                phase, start = token
                e = _t.cuda.Event(enable_timing=True)
                e.record()
                self.pending.append((phase, start, e))

            def harvest(self):
                still = []
                for phase, s, e in self.pending:
                    if e.query() and s.query():
                        slot = self.acc.setdefault(phase, [0.0, 0])
                        slot[0] += s.elapsed_time(e)
                        slot[1] += 1
                    else:
                        still.append((phase, s, e))
                self.pending = still

            def tick(self, logger_):
                self.steps += 1
                self.harvest()
                if self.steps % self.every_n:
                    return
                if not self.acc:
                    return
                total = sum(v[0] for v in self.acc.values())
                parts = []
                for phase, (ms, n) in sorted(
                    self.acc.items(), key=lambda kv: -kv[1][0]
                ):
                    share = 100.0 * ms / total if total else 0.0
                    parts.append(f"{phase}={ms / n:.3f}ms({share:.0f}%)")
                logger_.info(
                    "''' + MARKER + '''| draft steps=%d | per step: %s | sum=%.3fms",
                    self.steps,
                    " ".join(parts),
                    total / max(1, min(v[1] for v in self.acc.values())),
                )
                self.acc = {}

        self._gfx_timer = _PhaseTimer(every)
        logger.info(
            "''' + MARKER + ''': draft phase timing on, logging every %d steps. "
            "Only valid while the draft decode runs eager "
            "(VLLM_GFX1X_SPEC_CUDAGRAPH=prefill or 0).",
            every,
        )
        return self._gfx_timer

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._prepare_eplb_forward(num_reqs)
        _gfx_t = self._gfx_draft_timer()
        _gfx_tok = _gfx_t.begin("forward") if _gfx_t is not None else None
'''

# --- 2. close the forward phase, open sampling --------------------------------
OLD_SAMPLE = '''        # Sample the draft tokens.
        sample_hidden_states = last_hidden_states[:num_reqs]
        sample_src_positions = self.sample_src_positions[:num_reqs]
        draft_tokens = self.sample_draft(
'''
NEW_SAMPLE = '''        if _gfx_t is not None:
            _gfx_t.end(_gfx_tok)
            _gfx_tok = _gfx_t.begin("sample")

        # Sample the draft tokens.
        sample_hidden_states = last_hidden_states[:num_reqs]
        sample_src_positions = self.sample_src_positions[:num_reqs]
        draft_tokens = self.sample_draft(
'''

# --- 3. close sampling, open the input update, close it at the end ------------
OLD_UPDATE = '''        # Update the inputs for the next step.
        update_draft_inputs(
            draft_tokens,
            self.current_draft_step,
            hidden_states,
            self.draft_tokens,
            self.hidden_states,
            self.input_buffers,
            self.sample_src_positions,
            num_reqs,
            self.max_model_len,
            self.num_speculative_steps,
            advance_draft_positions=self.advance_draft_positions,
        )
'''
NEW_UPDATE = '''        if _gfx_t is not None:
            _gfx_t.end(_gfx_tok)
            _gfx_tok = _gfx_t.begin("update_inputs")

        # Update the inputs for the next step.
        update_draft_inputs(
            draft_tokens,
            self.current_draft_step,
            hidden_states,
            self.draft_tokens,
            self.hidden_states,
            self.input_buffers,
            self.sample_src_positions,
            num_reqs,
            self.max_model_len,
            self.num_speculative_steps,
            advance_draft_positions=self.advance_draft_positions,
        )
        if _gfx_t is not None:
            _gfx_t.end(_gfx_tok)
            _gfx_t.tick(logger)
'''

STEPS = (
    ("timer helper + forward begin", OLD_HEAD, NEW_HEAD),
    ("sample phase", OLD_SAMPLE, NEW_SAMPLE),
    ("update phase", OLD_UPDATE, NEW_UPDATE),
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
        print(("OK: patch 74 present in " if ok else "FAIL: patch 74 marker not found in ") + str(f))
        return 0 if ok else 1
    if MARKER in s:
        print(f"SKIP: patch 74 already applied to {f}")
        return 0
    if "logger" not in s:
        print(f"ERROR: {f} has no module logger; re-audit patch 74", file=sys.stderr)
        return 42
    for name, old, _ in STEPS:
        if s.count(old) != 1:
            print(
                f"ERROR: {name} anchor found {s.count(old)} times in {f}; re-audit patch 74",
                file=sys.stderr,
            )
            return 42
    for _, old, new in STEPS:
        s = s.replace(old, new, 1)
    f.write_text(s)
    print(f"OK: patch 74 applied to {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
