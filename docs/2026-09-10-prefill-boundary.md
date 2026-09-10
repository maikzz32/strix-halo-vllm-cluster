# Prefill boundary investigation

The installed scheduler source (SHA256
`abca7134821e2fb5cc8572df5c4a0b570ecf702646254327bc786fc452074983`)
contains `_mamba_block_aligned_split`: it stops prefill at reusable state
boundaries, with extra rules for speculative decoding and partial prefix hits.
`_reserve_prefill_lookahead` additionally protects the multi-module MTP state
from invalid trailing draft inputs. These guards cannot simply be removed.

A live boundary probe used unique exact-length prompts of 504, 511, 512, 513,
800, 801 and 1024 tokens, two requests each, producing exactly one token.
All requests had zero prefix-cache hits and matching repeat output hashes.
Observed TTFT ranged from approximately 604–630 ms around 504–513 tokens,
850–883 ms around 800 tokens, and 1023 ms at 1024 tokens. This does not identify
an extra scheduler step or a removable boundary.

The probe also collected `iteration_tokens_total_count` and `_sum`. Each
request advanced count by one and sum by prompt length plus one. **This is not
proof of one model forward.** In the installed AsyncLLM output handler,
IterationStats is created only when EngineCoreOutputs contains outputs;
the logger observes computed prompt tokens plus generated tokens from that
object. Outputless prefill execution is therefore not independently counted
by this measurement. GPU scopes remain stronger evidence for the observed
504+8 execution pattern.

The precise scheduler/runtime condition producing that split remains
unproven. In particular, the actual cache block sizes and state checkpoint
conditions must be established before changing its behavior. No cache mode,
scheduler or model setting was changed by this probe.

Evidence: `bench/records/2026-09-10-prefill-boundary-probe.json` and the
`--phase-metrics` option of `bench/context_latency.py`. The option now also
collects iteration histogram deltas without asserting that those counts equal
physical forwards. This is diagnostic evidence, not a serving improvement.
