# GPU high mode: small serving improvement

Selecting `high` on all four GPUs improved the matched ShareGPT48/C1 result
from a bracketed control average of 55.488 to 56.364 output tokens/s (+1.58%).
The original mode was restored and verified between the candidate and final
control. After validation, `high` was reapplied to all four live nodes.
The 60-token/s goal remains open.

| ShareGPT48/C1 | Auto before | High | Auto restored |
|---|---:|---:|---:|
| Output tokens/s | 55.569 | 56.364 | 55.407 |
| Mean TTFT, ms | 537.985 | 545.612 | 543.024 |
| Mean TPOT, ms | 15.722 | 15.414 | 15.777 |
| Successful requests | 48 | 48 | 48 |

All 48 texts, output lengths and speculation counts match across the three
phases. Each generated 11839 tokens from 14767 input tokens. Hermes automatic
tool selection, tool-result roundtrip, streamed tool calls and default thinking
passed with high mode, after restoration and after promotion. Mean TTFT did
not improve; this is a modest sustained-generation improvement. A single
bracketed serving trial does not establish a confidence interval.

An earlier six-stream fixed-prompt auto/high/auto trial gave a median paired
decode gain of 2.50%, with exact request, usage, content and reasoning parity.
Its 70–72 decode tokens/s are a different metric and must not be advertised
as the ShareGPT goal result.

## Scope and reproduction

Only `/sys/class/drm/card*/device/power_dpm_force_performance_level` changed.
The script identifies the single AMD GPU by vendor and verifies the resolved
path and mode. Model weights, arithmetic, vLLM configuration, CPU preferences,
manual clock settings and power limits were unchanged. The same running
service and diagnostic graph-inventory preload served all phases:
`f16d955b6e024c02be9a76c92cc787d7`.

The kernel documentation defines `auto` as dynamic power-profile selection
and `high` as the highest clock power state. It also distinguishes this from
`profile_peak`, which was not tested here. [Linux amdgpu documentation](https://docs.kernel.org/gpu/amdgpu/thermal.html).

A single high-mode sensor snapshot confirmed 2.897–2.900 GHz on all ranks.
It is not a frequency-residency or power-efficiency study. Reported APU power
includes CPU/SoC power, and the sampled temperatures were 68–77 C.

From the repository root, with SSH and passwordless sudo already configured:

```sh
python tools/gpu_performance_profile.py read
python tools/gpu_performance_profile.py high --snapshot results/gpu-original.json
python tools/gpu_performance_profile.py restore --snapshot results/gpu-original.json
```

The high command requires all original modes to be auto, saves their values
before modification and rolls back attempted nodes on failure. Restoration
attempts every node even if one fails. The live promotion's saved snapshot is
`bench/records/2026-09-10-gpu-high-production.json`; use it with the restore
command if reverting this particular deployment. Preserve a separate copy
when using it operationally, since restore records its verification in the file.

This is a live sysfs setting, not a boot-persistent service. Recheck it after
host reboot or GPU reset. `tools/bench_gpu_profile.py` reproduces the short
probe and always restores the starting modes. For the full serving comparison,
run the unchanged ShareGPT48/C1 protocol in auto/high/auto order, exporting
each result and its external GPU state. No model restart is needed.

Public records `2026-09-10-gpu-serving-{before,high,after}.json` retain timings
and output hashes; their comparison records validate parity. Original/applied/
restored GPU states are stored alongside them. Raw generated texts remain local.
