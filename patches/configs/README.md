# Fused-MoE tuned Triton configs for gfx1151 (AMD Radeon 8060S)

This folder is baked into the image at `/opt/vllm-tuned-configs` and exported
as `VLLM_TUNED_CONFIG_FOLDER` (see `docker/Dockerfile.fedora`). vLLM's fused
MoE layer checks `$VLLM_TUNED_CONFIG_FOLDER` **before** the configs bundled in
the wheel (`vllm/model_executor/layers/fused_moe/fused_moe.py::get_moe_configs`),
so dropping a file here overrides the stock heuristics without any source
patching.

## File name scheme

```
E={num_experts},N={shard_intermediate_size},device_name={device}[,dtype=...][,block_shape=...].json
```

- `N` is `w2.shape[2]`, i.e. `moe_intermediate_size // TP` (the per-expert
  intermediate size after `silu_and_mul`). The files here assume **TP=1**;
  after tuning at other TP values the tuner writes the correct name itself.
- `device_name` is `torch.cuda.get_device_name(0)` with spaces/slashes
  replaced by `_`. We assume `AMD_Radeon_8060S` for Strix Halo — **confirm on
  hardware** (APU memory/device reporting is buggy upstream, ROCm/hip#3892):

  ```sh
  python3 -c "import torch; print(torch.cuda.get_device_name(0).replace(' ', '_'))"
  ```

  If it prints something else, rename the files accordingly (or let
  `bench/tune_moe.py` write them — it runs on the hardware and gets the name
  right automatically).

## Config format

Keys are M-bucket upper bounds (`"1"`, `"2"`, ..., `"4096"`), values are
Triton configs (`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`,
`num_stages`). The optional top-level `triton_version` key is ignored by the
loader; we abuse it to mark untuned placeholders. CDNA-only keys
(`matrix_instr_nonkdim`, `kpack`) from the MI300X configs shipped with vLLM
are deliberately omitted — they do not exist on RDNA.

## Status: PLACEHOLDERS — do not ship benchmarks against these

All three files are **untuned placeholders** with RDNA-sane starting values
(all dims multiples of 16; `BLOCK_SIZE_M` 16/32/64 by M-bucket,
`BLOCK_SIZE_N=64`, `BLOCK_SIZE_K=128`, `GROUP_SIZE_M` 1/8, `num_warps=4`,
`num_stages=2`). They exist so the folder mechanism is exercised end-to-end.
Replace them with `bench/tune_moe.py` output before judging performance.

| File | Model (`models/registry.yaml`) | E | N |
|---|---|---|---|
| `E=512,N=2560,device_name=AMD_Radeon_8060S.json` | qwen38-flash-next | 512 | 2560 |
| `E=288,N=4096,device_name=AMD_Radeon_8060S.json` | glm53-flash | 288 | 4096 |
| `E=256,N=512,device_name=AMD_Radeon_8060S.json` | qwen36-35b-a3b | 256 | 512 |

E/N for qwen36-35b-a3b come from its HF `config.json` (`num_experts=256`,
`moe_intermediate_size=512`).

---

## Tuning auf der Hardware (Deutsch)

Die Platzhalter-JSONs durch echte Tuning-Ergebnisse ersetzen:

```sh
# Im Repo-Root auf einem gfx1151-Knoten (Container-Image muss gebaut sein):
python3 bench/tune_moe.py --model Qwen/Qwen3.6-35B-A3B \
    --image ghcr.io/<org>/vllm-gfx1151:dev --tp-size 1 \
    --vllm-src /pfad/zum/vllm-checkout

# Ergebnis landet direkt in patches/configs/ (per Mount) und wird beim
# naechsten Image-Build mitgebacken. Fuer TP>1 den Tuner je TP-Stufe
# wiederholen; der Dateiname enthaelt dann N = moe_intermediate_size / TP.
```

Details und alle Optionen: `python3 bench/tune_moe.py --help`.
