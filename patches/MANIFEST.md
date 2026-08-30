# Patch manifest — gfx1151 patch layer

Every patch is an idempotent Python script with a `--check` mode. Common
contract:

- Target root: the vLLM source checkout, passed via `--src` (default
  `/opt/vllm`, env `VLLM_SRC` in `apply_all.sh`). **Must match the checkout
  path used by the Dockerfile.** Patch 20 is the exception: it targets the
  installed torch package.
- Target files are located defensively (known relative path, then name
  search); never by hardcoded line numbers.
- Exit codes: `0` applied / check passed, `1` check failed,
  `42` = target pattern not found because upstream moved — **re-audit the
  patch, do not silently skip**.
- Each patch leaves a `# gfx1151-patch: NN_name` marker; `verify_compat.py`
  greps for these markers (fail-closed) and checks that `vllm` and `ray`
  import.

These patches were written against researched upstream behaviour but have
**not yet run against a real vLLM/torch checkout**. Re-validate all of them
on the first real image build.

---

## 10_triton_device_ordinal.py

- **Purpose:** Pin Triton's runtime driver to device ordinal 0. In vLLM's
  forked EngineCore subprocess Triton can initialise with an "invalid device
  ordinal" on gfx1151. Safe because each cluster node has exactly one iGPU.
- **Target:** `vllm/triton_utils/importing.py` (inserts the wrapper
  immediately before `if not HAS_TRITON:`; anchor re-audited for vLLM
  v0.28.0, which removed the file's top-level `import triton`).
- **Upstream reference:** ROCm/TheRock#4552
  (<https://github.com/ROCm/TheRock/issues/4552>).
- **Status:** expected-to-need-adjustment. The wrapped internal
  `triton.runtime.driver.active.utils.load_binary` signature varies between
  Triton releases; `torch.cuda.set_device(0)` is the stable part.
- **Re-audit trigger:** exit 42 (anchor `import triton` gone), or Triton
  still raising "invalid device ordinal" after the patch landed.
- **Date:** 2026-08-30.

## 20_rocm_smi_rtld.py

- **Purpose:** Keep the rtld mode of torch's ROCm library preload
  (TheRock `rocm_sdk.initialize_process`, default `RTLD_GLOBAL`) runtime-
  switchable via `VLLM_GFX1X_ROCM_SMI_RTLD` for the unresolved symbol-clash
  A/B. Re-audited for torch 2.13.0+rocm7.14.0.
- **Target:** `_rocm_init.py` of the **installed torch package** (not the
  vLLM checkout; located via `importlib.util.find_spec("torch")`).
- **Upstream reference:** none pinned; re-audit notes should link the torch
  ROCm issue once confirmed on the real build.
- **Status:** re-audited for torch 2.13.0+rocm7.14.0 (TheRock
  `rocm_sdk.initialize_process` loader; pre-TheRock CDLL layouts are
  rejected with exit 42). Applied in BOTH stages: the builder via
  apply_all.sh (patches its own torch for the marker check), and the final
  stage post-install (that stage reinstalls torch pristine).
- **Re-audit trigger:** exit 42 (no `CDLL` + "rocm_smi" line, or multi-line
  call), or the symbol clash persisting after the patch.
- **Date:** 2026-08-30.

## 30_tensorizer_pin.py

- **Purpose:** Relax the `tensorizer` dependency constraint, which does not
  resolve on Python 3.14 (Fedora 44 interpreter).
- **Target:** first file among `requirements/{common,rocm,build}.txt`,
  `requirements.txt`, `pyproject.toml`, `setup.py` (fallback: any
  `requirements/*.txt`) that contains a `tensorizer` requirement line.
- **Upstream reference:** none pinned; check vLLM's requirements on each
  rebase.
- **Verification:** conditional in verify_compat.py - the marker lives in
  `requirements/*.txt`, which ships in the builder's source checkout
  (enforced there) but not in the installed wheel, so the final-stage
  scan legitimately SKIPs it.
- **Status:** expected-to-need-adjustment. Currently relaxes to an unpinned
  `tensorizer` requirement (extras preserved); **replace with an exact pin
  once a known-good version is confirmed on the first py3.14 build**.
- **Re-audit trigger:** exit 42 (no tensorizer requirement found — upstream
  dropped/renamed it, which may mean the patch is obsolete).
- **Date:** 2026-08-30.

## 40_aiter_gfx1x_gating.py

- **Purpose:** (a) ensure the `VLLM_GFX1X_MOE_TUNE` env exists (default `0`;
  enabling gives a measured 11.3 → 21.8 tok/s decode on gfx1151 MoE) and
  (b) force the AITER master toggle `VLLM_ROCM_USE_AITER` off on `gfx1*`
  arch, since AITER sampler/MoE/FP8-linear kernels are unsupported there.
  Attention backend selection is deliberately untouched
  (`ROCM_AITER_UNIFIED_ATTN` is used by qwen36-35b-a3b with the master
  toggle off; see `models/registry.yaml`).
- **Target:** `vllm/envs.py`.
- **Upstream reference:** AITER gfx1x gating in vLLM (env defaults in
  `vllm/envs.py`); the measured MoE speedup comes from the Strix Halo
  community builds (kyuz0).
- **Status:** mixed. The env entry + guard are expected to work if envs.py
  keeps its current lazy `environment_variables` dict layout. **Not
  verified:** where `VLLM_GFX1X_MOE_TUNE` is consumed (fused-MoE layer) if
  upstream lacks it — the apply step greps for a consumer and prints a loud
  WARN if none is found.
- **Re-audit trigger:** exit 42 (`VLLM_ROCM_USE_AITER` or
  `def __getattr__` anchor gone), the consumer WARN, or upstream adopting
  `VLLM_GFX1X_MOE_TUNE` (then the env-entry half becomes a no-op by design).
- **Date:** 2026-08-30.

---

## librccl/

Not a patch script: drop-in `librccl.so` for the `RCCL_IMPL=custom` image
build. See `librccl/README.md`.

---

## Performance patch series (50-58)

The gfx1151 performance patches (MoE tuning, sparse-indexer kernels, W8A8
skinny GEMM, AITER Triton enablement, APU memory reporting, PLE offload,
GLM MTP dispatch) are documented as per-patch fragments in `manifest.d/`
(one file per patch: `manifest.d/<NN>.md`). They follow the same contract
as above (marker, --check, exit 42 on moved anchors); patches 57/58
additionally SKIP cleanly on trees without the respective model package.
Strategy and measured source claims: `../docs/PERFORMANCE.md`.
