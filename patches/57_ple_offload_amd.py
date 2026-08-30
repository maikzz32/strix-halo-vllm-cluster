#!/usr/bin/env python3
"""Patch 57: port VLLM_PLE_CPU_OFFLOAD (PLE pinned-host offload) to the
ROCm/AMD side of Qwen4Exp (Qwen3.8-Flash-Next) on gfx1151.

The upstream PR implements the pinned-host PLE embedding only for NVIDIA
(vllm/models/qwen4_exp/nvidia/ple_layer.py); the AMD side keeps the full
table device-resident. On Strix Halo (gfx1151) "pinned host memory" is
the same unified LPDDR5 the iGPU uses, so the UVA gather runs at full
~256 GB/s with no PCIe hop — a structural advantage over PCIe-bound dGPU
hosts (and it frees device-side KV budget; needs the APU memory-reporting
fix vllm#40963 to budget correctly).

What the patch does (anchor-based, never line numbers):

  1. vllm/models/qwen4_exp/amd/ple_layer.py
     - extra imports (envs, triton shim, UVA helpers, TP all-reduce),
     - adds _lookup_ple_embedding_from_pinned_kernel (verbatim from the
       NVIDIA file; it is platform-generic) and Qwen4ExpPinnedHostEmbedding
       (pinned CPU weight + UVA view + Triton lookup + TP-range masking
       via shard_indices + all-reduce),
     - selects the pinned embedding when VLLM_PLE_CPU_OFFLOAD=1 (lazy envs
       read at model build time, after Ray applied worker env),
     - splits Qwen4ExpNGramEmbedding.forward into compute_ngram_ids() +
       forward() and adds start/launch/wait/finalize prefetch methods that
       call torch.cuda.Stream directly (works through the HIP shim). The
       NVIDIA direct_register_custom_op wrappers for start/wait prefetch
       are deliberately dropped: they exist only to survive torch.compile /
       CUDA-graph capture, and gfx1151 runs --enforce-eager (vllm#32180).
       The qwen4_exp_amd_ple_ngram_embedding custom op is kept untouched
       for the VLLM_PLE_CPU_OFFLOAD=0 path,
     - adds Qwen4ExpPLELayer.start_prefetch().

  2. vllm/models/qwen4_exp/amd/model.py (best effort)
     - wires _start_layer_ple_prefetch into the decoder-layer loop so the
       pinned lookup overlaps the preceding layer. If these anchors moved,
       the patch WARNs and continues: ple_layer.py then falls back to a
       synchronous pinned lookup (correct, just without the overlap).

  3. vllm/envs.py (only if VLLM_PLE_CPU_OFFLOAD is missing there)
     - registers the env knob in the lazy environment_variables dict. The
       Qwen3.8 PR branches already carry it; this is for trees that have
       qwen4_exp but not the env.

SKIP semantics (differs from fail-closed): if the tree has no
vllm/models/qwen4_exp package (stable/main builds), the patch prints a
note and exits 0 — it only applies to Qwen3.8 PR builds. But if qwen4_exp
exists and anchors inside amd/ple_layer.py moved, exit 42 (re-audit).

Upstream references:
  - vllm#53899 (Qwen3.8 model + PLE offload, author build branch
    peakcrosser7/vllm release/qwen38next_offload)
  - PLE offload source: peakcrosser7/vllm branch hhy/ple_offload_uva
    (nvidia/ple_layer.py). Anchors verified against BOTH branches.

STATUS: ported, not validated. Must run against a real Qwen3.8 PR
checkout on gfx1151 hardware.

Usage:
    python3 57_ple_offload_amd.py --src /opt/vllm          # apply
    python3 57_ple_offload_amd.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / skipped / check passed, 1 = check failed / error,
            42 = anchor not found (upstream moved; re-audit needed).
"""

import argparse
import re
import sys
from pathlib import Path

MARKER = "gfx1151-patch: ple-offload-amd"
EXIT_REAUDIT = 42

PKG_REL = "vllm/models/qwen4_exp"
PLE_REL = f"{PKG_REL}/amd/ple_layer.py"
MODEL_REL = f"{PKG_REL}/amd/model.py"
ENVS_REL = "vllm/envs.py"

# ---------------------------------------------------------------------------
# ple_layer.py edits
# ---------------------------------------------------------------------------

PLE_IMPORT_OLD = "from vllm.utils.torch_utils import direct_register_custom_op\n"
PLE_IMPORT_NEW = f"""from vllm.utils.torch_utils import direct_register_custom_op

# {MARKER} — imports for the pinned-host offload path
import vllm.envs as envs

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
"""

PLE_CLASS_ANCHOR = "class Qwen4ExpNGramEmbedding(nn.Module):"
# Triton kernel taken verbatim from nvidia/ple_layer.py (platform-generic).
# The embedding class is a simplified port: no Qwen4ExpPLEEmbeddingMethod /
# FP8-method plumbing (the AMD side never quantizes this table), and the
# weight is re-allocated pinned after the base class built it on-device.
PLE_PINNED_BLOCK = f"""# {MARKER} (upstream: vllm#53899, NVIDIA path in
# peakcrosser7/vllm hhy/ple_offload_uva nvidia/ple_layer.py).
# On gfx1151 the pinned "host" memory is the same unified LPDDR5 the iGPU
# reads, so the UVA gather below runs at full ~256 GB/s instead of
# crossing PCIe as on a dGPU host.
@triton.jit
def _lookup_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    BLOCK_D: tl.constexpr,
):
    \"\"\"Look up TP-owned PLE rows through a UVA view of pinned host memory.\"\"\"
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values, 0.0),
        mask=mask,
    )


class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):
    \"\"\"PLE table loaded into pinned CPU memory and looked up through UVA.\"\"\"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype | None = None,
        padding_size: int,
        prefix: str,
    ) -> None:
        if not is_uva_available():
            raise RuntimeError("VLLM_PLE_CPU_OFFLOAD requires UVA support")
        if not hasattr(torch.ops._C, "get_cuda_view_from_cpu_tensor"):
            raise RuntimeError(
                "torch.ops._C.get_cuda_view_from_cpu_tensor is missing from "
                "the built vllm._C extension; the HIP build does not export "
                "the UVA view op (HIP build gap). Fix the ROCm _C build or "
                "run with VLLM_PLE_CPU_OFFLOAD=0."
            )
        super().__init__(
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
        )
        # Mirror the NVIDIA path: keep the generic quant post-load pass from
        # staging the full table back on the accelerator.
        if hasattr(self, "quant_method"):
            del self.quant_method
        # Re-allocate the weight in pinned host memory (the base class
        # allocated it on the current device). Weight loading copies the
        # checkpoint shards straight into this buffer via
        # copy_ple_embedding_shard_() from ..common.ple.
        device_weight = self.weight
        pinned = torch.empty(
            device_weight.shape,
            dtype=device_weight.dtype,
            device="cpu",
            pin_memory=True,
        )
        weight = nn.Parameter(pinned, requires_grad=False)
        weight.__dict__.update(vars(device_weight))
        self.weight = weight
        self._uva_weight = get_accelerator_view_from_cpu_tensor(self.weight)
        self._block_d = triton.next_power_of_2(self.embedding_dim)

    def lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        \"\"\"Look up local TP rows from pinned memory into a BF16 GPU tensor.\"\"\"
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(
                expected_shape,
                dtype=torch.bfloat16,
                device=input_ids.device,
            )
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != torch.bfloat16
            or output.device != input_ids.device
        ):
            raise ValueError(
                "PLE prefetch output must match the input shape and be BF16 "
                "on the input device"
            )

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            _lookup_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self._uva_weight,
                flat_ids,
                output,
                self.embedding_dim,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                BLOCK_D=self._block_d,
            )
        return output

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        \"\"\"Combine TP-local rows after the pinned-memory lookup.\"\"\"
        if self.tp_size > 1:
            return tensor_model_parallel_all_reduce(output)
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.lookup(input_ids))


{PLE_CLASS_ANCHOR}"""

PLE_CTOR_OLD = """        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
        )
"""
PLE_CTOR_NEW = f"""        # {MARKER}: pick the pinned-host table when offload is enabled.
        # Lazy envs resolution: this runs at model build time in the worker,
        # after Ray applied the worker environment.
        embedding_cls = (
            Qwen4ExpPinnedHostEmbedding
            if envs.VLLM_PLE_CPU_OFFLOAD
            else VocabParallelEmbedding
        )
        self.ngram_embedding = embedding_cls(
            padded_vocab_size,
            self.head_dim,
            padding_size=divisor,
            prefix=f"{{prefix}}.ngram_embedding",
        )
        # {MARKER}: prefetch state for the offload path.
        self._prefetch_enabled = bool(envs.VLLM_PLE_CPU_OFFLOAD)
        self._prefetch_pending = False
        self._prefetch_stream: torch.cuda.Stream | None = None
        self._prefetch_buffer: torch.Tensor | None = None
        if self._prefetch_enabled:
            if not isinstance(self.ngram_embedding, Qwen4ExpPinnedHostEmbedding):
                raise TypeError("PLE CPU offload requires a pinned host embedding")
            self._prefetch_buffer = torch.empty(
                max_total_tokens,
                embedding_dim,
                dtype=torch.bfloat16,
                device=self.ngram_embedding._uva_weight.device,
            )
"""

PLE_FWD_HEAD_OLD = """    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1).long()
"""
PLE_FWD_HEAD_NEW = f"""    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        # {MARKER}
        if self._prefetch_enabled:
            return self._finalize_prefetched(
                input_ids, query_start_loc, ngram_context
            )
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        output = ngram_ids.new_empty(
            (ngram_ids.shape[0], self.embedding_dim),
            dtype=self.ngram_embedding.params_dtype,
        )
        # Kept from the stock AMD path: the custom op keeps the large PLE
        # table out of Inductor's FX graph. Moot under --enforce-eager but
        # harmless, and still used when VLLM_PLE_CPU_OFFLOAD=0.
        torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding(
            ngram_ids,
            output,
            self.layer_name,
        )
        return output

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"Compute n-gram embedding indices for the current request layout.\"\"\"
        input_ids = input_ids.reshape(-1).long()
"""

PLE_FWD_TAIL_OLD = """        ngram_ids = torch.cat(id_blocks, dim=-1)
        output = ngram_ids.new_empty(
            (ngram_ids.shape[0], self.embedding_dim),
            dtype=self.ngram_embedding.params_dtype,
        )
        torch.ops.vllm.qwen4_exp_amd_ple_ngram_embedding(
            ngram_ids,
            output,
            self.layer_name,
        )
        return output
"""
PLE_FWD_TAIL_NEW = """        return torch.cat(id_blocks, dim=-1)
"""

PLE_METHODS_ANCHOR = (
    "        return loaded\n\n\nclass Qwen4ExpPLELayer(nn.Module, MambaBase):"
)
PLE_METHODS_NEW = f"""        return loaded

    # ------------------------------------------------------------------
    # {MARKER} — pinned-host PLE offload machinery.
    # Dropped from the NVIDIA port: the direct_register_custom_op wrappers
    # for start/wait prefetch (qwen4_exp_ple_start_prefetch /
    # qwen4_exp_ple_wait_prefetch). They exist only to survive
    # torch.compile / CUDA-graph capture; gfx1151 runs --enforce-eager
    # (vllm#32180), so the prefetch functions are called directly.
    # ------------------------------------------------------------------

    def start_prefetch(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        \"\"\"Start the pinned lookup while the preceding decoder layer runs.\"\"\"
        if not self._prefetch_enabled:
            return
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        output = self._get_prefetch_output(ngram_ids.shape[0])
        self._launch_prefetch(ngram_ids, output)
        self._prefetch_pending = True

    def _get_prefetch_output(self, num_tokens: int) -> torch.Tensor:
        \"\"\"Return the persistent output slice for the current step.\"\"\"
        assert self._prefetch_buffer is not None
        assert num_tokens <= self._prefetch_buffer.shape[0]
        return self._prefetch_buffer[:num_tokens]

    def _launch_prefetch(
        self,
        ngram_ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        \"\"\"Launch the UVA lookup on a dedicated stream.

        torch.cuda.Stream works on ROCm through the HIP shim.
        \"\"\"
        embedding = self.ngram_embedding
        if not isinstance(embedding, Qwen4ExpPinnedHostEmbedding):
            raise TypeError("PLE prefetch requires a pinned host embedding")
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream()
        stream = self._prefetch_stream
        stream.wait_stream(torch.cuda.current_stream())
        ngram_ids.record_stream(stream)
        with torch.cuda.stream(stream):
            embedding.lookup(
                ngram_ids,
                output=output.unflatten(-1, (self.ngram_heads, -1)),
            )

    def _wait_prefetch(self) -> None:
        \"\"\"Join the PLE side stream to the current compute stream.\"\"\"
        if self._prefetch_stream is not None:
            torch.cuda.current_stream().wait_stream(self._prefetch_stream)

    def _finalize_prefetched(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        \"\"\"Wait for the side stream and return reduced PLE embeddings.\"\"\"
        embedding = self.ngram_embedding
        if not isinstance(embedding, Qwen4ExpPinnedHostEmbedding):
            raise TypeError("PLE CPU offload requires a pinned host embedding")
        if self._prefetch_pending:
            self._wait_prefetch()
            self._prefetch_pending = False
            output = self._get_prefetch_output(input_ids.shape[0])
        else:
            # Prefetch wiring inactive (amd/model.py prefetch calls missing
            # or skipped): run the pinned lookup synchronously on the
            # current stream. Correct, just without the overlap.
            ngram_ids = self.compute_ngram_ids(
                input_ids, query_start_loc, ngram_context
            )
            output = self._get_prefetch_output(ngram_ids.shape[0])
            embedding.lookup(
                ngram_ids, output=output.unflatten(-1, (self.ngram_heads, -1))
            )
        return embedding.reduce(output)


class Qwen4ExpPLELayer(nn.Module, MambaBase):"""

PLE_LAYER_ANCHOR = (
    "        compilation_config.static_forward_context[prefix] = self\n"
)
PLE_LAYER_NEW = f"""        compilation_config.static_forward_context[prefix] = self

    # {MARKER}
    def start_prefetch(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        \"\"\"Start the pinned PLE lookup while the preceding decoder layer runs.\"\"\"
        self.ple_embedding.start_prefetch(input_ids, query_start_loc, ngram_context)
"""

PLE_EDITS = (
    ("imports", PLE_IMPORT_OLD, PLE_IMPORT_NEW),
    ("pinned embedding class", PLE_CLASS_ANCHOR, PLE_PINNED_BLOCK),
    ("ngram_embedding construction", PLE_CTOR_OLD, PLE_CTOR_NEW),
    ("NGramEmbedding.forward head", PLE_FWD_HEAD_OLD, PLE_FWD_HEAD_NEW),
    ("NGramEmbedding.forward tail", PLE_FWD_TAIL_OLD, PLE_FWD_TAIL_NEW),
    ("prefetch methods", PLE_METHODS_ANCHOR, PLE_METHODS_NEW),
    ("PLELayer.start_prefetch", PLE_LAYER_ANCHOR, PLE_LAYER_NEW),
)

# ---------------------------------------------------------------------------
# model.py edits (best effort: WARN + continue on moved anchors)
# ---------------------------------------------------------------------------

MODEL_HELPER_ANCHOR = """    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
"""
MODEL_HELPER_NEW = f"""    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    # {MARKER}
    @staticmethod
    def _start_layer_ple_prefetch(
        layer: nn.Module,
        input_ids: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> None:
        \"\"\"Start a layer's PLE prefetch when the required inputs exist.\"\"\"
        ple: Qwen4ExpPLELayer | None = getattr(layer, "ple", None)
        if ple is None:
            return
        if input_ids is None or query_start_loc is None or ngram_context is None:
            raise RuntimeError("PLE inputs were not prepared")
        ple.start_prefetch(input_ids, query_start_loc, ngram_context)
"""

MODEL_LOOP_PRE_OLD = """        block_output = None
        injection = None
        last_layer = None
        for layer_idx, layer in islice(
"""
MODEL_LOOP_PRE_NEW = f"""        block_output = None
        injection = None
        last_layer = None
        # {MARKER}
        if self.start_layer < self.end_layer:
            self._start_layer_ple_prefetch(
                self.layers[self.start_layer],
                input_ids,
                query_start_loc,
                ngram_context,
            )
        for layer_idx, layer in islice(
"""

MODEL_LOOP_IN_OLD = """            last_layer = layer
            hidden_states, block_output, injection = layer(
"""
MODEL_LOOP_IN_NEW = f"""            last_layer = layer
            # {MARKER}
            if layer_idx + 1 < self.end_layer:
                self._start_layer_ple_prefetch(
                    self.layers[layer_idx + 1],
                    input_ids,
                    query_start_loc,
                    ngram_context,
                )
            hidden_states, block_output, injection = layer(
"""

MODEL_EDITS = (
    ("prefetch helper", MODEL_HELPER_ANCHOR, MODEL_HELPER_NEW),
    ("prefetch before layer loop", MODEL_LOOP_PRE_OLD, MODEL_LOOP_PRE_NEW),
    ("prefetch inside layer loop", MODEL_LOOP_IN_OLD, MODEL_LOOP_IN_NEW),
)

# ---------------------------------------------------------------------------
# envs.py edit (only when the knob is missing entirely)
# ---------------------------------------------------------------------------

ENVS_DICT_RE = re.compile(
    r"^environment_variables:\s*dict\[str,\s*Callable\[\[\],\s*Any\]\]\s*=\s*\{\s*$",
    flags=re.MULTILINE,
)
ENVS_ENTRY = (
    f'    # {MARKER}\n'
    '    "VLLM_PLE_CPU_OFFLOAD": '
    'lambda: os.getenv("VLLM_PLE_CPU_OFFLOAD", "0") == "1",\n'
)


def find_pkg(src: Path) -> Path | None:
    cand = src / PKG_REL
    if cand.is_dir():
        return cand
    matches = sorted(p for p in src.rglob("qwen4_exp") if p.is_dir())
    return matches[0] if matches else None


def find_file(src: Path, rel: str, name: str,
              parent: str | None = None) -> Path | None:
    cand = src / rel
    if cand.is_file():
        return cand
    matches = sorted(
        p for p in src.rglob(name)
        if parent is None or p.parent.name == parent
    )
    return matches[0] if matches else None


def replace_once(content: str, old: str, new: str) -> str | None:
    """Replace exactly one occurrence; None if the anchor is absent/moved."""
    if content.count(old) != 1:
        return None
    return content.replace(old, new, 1)


def apply_edits(target: Path, edits: tuple[tuple[str, str, str], ...],
                hard: bool) -> tuple[str | None, str | None]:
    """Apply anchored edits. Returns (patched_content, error_message)."""
    content = target.read_text()
    for what, old, new in edits:
        patched = replace_once(content, old, new)
        if patched is None:
            kind = "ERROR" if hard else "WARN"
            return None, (f"{kind}: anchor for '{what}' not found (or not "
                          f"unique) in {target}")
        content = patched
    return content, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    pkg = find_pkg(src)
    if pkg is None:
        print(f"SKIP: {PKG_REL} not found under {src}. This patch only "
              f"applies to Qwen3.8 (qwen4_exp) PR builds; stable/main trees "
              f"have nothing to do.")
        return 0

    ple = find_file(src, PLE_REL, "ple_layer.py", parent="amd")
    if ple is None or "qwen4_exp" not in ple.parts:
        print(f"ERROR: qwen4_exp exists but {PLE_REL} was not found under "
              f"{src}. Upstream restructured the package; re-audit this "
              f"patch.", file=sys.stderr)
        return EXIT_REAUDIT

    ple_content = ple.read_text()

    if args.check:
        if MARKER in ple_content:
            print(f"OK: patch 57 present in {ple}")
            return 0
        print(f"FAIL: patch 57 marker not found in {ple}", file=sys.stderr)
        return 1

    if MARKER in ple_content:
        print(f"SKIP: patch 57 already applied to {ple}")
        return 0

    # --- 1. amd/ple_layer.py (hard requirement: exit 42 on moved anchors)
    patched, err = apply_edits(ple, PLE_EDITS, hard=True)
    if patched is None:
        print(f"{err}. Upstream moved the AMD PLE layer; re-audit this "
              f"patch.", file=sys.stderr)
        return EXIT_REAUDIT
    ple.write_text(patched)
    print(f"OK: patch 57 applied to {ple}")

    # --- 2. amd/model.py (best effort; ple_layer falls back to a
    #        synchronous pinned lookup when the wiring is absent)
    model = find_file(src, MODEL_REL, "model.py", parent="amd")
    if model is None or "qwen4_exp" not in model.parts:
        print(f"WARN: {MODEL_REL} not found; PLE offload works but WITHOUT "
              f"prefetch overlap (synchronous fallback). Re-audit the "
              f"model.py wiring.", file=sys.stderr)
    elif MARKER in model.read_text():
        print(f"SKIP: model.py prefetch wiring already present in {model}")
    else:
        patched, err = apply_edits(model, MODEL_EDITS, hard=False)
        if patched is None:
            print(f"{err}. PLE offload works but WITHOUT prefetch overlap "
                  f"(synchronous fallback). Re-audit the model.py wiring.",
                  file=sys.stderr)
        else:
            model.write_text(patched)
            print(f"OK: patch 57 prefetch wiring applied to {model}")

    # --- 3. vllm/envs.py (only if the knob is missing; PR branches have it)
    envs = find_file(src, ENVS_REL, "envs.py")
    if envs is None:
        print(f"ERROR: {ENVS_REL} not found under {src}; cannot verify the "
              f"VLLM_PLE_CPU_OFFLOAD knob. Re-audit this patch.",
              file=sys.stderr)
        return EXIT_REAUDIT
    envs_content = envs.read_text()
    if "VLLM_PLE_CPU_OFFLOAD" in envs_content:
        print(f"OK: VLLM_PLE_CPU_OFFLOAD already registered in {envs}")
    else:
        m = ENVS_DICT_RE.search(envs_content)
        if not m:
            print(f"ERROR: VLLM_PLE_CPU_OFFLOAD missing and the "
                  f"environment_variables dict anchor was not found in "
                  f"{envs}. envs.py was restructured; re-audit this patch.",
                  file=sys.stderr)
            return EXIT_REAUDIT
        envs.write_text(envs_content[:m.end()] + "\n" + ENVS_ENTRY
                        + envs_content[m.end():])
        print(f"OK: VLLM_PLE_CPU_OFFLOAD registered in {envs}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
