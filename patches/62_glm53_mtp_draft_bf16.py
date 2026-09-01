#!/usr/bin/env python3
"""Patch 62: build the GLM-5.3 MTP draft layer quant-free (BF16).

MTP speculative decoding for GLM-5.3-Flash loads the draft head
(``Glm5NextMTP``, ``vllm/models/glm5next/nvidia/mtp.py``, PR #53906 branch)
from the SAME checkpoint as the target. ``SpeculativeConfig.__post_init__``
copies the target's quantization into the draft ModelConfig
(``self.quantization = self.target_model_config.quantization``), so every
quant_config consumer inside ``Glm5NextMultiTokenPredictorLayer`` is built
with the compressed-tensors pack-quantized int4 (group 32) method.

That is wrong for the local checkpoint (/home/maik/glm53_flash): the
llm-compressor ignore list covers every layer-45 weight (2334 ignore
entries; zero ``weight_packed`` keys under ``layers.45.`` in the
safetensors index — verified), i.e. the whole MTP layer is BF16. The
checkpoint's ignore names (``model.language_model.layers.45.*``) can never
match the draft's module prefixes (``model.layers.45.mtp_block.*``), so
``should_ignore_layer`` cannot save it (probed: returns False for every
draft-style prefix). Without this patch the draft weight load dies on the
first BF16 expert weight handed to a WNA16-packed param.

quant_config consumers inside the MTP layer (all covered by this patch):
  - ``Glm5NextDecoderLayer`` -> ``Glm5NextMoE`` -> ``FusedMoEFactory``
    (routed experts) AND ``Glm5NextMLP`` shared_experts
    (``n_shared_experts=1``; the checkpoint has BF16
    ``mlp.shared_experts.{gate,up,down}_proj.weight`` for layer 45),
    reached via ``vllm_config.quant_config`` inside the decoder layer,
  - ``SharedHead`` -> ``ParallelLMHead`` (the local ``quant_config``
    variable; the head is always del'd and replaced by the target's BF16
    lm_head in ``llm_base_proposer``, so quant-free is exact there too).

The MLA projections and the sparse indexer are already hard-coded BF16
inside Glm5NextDecoderLayer ("MLA projections are BF16 in checkpoint").

NOTE: like patch 61, this is specific to the local BF16-MTP int4 re-quant.
Do NOT apply to deployments serving the upstream zai-org FP8 checkpoint,
whose MTP-layer experts are quantized and need the inherited method.

SKIP semantics (same as patch 61): no glm5next package in the tree (e.g. a
build off vLLM main without PR #53906) -> SKIP, exit 0. glm5next present
but an anchor moved -> exit 42 (re-audit). The patched file is
ast.parse'd before writing (fail closed). Idempotent via marker.

Usage:
    python3 62_glm53_mtp_draft_bf16.py --src /opt/vllm          # apply
    python3 62_glm53_mtp_draft_bf16.py --src /opt/vllm --check  # verify only

Exit codes: 0 = applied / skipped / check passed, 1 = check failed,
            42 = glm5next present but anchors moved (re-audit needed).
"""

import argparse
import ast
import sys
from pathlib import Path

MARKER = "gfx1151-patch: 62_glm53_mtp_draft_bf16"
EXIT_REAUDIT = 42

REL_PATH = "vllm/models/glm5next/nvidia/mtp.py"

# Anchor 1: module imports (verbatim against the glm-release checkout).
# mtp.py imports typing but not copy.
IMPORT_OLD = "import typing\nfrom collections.abc import Callable, Iterable\n"
IMPORT_NEW = "import copy\nimport typing\nfrom collections.abc import Callable, Iterable\n"

# Anchor 2: the quant_config read at the top of
# Glm5NextMultiTokenPredictorLayer.__init__ (verbatim; exactly 1 match in
# mtp.py — Glm5NextMTP.__init__ reads vllm_config.model_config instead).
QUANT_OLD = """        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
"""
QUANT_NEW = """        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        # {marker}: the local int4 re-quant keeps the MTP layer BF16
        # (per the llm-compressor ignore list), but the draft ModelConfig
        # inherits the target's compressed-tensors int4 method via
        # SpeculativeConfig.__post_init__; the MoE experts and the
        # shared-expert MLP would be built WNA16-packed and crash the draft
        # weight load. Build the whole MTP layer quant-free: the local
        # variable feeds SharedHead (its LM head is always replaced by the
        # target's BF16 lm_head in the proposer), the config copy feeds
        # Glm5NextDecoderLayer (-> MoE experts + shared_experts MLP).
        quant_config = None
        vllm_config = copy.copy(vllm_config)
        vllm_config.quant_config = None
""".replace("{marker}", MARKER)


def glm5_present(src: Path) -> bool:
    """True if the tree carries GLM-5.3 (glm5next) support at all."""
    if (src / "vllm" / "models" / "glm5next").is_dir():
        return True
    models_dir = src / "vllm" / "model_executor" / "models"
    if models_dir.is_dir():
        registry = models_dir / "registry.py"
        if registry.is_file() and "glm5" in registry.read_text(
                errors="ignore").lower():
            return True
    return False


def find_target(src: Path) -> Path | None:
    cand = src / REL_PATH
    if cand.is_file():
        return cand
    matches = sorted(p for p in src.rglob("mtp.py")
                     if "glm5next" in str(p).lower())
    return matches[0] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="/opt/vllm", help="vLLM source checkout root")
    ap.add_argument("--check", action="store_true", help="verify only, do not modify")
    args = ap.parse_args()
    src = Path(args.src)

    if not glm5_present(src):
        print(f"SKIP: no glm5next package under {src} — patch 62 only "
              f"applies to GLM-5.3 builds (PR #53906 branch).")
        return 0

    target = find_target(src)
    if target is None:
        print(f"ERROR: glm5next present but {REL_PATH} not found under "
              f"{src}. Upstream moved the MTP file; re-audit.",
              file=sys.stderr)
        return EXIT_REAUDIT

    text = target.read_text(encoding="utf-8")

    if args.check:
        ok = MARKER in text and "vllm_config.quant_config = None" in text
        print(("OK" if ok else "MISSING") + f": patch 62 in {target}")
        return 0 if ok else 1

    if MARKER in text:
        print(f"SKIP: patch 62 already applied to {target}")
        return 0

    count = text.count(IMPORT_OLD)
    if count != 1:
        print(f"ERROR: import anchor matched {count}x in {target} "
              f"(expected 1); re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT
    count = text.count(QUANT_OLD)
    if count != 1:
        print(f"ERROR: quant_config anchor matched {count}x in {target} "
              f"(expected 1); re-audit this patch.", file=sys.stderr)
        return EXIT_REAUDIT

    patched = text.replace(IMPORT_OLD, IMPORT_NEW, 1)
    patched = patched.replace(QUANT_OLD, QUANT_NEW, 1)

    # Fail-closed: never write a tree that does not parse.
    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"ERROR: patched {target} no longer parses ({e}); re-audit "
              f"this patch.", file=sys.stderr)
        return EXIT_REAUDIT

    target.write_text(patched, encoding="utf-8", newline="\n")
    print(f"OK: patch 62 applied to {target} "
          f"(MTP draft layer forced quant-free / BF16)")

    # Best-effort: the audited tree has exactly one quant_config read in the
    # MTP layer (the local variable in Glm5NextMultiTokenPredictorLayer).
    # `self.quant_config = vllm_config.quant_config` on the Glm5NextMTP
    # wrapper is bookkeeping only and is expected to remain. If further
    # local reads show up, upstream added a quant consumer to the MTP layer.
    leftover = patched.count("\n        quant_config = vllm_config.quant_config\n")
    if leftover:
        print(f"WARN: {leftover} further read(s) of "
              f"'quant_config = vllm_config.quant_config' remain in "
              f"{target}; re-audit whether they also reach the MTP layer.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
