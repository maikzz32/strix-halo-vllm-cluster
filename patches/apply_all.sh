#!/usr/bin/env bash
# apply_all.sh — apply every gfx1151 patch in order, then run verify_compat.py.
#
# Fail-closed: any patch failure (including exit 42 = "upstream moved,
# re-audit needed") or a failed compatibility check aborts with non-zero.
# Idempotent: already-applied patches report SKIP and succeed.
#
# Usage:
#   patches/apply_all.sh                        # apply all patches, then verify
#   patches/apply_all.sh --check                # check-only mode for everything
#   patches/apply_all.sh --skip-verify          # apply only (builder stage runs
#                                               # verify_compat separately with
#                                               # --skip-imports, because vLLM is
#                                               # not pip-installed yet there)
#
# Environment:
#   VLLM_SRC   path to the vLLM source checkout (default: /opt/vllm).
#              Must match the checkout location used in docker/
#              (Dockerfile.fedora sets VLLM_SRC=/opt/src/vllm explicitly).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_SRC="${VLLM_SRC:-/opt/vllm}"

CHECK_ARGS=()
SKIP_VERIFY=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ARGS=(--check) ;;
        --skip-verify) SKIP_VERIFY=1 ;;
        *) echo "ERROR: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

shopt -s nullglob
patches=("$HERE"/[0-9]*_*.py)
if [[ ${#patches[@]} -eq 0 ]]; then
    echo "ERROR: no patch scripts (NN_name.py) found in $HERE" >&2
    exit 1
fi

for patch in "${patches[@]}"; do
    echo "==> $(basename "$patch")"
    python3 "$patch" --src "$VLLM_SRC" ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"}
done

echo "==> verify_compat.py"
if [[ $SKIP_VERIFY -eq 1 ]]; then
    echo "(skipped: --skip-verify)"
else
    python3 "$HERE/verify_compat.py" --src "$VLLM_SRC"
fi

echo "All patches applied and compatibility check passed."
