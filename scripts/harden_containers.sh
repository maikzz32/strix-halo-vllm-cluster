#!/usr/bin/env bash
# harden_containers.sh — idempotent in-container fixes after cluster_up.sh.
#
# Every freshly created ray-head/ray-worker container from the gfx1151 image
# needs two fixes (both are lost on container recreation; the image itself
# stays stock):
#
#   1. gcc + gcc-c++ — the Triton JIT compiles kernels at serve time; without
#      a compiler the first `vllm serve` dies in the compile phase.
#   2. remove amd-aiter/aiter — its JIT crashes already on IMPORT on gfx1x
#      (VLLM_ROCM_USE_AITER=0 from the registry gates usage, not the import).
#
# Requires the containers to be RUNNING (works via podman exec); containers
# that are absent or stopped are skipped with a note. Safe to re-run:
# packages are only installed/removed when missing/present.
# cluster_up.sh calls this automatically at the end (SKIP_HARDEN=1 disables).
#
# Usage: harden_containers.sh [inventory.yaml]
# Env:   SSH_OPTS (extra ssh options)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="${PYTHON:-python3}"
INVENTORY="${1:-$REPO_ROOT/ansible/inventory.yaml}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=5}"

NAMES=(); TARGETS=()
while read -r name host user; do
  NAMES+=("$name")
  if [ -n "$user" ]; then TARGETS+=("$user@$host"); else TARGETS+=("$host"); fi
done < <("$PY" "$SCRIPT_DIR/lib/registry.py" nodes "$INVENTORY" | tr -d '\r')

ssh_node() { # ssh_node <idx> <cmd...>
  local idx="$1"; shift
  # shellcheck disable=SC2086
  ssh $SSH_OPTS "${TARGETS[$idx]}" "$@"
}

harden() { # harden <idx> <container>
  local idx="$1" cname="$2"
  if ! ssh_node "$idx" "podman ps --format '{{.Names}}' | grep -qx '$cname'" 2>/dev/null; then
    echo "harden: $cname on ${NAMES[$idx]} not running — skipping (cluster_up.sh first?)"
    return 0
  fi
  echo "harden: $cname on ${NAMES[$idx]}"
  # gcc/gcc-c++ for the Triton JIT (rpm check = idempotent).
  ssh_node "$idx" "podman exec '$cname' bash -c \
    'rpm -q gcc gcc-c++ >/dev/null 2>&1 || dnf install -y -q gcc gcc-c++'"
  # amd-aiter/aiter import crash on gfx1x — remove only when present.
  ssh_node "$idx" "podman exec '$cname' bash -c \
    'pip3 list --format=freeze 2>/dev/null | grep -qiE \"^(amd-)?aiter==\" \
     && pip3 uninstall -y amd-aiter aiter || true'"
  # Verify: compiler present, aiter gone.
  if ssh_node "$idx" "podman exec '$cname' bash -c \
    'command -v gcc >/dev/null && command -v g++ >/dev/null \
     && ! pip3 list --format=freeze 2>/dev/null | grep -qiE \"^(amd-)?aiter==\"'"; then
    echo "harden: $cname on ${NAMES[$idx]} OK"
  else
    echo "harden: $cname on ${NAMES[$idx]} FAILED verification" >&2
    return 1
  fi
}

FAILED=0
for i in "${!NAMES[@]}"; do
  if [ "$i" -eq 0 ]; then CNAME=ray-head; else CNAME=ray-worker; fi
  harden "$i" "$CNAME" || FAILED=1
done

if [ "$FAILED" -ne 0 ]; then
  echo "harden: at least one node failed — Triton JIT / aiter import will break serving there" >&2
  exit 1
fi
echo "harden: done"
