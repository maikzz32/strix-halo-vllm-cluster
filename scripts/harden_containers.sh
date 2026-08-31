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
#   3. TRANSITIONAL (E830 nodes only): the current image ships rdma-core/
#      libibverbs 61.0 (fc44), which does not know the Intel E830
#      (PCI 8086:12de) and silently drops its RoCE devices at provider
#      enumeration -> ibv_devices is empty, RCCL cannot use RDMA. Fix in the
#      running container: download rdma-core/libibverbs from the Fedora 45
#      HOST repos (64.0) and `rpm -Uvh --nodeps` them into the container.
#      REMOVE this step once the image carries rdma-core >= 64
#      (docker/Dockerfile.fedora: RDMA_CORE_MIN_MAJOR=64).
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

# TRANSITIONAL E830 rdma-core fix (see header item 3 — remove once the image
# carries rdma-core >= 64). Verified live on node4 by the bench agent.
fix_e830() { # fix_e830 <idx> <container>
  local idx="$1" cname="$2"
  # Only E830 hosts (Intel 8086:12de) are affected; E810 knows its IDs in v61.
  ssh_node "$idx" "lspci -d 8086:12de 2>/dev/null | grep -q ." || return 0
  if ! ssh_node "$idx" "podman ps --format '{{.Names}}' | grep -qx '$cname'" 2>/dev/null; then
    echo "harden: $cname on ${NAMES[$idx]} (E830) not running — rdma-core fix skipped"
    return 0
  fi
  local ver major
  ver="$(ssh_node "$idx" "podman exec '$cname' rpm -q --qf '%{VERSION}' libibverbs" 2>/dev/null || true)"
  major="${ver%%.*}"
  if [ -n "$major" ] && [ "$major" -ge 64 ] 2>/dev/null; then
    echo "harden: $cname on ${NAMES[$idx]} (E830): libibverbs $ver >= 64 — ok"
    return 0
  fi
  echo "harden: $cname on ${NAMES[$idx]} (E830): libibverbs ${ver:-missing} < 64 — installing fc45 rdma-core"
  # Host is Fedora 45 -> dnf download yields rdma-core/libibverbs 64.0.
  ssh_node "$idx" "mkdir -p /tmp/rdma64 && dnf download -y -q rdma-core libibverbs --destdir=/tmp/rdma64"
  ssh_node "$idx" "podman cp /tmp/rdma64/. '$cname':/tmp/rdma64 \
    && podman exec '$cname' bash -c 'rpm -Uvh --nodeps /tmp/rdma64/*.rpm && rm -rf /tmp/rdma64' \
    && rm -rf /tmp/rdma64"
  # Verify: both E830 devices must enumerate now.
  if ssh_node "$idx" "podman exec '$cname' bash -c 'ibv_devices 2>/dev/null | tail -n +3 | grep -q .'"; then
    echo "harden: $cname on ${NAMES[$idx]} (E830): ibv_devices enumerates devices — OK"
  else
    echo "harden: $cname on ${NAMES[$idx]} (E830): ibv_devices still EMPTY after rdma-core fix" >&2
    return 1
  fi
}

FAILED=0
for i in "${!NAMES[@]}"; do
  if [ "$i" -eq 0 ]; then CNAME=ray-head; else CNAME=ray-worker; fi
  harden "$i" "$CNAME" || FAILED=1
  fix_e830 "$i" "$CNAME" || FAILED=1
done

if [ "$FAILED" -ne 0 ]; then
  echo "harden: at least one node failed — serving will break there (gcc / aiter / E830 rdma-core)" >&2
  exit 1
fi
echo "harden: done"
