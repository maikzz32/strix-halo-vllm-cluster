#!/usr/bin/env bash
# dist_cache.sh <model> <profile> [inventory.yaml] — distribute a warmup
# snapshot to node2..4.
#
# Takes the snapshot written by scripts/warmup.sh
# ($WARM_CACHE_ROOT/<model>/<profile>/) and pushes it to every non-head node:
#
#   triton/     -> mountpoint of the node's 'triton-cache' volume (the same
#                  location cluster_up.sh bind-mounts as TRITON_CACHE_DIR;
#                  resolved per node via `podman volume inspect`, created if
#                  missing). The Triton cache is arch-keyed (gfx1151), so one
#                  snapshot hits on all identical gfx1151 nodes.
#   vllm_cache/ -> best-effort `podman cp` into the node's running runtime
#                  container (/root/.cache/vllm); skipped with a warning when
#                  no container is up (the directory lives in the container
#                  layer — there is no host path to push to).
#
# Transfer: rsync-over-ssh, fallback scp. Every node is verified afterwards
# with a sha256 manifest (sha256sum -c on the target). Per-node status is
# printed; exit code is non-zero if any node failed.
#
# Usage: dist_cache.sh <model> <profile> [inventory.yaml]
# Env:   WARM_CACHE_ROOT (default /var/lib/vllm/warm-cache),
#        VOLUME_NAME (triton-cache), SSH_OPTS (extra ssh options)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="${PYTHON:-python3}"

usage() { sed -n '2,24p' "$0"; exit "${1:-1}"; }

[ $# -ge 2 ] && [ $# -le 3 ] || usage
MODEL="$1"
PROFILE="$2"
INVENTORY="${3:-$REPO_ROOT/ansible/inventory.yaml}"

WARM_CACHE_ROOT="${WARM_CACHE_ROOT:-/var/lib/vllm/warm-cache}"
VOLUME_NAME="${VOLUME_NAME:-triton-cache}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=5}"
SNAP="$WARM_CACHE_ROOT/$MODEL/$PROFILE"

if [ ! -f "$SNAP/fingerprint" ] || [ ! -d "$SNAP/triton" ]; then
  echo "dist_cache: no complete snapshot at $SNAP" >&2
  echo "  (needs fingerprint + triton/ — run scripts/warmup.sh $MODEL $PROFILE first)" >&2
  exit 1
fi

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

# sha256 manifest of the snapshot's Triton tree, checked on each target node.
# *.autotune.json is excluded: Triton rewrites those per node at runtime with
# the node's own autotune results, so they legitimately differ (node3 failed
# the check on chunk_gated_delta_rule_fwd_kernel_h_blockdim64.autotune.json,
# 2026-09-03). They are still copied, just not verified.
MANIFEST="$SNAP/triton.sha256"
( cd "$SNAP/triton" && find . -type f ! -name '*.autotune.json' -print0 | sort -z | xargs -0 sha256sum ) > "$MANIFEST"
NFILES=$(wc -l < "$MANIFEST")
echo "dist_cache: snapshot $SNAP ($NFILES files, fingerprint $(cut -c1-12 "$SNAP/fingerprint")...)"

FAILED=()
for i in "${!NAMES[@]}"; do
  [ "$i" -eq 0 ] && continue   # head (node1) is the source
  name="${NAMES[$i]}"
  echo "dist_cache: --- $name ---"

  # Resolve (and if necessary create) the node's cache volume mountpoint.
  if ! MP="$(ssh_node "$i" "podman volume inspect '$VOLUME_NAME' --format '{{.Mountpoint}}' 2>/dev/null || \
      { podman volume create '$VOLUME_NAME' >/dev/null && podman volume inspect '$VOLUME_NAME' --format '{{.Mountpoint}}'; }")" \
      || [ -z "$MP" ]; then
    echo "dist_cache: $name FAIL: cannot resolve volume '$VOLUME_NAME' mountpoint" >&2
    FAILED+=("$name")
    continue
  fi
  echo "dist_cache: $name target: $MP"

  # Transfer: rsync when available on both ends, scp otherwise.
  if command -v rsync >/dev/null 2>&1 && ssh_node "$i" 'command -v rsync >/dev/null 2>&1'; then
    # shellcheck disable=SC2086
    if ! rsync -az --delete -e "ssh $SSH_OPTS" "$SNAP/triton/" "${TARGETS[$i]}:$MP/"; then
      echo "dist_cache: $name FAIL: rsync failed" >&2
      FAILED+=("$name")
      continue
    fi
  else
    echo "dist_cache: $name: rsync unavailable, falling back to scp"
    # shellcheck disable=SC2086
    if ! ssh_node "$i" "mkdir -p '$MP' && find '$MP' -mindepth 1 -delete" \
       || ! scp -r $SSH_OPTS -q "$SNAP/triton/." "${TARGETS[$i]}:$MP/"; then
      echo "dist_cache: $name FAIL: scp failed" >&2
      FAILED+=("$name")
      continue
    fi
  fi

  # Verify: checksums on the target.
  # shellcheck disable=SC2086
  if ! scp $SSH_OPTS -q "$MANIFEST" "${TARGETS[$i]}:$MP/.warm-cache-manifest.sha256" \
     || ! ssh_node "$i" "cd '$MP' && sha256sum -c --quiet .warm-cache-manifest.sha256"; then
    echo "dist_cache: $name FAIL: checksum verification failed" >&2
    FAILED+=("$name")
    continue
  fi

  # Best-effort: vllm_cache into the running container (non-fatal).
  if [ -d "$SNAP/vllm_cache" ]; then
    CNAME="$(ssh_node "$i" "podman ps --format '{{.Names}}' | grep -E '^(ray-worker|ray-head|strix-vllm)$' | head -1" || true)"
    if [ -n "$CNAME" ]; then
      if tar -C "$SNAP/vllm_cache" -cf - . | ssh_node "$i" "podman cp - '$CNAME':/root/.cache/vllm" 2>/dev/null; then
        echo "dist_cache: $name: vllm_cache -> container $CNAME"
      else
        echo "dist_cache: $name: WARNING: vllm_cache podman cp failed (non-fatal)" >&2
      fi
    else
      echo "dist_cache: $name: no running container — vllm_cache skipped (container-layer path)" >&2
    fi
  fi

  echo "dist_cache: $name OK ($NFILES files verified)"
done

echo
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "dist_cache: FAILED nodes: ${FAILED[*]}" >&2
  exit 1
fi
echo "dist_cache: all $(( ${#NAMES[@]} - 1 )) worker nodes updated and verified"
