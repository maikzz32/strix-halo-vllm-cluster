#!/usr/bin/env bash
# warmup.sh <model> <profile> [--force] — compile-cache warmup + snapshot.
#
# Runs scripts/serve.sh once on the head node (node1) so every Triton/Inductor
# kernel for this (model, profile) is JIT-compiled exactly the way production
# serving compiles it, sends one short generation request, tears the server
# down and snapshots the compile caches into:
#
#   $WARM_CACHE_ROOT/<model>/<profile>/
#     triton/       content of the 'triton-cache' volume (TRITON_CACHE_DIR)
#     vllm_cache/   content of /root/.cache/vllm (torch_compile_cache, ...)
#     fingerprint   image + registry env/args hash; skip-if-fresh marker
#
# Why serve.sh is run UNMODIFIED (deliberately no --max-model-len shrink):
# vLLM's torch_compile_cache hash covers the whole config (max_model_len
# included) — warming with a different config would silently fork the hash and
# the snapshot would never hit in production. The Triton cache keys include
# the GPU arch, so one gfx1151 snapshot hits on every identical gfx1151 node
# (same triton/torch/env); dist_cache.sh distributes it to node2..4.
#
# Measured precedent: cold start 294 s -> 82 s with warm caches. Setting
# TRITON_STORE_BINARY_ONLY=1 in the image env shrinks the Triton cache ~77 %.
#
# Prerequisites: cluster_up.sh has run (container $CONTAINER_NAME exists).
# The containers run rootless as the login user, so the volume mountpoint is
# directly readable; for rootful setups set VLLM_HOST_TRITON_CACHE explicitly.
#
# Idempotent: skips the run when the stored fingerprint matches the current
# image + registry env/args, unless --force is given.
#
# Usage: warmup.sh <model> <profile> [--force]
# Env:   WARM_CACHE_ROOT (default /var/lib/vllm/warm-cache),
#        VLLM_HOST_TRITON_CACHE (host path override for the Triton cache;
#          default: resolved via `podman volume inspect triton-cache`),
#        CONTAINER_NAME (ray-head), VOLUME_NAME (triton-cache),
#        PORT (8000), HEALTH_TIMEOUT_S (1800), SSH_OPTS, VLLM_IMAGE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REG="$SCRIPT_DIR/lib/registry.py"
PY="${PYTHON:-python3}"
INVENTORY="${INVENTORY:-$REPO_ROOT/ansible/inventory.yaml}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=5}"

usage() { sed -n '2,33p' "$0"; exit "${1:-1}"; }

FORCE=0
case "${3:-}" in
  ""|--force) [ "${3:-}" = "--force" ] && FORCE=1 ;;
  *) usage ;;
esac
[ $# -ge 2 ] || usage
MODEL="$1"
PROFILE="$2"

IMAGE="${VLLM_IMAGE:-ghcr.io/maikzz32/strix-vllm-gfx1151:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-ray-head}"
VOLUME_NAME="${VOLUME_NAME:-triton-cache}"
PORT="${PORT:-8000}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"  # cold JIT precedent: ~294 s; be generous
WARM_CACHE_ROOT="${WARM_CACHE_ROOT:-/var/lib/vllm/warm-cache}"
SNAP="$WARM_CACHE_ROOT/$MODEL/$PROFILE"
BASE_URL="http://127.0.0.1:$PORT"

# --- fingerprint (skip-if-fresh) ---------------------------------------------
# Covers everything that keys the compile caches: image (triton/torch/vllm
# versions) + merged registry env + extra CLI args + model + profile.
FP="$({
  echo "image=$IMAGE"
  echo "model=$MODEL"
  echo "profile=$PROFILE"
  "$PY" "$REG" hf_repo "$MODEL"
  "$PY" "$REG" env "$MODEL"
  "$PY" "$REG" extra_args "$MODEL"
} | sha256sum | awk '{print $1}')"

if [ "$FORCE" -eq 0 ] && [ -f "$SNAP/fingerprint" ] && [ "$(cat "$SNAP/fingerprint")" = "$FP" ]; then
  echo "warmup: snapshot $SNAP is fresh (fingerprint match) — skipping (use --force to redo)"
  exit 0
fi

if ! podman container exists "$CONTAINER_NAME"; then
  echo "warmup: container '$CONTAINER_NAME' not found — run scripts/cluster_up.sh first" >&2
  exit 1
fi
mkdir -p "$SNAP"

# --- teardown (trap) -----------------------------------------------------------
SERVE_PID=""
teardown() {
  [ -z "$SERVE_PID" ] || { kill "$SERVE_PID" 2>/dev/null || true; wait "$SERVE_PID" 2>/dev/null || true; }
  pkill -f 'vllm serve' 2>/dev/null || true
  # Workers only: registry.py lists the head (node1) first, so skip line 1.
  local host user
  while read -r _ host user; do
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "${user:+$user@}$host" 'pkill -f "vllm serve" || true' 2>/dev/null || true
  done < <("$PY" "$REG" nodes "$INVENTORY" | tail -n +2)
}
trap teardown EXIT

# --- serve once (registry config verbatim — see header why) --------------------
SERVE_LOG="$SNAP/serve.log"
echo "warmup: starting serve.sh $MODEL $PROFILE (log: $SERVE_LOG)"
PORT="$PORT" bash "$SCRIPT_DIR/serve.sh" "$MODEL" "$PROFILE" >"$SERVE_LOG" 2>&1 &
SERVE_PID=$!

echo "warmup: waiting for /health (up to ${HEALTH_TIMEOUT_S}s, cold JIT compiles everything)"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "warmup: serve.sh exited early — see $SERVE_LOG" >&2
    exit 1
  fi
  if curl -sf -m 5 "$BASE_URL/health" >/dev/null 2>&1; then break; fi
  sleep 5
done
if ! curl -sf -m 5 "$BASE_URL/health" >/dev/null 2>&1; then
  echo "warmup: server not healthy after ${HEALTH_TIMEOUT_S}s — see $SERVE_LOG" >&2
  exit 1
fi

HF_REPO="$("$PY" "$REG" hf_repo "$MODEL")"
echo "warmup: healthy — sending one short generation to exercise decode kernels"
curl -sf -m 180 "$BASE_URL/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$HF_REPO\",\"prompt\":\"Hello,\",\"max_tokens\":8,\"temperature\":0}" \
  >/dev/null

echo "warmup: tearing down the server"
teardown
SERVE_PID=""
trap - EXIT

# --- snapshot -------------------------------------------------------------------
# Triton cache: host mountpoint of the named volume created by cluster_up.sh.
TRITON_SRC="${VLLM_HOST_TRITON_CACHE:-}"
if [ -z "$TRITON_SRC" ]; then
  TRITON_SRC="$(podman volume inspect "$VOLUME_NAME" --format '{{.Mountpoint}}' 2>/dev/null || true)"
fi
if [ -z "$TRITON_SRC" ] || [ ! -d "$TRITON_SRC" ]; then
  echo "warmup: cannot locate Triton cache (volume '$VOLUME_NAME')." >&2
  echo "  Set VLLM_HOST_TRITON_CACHE to the host path of TRITON_CACHE_DIR." >&2
  exit 1
fi
echo "warmup: snapshotting Triton cache from $TRITON_SRC"
rm -rf "$SNAP/triton"
mkdir -p "$SNAP/triton"
tar -C "$TRITON_SRC" -cf - . | tar -C "$SNAP/triton" -xf -

# vLLM cache (torch_compile_cache etc.): lives in the container layer, pull it
# out via podman cp. Ephemeral — see docs/RUNBOOK.md entry on hash forking.
if podman exec "$CONTAINER_NAME" test -d /root/.cache/vllm 2>/dev/null; then
  echo "warmup: snapshotting /root/.cache/vllm from $CONTAINER_NAME"
  rm -rf "$SNAP/vllm_cache"
  mkdir -p "$SNAP/vllm_cache"
  podman cp "$CONTAINER_NAME:/root/.cache/vllm/." "$SNAP/vllm_cache/" \
    || echo "warmup: WARNING: podman cp of /root/.cache/vllm failed (non-fatal)" >&2
fi

# fingerprint last: it is the completeness marker dist_cache.sh checks for.
echo "$FP" > "$SNAP/fingerprint"
echo "warmup: snapshot complete -> $SNAP"
echo "warmup: next step: scripts/dist_cache.sh $MODEL $PROFILE   (push to node2..4)"
