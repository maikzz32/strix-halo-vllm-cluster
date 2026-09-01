#!/usr/bin/env bash
# cudagraph_ab.sh [model] [profile] — A/B test on gfx1151:
#   leg A: --enforce-eager
#   leg B: --compilation-config '{"cudagraph_mode":"NONE"}'
#
# STATUS 2026-08-31: A/B ran on qwen38-27b-ablit/tp4 — NONE won (+7 % C=1,
# +9.4 % C=16, 600 s soak clean) and was promoted to the registry default.
# Re-running this script now requires an '--enforce-eager' line in the
# registry to swap from (it errors out otherwise, by design).
#
# Background: HIP graph CAPTURE deadlocks the gfx1151 driver (vllm#32180),
# which is why --enforce-eager is mandatory today. cudagraph_mode NONE drops
# only the capture and KEEPS the Inductor fusion passes (precedent vllm#44988)
# — potentially the same throughput without the deadlock. That is exactly what
# this script measures, in this order (fail-fast):
#
#   1. eager leg:  bench/run_matrix.py --concurrencies 1,16 (baseline numbers)
#   2. none leg:   registry defaults.extra_args is swapped in-place from
#      --enforce-eager to the compilation-config (backup + trap-restore, so
#      the registry is never left modified — rollback is automatic)
#   3. soak:       10 minutes (SOAK_SECONDS) of sustained load with a hang
#      detector: watchdog on /health + dmesg scan for amdgpu resets/ring
#      timeouts. On hang: loud verdict, dmesg excerpt into bench/results/,
#      NONE marked FAILED, script exits 1.
#   4. none leg:   bench/run_matrix.py --concurrencies 1,16 (same cells)
#
# Results: bench/results/<ts>_cudagraph_{eager,none}_*.json + summary JSON.
# Compare with: python3 bench/report.py 'bench/results/*cudagraph_*.json'
#
# NOTE: serve.sh has no extra-args passthrough, so the NONE leg is expressed
# by temporarily editing models/registry.yaml (the single source of truth that
# serve.sh and run_matrix.py both resolve). The file is restored by an EXIT
# trap in every outcome, including Ctrl-C.
#
# Usage: bench/cudagraph_ab.sh [model] [profile]
# Env:   CONCURRENCIES (1,16), PROMPT_LENGTHS (512), SOAK_SECONDS (600),
#        SOAK_CONCURRENCY (4), SOAK_HEALTH_TIMEOUT_S (1800),
#        BENCH_BASE_URL (http://127.0.0.1:8000), SSH_OPTS
set -euo pipefail

MODEL="${1:-qwen36-35b-a3b}"
PROFILE="${2:-tp4}"          # multi-node so RCCL/Ray workers also soak
CONCURRENCIES="${CONCURRENCIES:-1,16}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-512}"
SOAK_SECONDS="${SOAK_SECONDS:-600}"
SOAK_CONCURRENCY="${SOAK_CONCURRENCY:-4}"
SOAK_HEALTH_TIMEOUT_S="${SOAK_HEALTH_TIMEOUT_S:-1800}"  # Inductor compiles on first NONE boot
BASE_URL="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
REG="models/registry.yaml"
PY="${PYTHON:-python3}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
EAGER_OUT="bench/results/${TS}_cudagraph_eager_${MODEL}_${PROFILE}.json"
NONE_OUT="bench/results/${TS}_cudagraph_none_${MODEL}_${PROFILE}.json"
SUMMARY_OUT="bench/results/${TS}_cudagraph_ab_${MODEL}_${PROFILE}.json"
mkdir -p bench/results

HF_REPO="$("$PY" scripts/lib/registry.py hf_repo "$MODEL")"

step() { echo; echo "### $*"; }

# --- registry swap (backup + trap-restore) --------------------------------------
REG_BACKUP="$(mktemp)"
cp "$REG" "$REG_BACKUP"
REG_PATCHED=0
restore_registry() {
  if [ "$REG_PATCHED" -eq 1 ]; then
    cp "$REG_BACKUP" "$REG"
    echo "cudagraph_ab: registry restored (--enforce-eager default back in place)"
  fi
  rm -f "$REG_BACKUP"
}
trap restore_registry EXIT

patch_registry_none() {
  "$PY" - "$REG" <<'EOF'
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    src = fh.read()
new, n = re.subn(
    r"(?m)^(\s+)- --enforce-eager[^\n]*$",
    "\\g<1>- --compilation-config\n\\g<1>- '{\"cudagraph_mode\":\"NONE\"}'",
    src,
    count=1,
)
if n != 1:
    sys.exit(f"cudagraph_ab: expected exactly one '- --enforce-eager' line in {path}")
with open(path, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(new)
EOF
  REG_PATCHED=1
}

# --- teardown helper (same approach as run_matrix.py) -----------------------------
teardown_server() { # never raises
  pkill -f 'vllm serve' 2>/dev/null || true
  local host user
  while read -r _ host user; do
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "${user:+$user@}$host" 'pkill -f "vllm serve" || true' 2>/dev/null || true
  done < <("$PY" scripts/lib/registry.py nodes ansible/inventory.yaml | tail -n +2)
}
# Ctrl-C during the soak must not leave a NONE-config server running.
trap 'teardown_server; exit 130' INT TERM

# --- leg A: eager baseline ---------------------------------------------------------
step "1/4 eager leg (baseline): run_matrix $MODEL/$PROFILE concurrencies=$CONCURRENCIES"
python3 bench/run_matrix.py \
  --models "$MODEL" --profiles "$PROFILE" \
  --concurrencies "$CONCURRENCIES" --prompt-lengths "$PROMPT_LENGTHS" \
  --output "$EAGER_OUT"

# --- leg B setup: swap registry to cudagraph_mode NONE -----------------------------
step "2/4 switching registry defaults to cudagraph_mode NONE (temporary)"
patch_registry_none
grep -A1 'compilation-config' "$REG" | sed 's/^/  /'

# --- leg B soak: 10 min sustained load + hang detector ------------------------------
step "3/4 NONE soak: ${SOAK_SECONDS}s sustained load (concurrency $SOAK_CONCURRENCY) + hang detector"
SOAK_LOG="bench/results/${TS}_cudagraph_none_soak_serve.log"
bash scripts/serve.sh "$MODEL" "$PROFILE" >"$SOAK_LOG" 2>&1 &
SERVE_PID=$!

DMESG_OK=1
DMESG_RE='amdgpu.*(reset|ring.*timeout|lockup|hang)|GPU lockup|ring .* timeout'
BASELINE_RESETS=0
if ! dmesg >/dev/null 2>&1; then
  DMESG_OK=0
  echo "cudagraph_ab: WARNING: dmesg not readable (kernel.dmesg_restrict?) — hang detection via /health only" >&2
else
  BASELINE_RESETS="$(dmesg | grep -icE "$DMESG_RE" || true)"
fi

wait_health() { # wait_health <timeout_s>; returns 1 on death/timeout
  local deadline=$(( $(date +%s) + $1 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    kill -0 "$SERVE_PID" 2>/dev/null || return 1
    curl -sf -m 5 "$BASE_URL/health" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

VERDICT="pass"
if ! wait_health "$SOAK_HEALTH_TIMEOUT_S"; then
  VERDICT="start_failure"
  echo "cudagraph_ab: server did not become healthy under NONE (see $SOAK_LOG)" >&2
else
  echo "cudagraph_ab: healthy — soaking until $(date -d "+$SOAK_SECONDS seconds" +%H:%M:%S 2>/dev/null || echo "+${SOAK_SECONDS}s")"
  SOAK_END=$(( $(date +%s) + SOAK_SECONDS ))
  FAILDIR="$(mktemp -d)"
  WPIDS=()
  for w in $(seq 1 "$SOAK_CONCURRENCY"); do
    ( while [ "$(date +%s)" -lt "$SOAK_END" ]; do
        curl -sf -m 180 "$BASE_URL/v1/completions" \
          -H 'Content-Type: application/json' \
          -d "{\"model\":\"$HF_REPO\",\"prompt\":\"The capital of France is\",\"max_tokens\":64,\"temperature\":0}" \
          >/dev/null 2>&1 || echo fail >> "$FAILDIR/$w"
      done ) &
    WPIDS+=($!)
  done

  HEALTH_FAILS=0
  while [ "$(date +%s)" -lt "$SOAK_END" ]; do
    sleep 15
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then VERDICT="process_died"; break; fi
    if curl -sf -m 5 "$BASE_URL/health" >/dev/null 2>&1; then
      HEALTH_FAILS=0
    else
      HEALTH_FAILS=$((HEALTH_FAILS + 1))
      echo "cudagraph_ab: /health timeout #$HEALTH_FAILS"
    fi
    if [ "$HEALTH_FAILS" -ge 3 ]; then VERDICT="hang_health"; break; fi
    if [ "$DMESG_OK" -eq 1 ]; then
      NOW_RESETS="$(dmesg | grep -icE "$DMESG_RE" || true)"
      if [ "$NOW_RESETS" -gt "$BASELINE_RESETS" ]; then VERDICT="hang_amdgpu_reset"; break; fi
    fi
  done
  [ "${#WPIDS[@]}" -eq 0 ] || wait "${WPIDS[@]}" 2>/dev/null || true   # reap soak workers only
  NREQ_FAILS="$(cat "$FAILDIR"/* 2>/dev/null | wc -l || true)"
  rm -rf "$FAILDIR"
  echo "cudagraph_ab: soak finished, verdict=$VERDICT, failed_requests=$NREQ_FAILS"
fi

if [ "$VERDICT" != "pass" ]; then
  HANG_LOG="bench/results/${TS}_cudagraph_NONE_HANG_dmesg.log"
  echo
  echo "================================================================================"
  echo " CUDAGRAPH NONE: FAILED ($VERDICT) — cudagraph_mode NONE is NOT safe on gfx1151"
  echo " dmesg excerpt -> $HANG_LOG"
  echo " Rollback: automatic (registry trap-restores --enforce-eager). If the GPU is"
  echo " stuck in reset, reboot the affected node(s)."
  echo "================================================================================"
  {
    echo "# verdict: $VERDICT  model=$MODEL profile=$PROFILE ts=$TS"
    echo "# serve log tail:"
    tail -n 40 "$SOAK_LOG" 2>/dev/null || true
    echo "# dmesg (amdgpu/reset lines + last 80 lines):"
    if [ "$DMESG_OK" -eq 1 ]; then
      dmesg | grep -iE "$DMESG_RE" || true
      dmesg | tail -n 80
    else
      echo "(dmesg not readable)"
    fi
  } > "$HANG_LOG"
  teardown_server
  cat > "$SUMMARY_OUT" <<EOF
{"model": "$MODEL", "profile": "$PROFILE", "ts": "$TS",
 "eager_results": "$EAGER_OUT", "none_results": null,
 "none_verdict": "$VERDICT", "hang_log": "$HANG_LOG"}
EOF
  exit 1
fi

step "soak passed — tearing down before the measured NONE leg"
teardown_server
sleep 10   # let ports/RCCL settle

# --- leg B measured: run_matrix under NONE -----------------------------------------
step "4/4 none leg: run_matrix $MODEL/$PROFILE concurrencies=$CONCURRENCIES"
python3 bench/run_matrix.py \
  --models "$MODEL" --profiles "$PROFILE" \
  --concurrencies "$CONCURRENCIES" --prompt-lengths "$PROMPT_LENGTHS" \
  --output "$NONE_OUT"

cat > "$SUMMARY_OUT" <<EOF
{"model": "$MODEL", "profile": "$PROFILE", "ts": "$TS",
 "eager_results": "$EAGER_OUT", "none_results": "$NONE_OUT",
 "none_verdict": "pass", "soak_seconds": $SOAK_SECONDS}
EOF

echo
echo "cudagraph_ab: DONE — NONE survived the ${SOAK_SECONDS}s soak."
echo "  eager:   $EAGER_OUT"
echo "  none:    $NONE_OUT"
echo "  summary: $SUMMARY_OUT"
echo "  compare: python3 bench/report.py '$EAGER_OUT' '$NONE_OUT'"
echo "  If NONE wins, promote it by replacing --enforce-eager in models/registry.yaml"
echo "  defaults.extra_args with --compilation-config '{\"cudagraph_mode\":\"NONE\"}'."
