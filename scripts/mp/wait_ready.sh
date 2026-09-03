#!/bin/bash
# wait_ready.sh <logdatei>  -- wartet bis /v1/models antwortet oder der Server stirbt
L="${1:?log}"
for i in $(seq 1 120); do
  curl -s -m 3 localhost:8000/v1/models >/dev/null 2>&1 && { echo "READY nach ~$((i*5))s"; exit 0; }
  pgrep -f "[s]bin/vllm serve" >/dev/null || { echo "PROZESS TOT"; grep -E "Error|Traceback" "$L" | tail -5; exit 1; }
  sleep 5
done; echo TIMEOUT; exit 2
