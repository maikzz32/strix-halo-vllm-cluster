#!/bin/bash
# start_mp_exp.sh <rank> <container> <ip> <tag>  -- mp-Verbund, Experiment-Variante ueber Env:
#   CG_MODE (NONE|FULL_DECODE_ONLY|PIECEWISE), CAP_SIZES (z.B. [1]), MTP_K (0 = kein MTP), EXTRA_E ("-e X=1 -e Y=2")
R="${1:?rank}"; C="${2:?container}"; IP="${3:?ip}"; T="${4:-exp}"
HL=""; [ "$R" != "0" ] && HL="--headless"
CC="{\"cudagraph_mode\":\"${CG_MODE:-NONE}\""; [ -n "${CAP_SIZES:-}" ] && CC="$CC,\"cudagraph_capture_sizes\":${CAP_SIZES}"; CC="$CC}"
SPEC=(); [ "${MTP_K:-3}" != "0" ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_K:-3}}")
exec podman exec ${EXTRA_E:-} -e VLLM_GFX1X_MOE_INT4_GEMV=1 \
  -e VLLM_ROCM_USE_AITER=0 -e VLLM_GFX1X_MOE_TUNE=1 -e VLLM_GFX1X_FAST_PLATFORM=1 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85 \
  -e NCCL_IB_GID_INDEX=1 -e NCCL_NET_GDR_LEVEL=0 -e VLLM_HOST_IP="$IP" \
  "$C" vllm serve /home/maik/qwen38_flashnext --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 --nnodes 4 --node-rank "$R" --master-addr 192.168.100.1 --master-port 50001 \
  --distributed-executor-backend mp $HL \
  --compilation-config "$CC" "${SPEC[@]}" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --async-scheduling > "/tmp/tp4_${T}_r$R.log" 2>&1
