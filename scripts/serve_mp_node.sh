#!/bin/bash
# serve_mp_node.sh <node-rank> <container> <own RoCE IP> [tag]
#
# Ray-free multi-node serving with vLLM's mp executor (--nnodes 4). Run it on
# EVERY node: rank 0 on node1 (ray-head container, becomes the API server) and
# ranks 1-3 on node2-4 (ray-worker containers, --headless: no API server, see
# docs/serving/parallelism_scaling.md). Measured 2026-09-03 on
# qwen38-flash-next-int4: 29.9 (Ray) -> 31.27 tok/s (greedy, c=1, MTP k=3).
# eager; with FULL_DECODE_ONLY target graphs + the RCCL envs: 41.0 tok/s,
# 600 s soak x4 without a failure.
# The env list mirrors models/registry.yaml defaults + serve.sh's distributed
# env; NCCL_SOCKET_IFNAME comes from the container env (cluster_up.sh).
# Before a restart make sure no VLLM::EngineCore / VLLM::Worker processes
# linger in any container: they keep --master-port and GPU memory.
# Log: /tmp/tp4_<tag>_r<rank>.log on each node.
R="${1:?rank}"; C="${2:?container}"; IP="${3:?ip}"; T="${4:-mp}"
HL=""; [ "$R" != "0" ] && HL="--headless"   # Follower: kein API-Server (docs/serving/parallelism_scaling.md)
# Graphs on the target model only (patch 65 gates the speculator's graphs);
# NCCL_GRAPH_MIXING_SUPPORT=1 + NCCL_LAUNCH_MODE=GROUP are REQUIRED with graphs:
# without either, all ranks freeze after ~15 iterations (E4/E6/E8, 2026-09-03).
CG_MODE="${CG_MODE:-FULL_DECODE_ONLY}"
exec podman exec -e VLLM_GFX1X_MOE_INT4_GEMV=1 -e VLLM_GFX1X_SPEC_CUDAGRAPH="${SPEC_CG:-0}" \
  -e NCCL_GRAPH_MIXING_SUPPORT=1 -e NCCL_LAUNCH_MODE=GROUP \
  -e VLLM_ROCM_USE_AITER=0 -e VLLM_GFX1X_MOE_TUNE=1 -e VLLM_GFX1X_FAST_PLATFORM=1 \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.85 \
  -e NCCL_IB_GID_INDEX=1 -e NCCL_NET_GDR_LEVEL=0 -e VLLM_HOST_IP="$IP" \
  "$C" vllm serve /home/maik/qwen38_flashnext --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 --nnodes 4 --node-rank "$R" --master-addr 192.168.100.1 --master-port 50001 \
  --distributed-executor-backend mp $HL \
  --compilation-config "{\"cudagraph_mode\":\"$CG_MODE\"}" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --async-scheduling > "/tmp/tp4_${T}_r$R.log" 2>&1
