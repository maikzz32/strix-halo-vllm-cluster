#!/bin/bash
# bench_c1.sh [zusatzargumente fuer vllm bench serve, z.B. --temperature 0]
set -u
M=/home/maik/qwen38_flashnext
DS=/home/maik/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
podman exec ray-head vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://127.0.0.1:8000 \
  --model "$M" \
  --dataset-name sharegpt --dataset-path "$DS" \
  --num-prompts 48 --max-concurrency 1 --seed 42 \
  --percentile-metrics ttft,tpot,itl,e2el "$@" 2>&1 | grep -vE "rocSHMEM|agent.cpp|^W0|amdsmi|^E-001h|it/s\]"
