#!/usr/bin/env bash
set -euo pipefail
TAG="${1:?result tag}"
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
OUT="/home/maik/strix-halo-next/results/$TAG"
mkdir -p "$OUT"
curl --fail --silent --max-time 10 http://127.0.0.1:8000/metrics > "$OUT/metrics-before.txt"
podman exec qwen029-tp2 vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://localhost:8000 --model /home/maik/qwen38_rest \
  --dataset-name sharegpt \
  --dataset-path /home/maik/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 48 --max-concurrency 1 --seed 42 --temperature 0 \
  --save-result --save-detailed --result-dir "/tmp/codex-benchmark-$TAG" --result-filename result.json \
  2>&1 | tee "$OUT/benchmark.log"
podman cp "qwen029-tp2:/tmp/codex-benchmark-$TAG/result.json" "$OUT/result.json"
curl --fail --silent --max-time 10 http://127.0.0.1:8000/metrics > "$OUT/metrics-after.txt"
