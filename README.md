# vLLM-Cluster für 4× AMD Strix Halo (gfx1151) über 25 GbE RoCE

Eigene Build-Pipeline + Cluster-Orchestrierung für vLLM auf 4 Strix-Halo-Nodes
(Ryzen AI Max+ 395, iGPU gfx1151 / RDNA 3.5, je 128 GB Unified Memory),
verbunden über 25 GbE RDMA (RoCEv2). Ziel: immer aktuelle, lauffähige Images
(Stable-Kanal: letztes vLLM-Release, Dev-Kanal: vLLM main für Day-0-Modelle)
und maximaler Durchsatz, entschieden durch eigene Benchmarks.

## Struktur

- `docker/` — Container-Image (Fedora 44, ROCm/torch gfx1151, vLLM aus Source)
- `patches/` — idempotente gfx1151-Patch-Schicht + fail-closed Kompatibilitätsprüfung
- `.github/workflows/` — Build-Pipeline (stable / dev / model-watch / rccl)
- `ansible/` — Provisionierung der 4 Fedora-Nodes (Base, RDMA, Runtime, Ray)
- `scripts/` — Cluster-Start (`cluster_up.sh`) und Serven (`serve.sh <modell> <profil>`)
- `bench/` — Benchmark-Harness (Single-Stream tok/s + Aggregat-Durchsatz, TP/PP/EP-Matrix)
- `models/registry.yaml` — zentrale Modell-Registry (Status, Parser, Profile, Blocker)
- `docs/` — Runbook und Hintergrunddokumente

## Parallel-Profile

`tp4` (Tensor-Parallel über Ray/RCCL), `pp4` (Pipeline-Parallel), `tp2pp2`,
`ep` (Expert-Parallel für MoE), `solo` (1 Node, Baseline). Welches Profil pro
Modell gewinnt, entscheidet `bench/run_matrix.py` — auf 25 GbE ist das
empirisch offen (Referenzdaten existieren nur für 100/200 GbE).

## Quickstart (Überblick)

1. Image bauen lassen (GitHub Actions, ghcr.io) oder lokal: `docker/`
2. Nodes provisionieren: `ansible-playbook -i ansible/inventory.yaml ansible/playbooks/site.yml`
3. Cluster hochfahren: `scripts/cluster_up.sh`
4. Modell serven: `scripts/serve.sh qwen36-35b-a3b tp4`
5. Benchmarks: `python3 bench/run_matrix.py --model qwen36-35b-a3b`

Details: `docs/RUNBOOK.md`.

## Bekannte Einschränkungen

- GLM-5.3-Flash ist upstream-blockiert (gfx950-gated), siehe `models/registry.yaml`.
- `--enforce-eager` ist Pflicht (HIP-Graph-Deadlocks auf gfx1151).
- `amd_iommu=off` vs. RDMA: ungelöster Trade-off, per `iommu_mode` parametrierbar,
  A/B-Test über `bench/iommu_ab.sh`.

## Performance-Programm (Ziel: schneller als DGX Spark)

Dev-Builds basieren auf dem jeweils frischesten vLLM-Dev/PR-Stand (Registry-Feld
`vllm_ref` pinnt PR-Heads per SHA; `model-watch` triggert Rebuilds, wenn Heads
sich bewegen). Die gfx1151-Performance-Patches liegen in `patches/` (Serie 50–58,
Doku: `patches/manifest.d/`): MXFP4-MoE-Tuning, Radix-Top-k, TileLang-Sparse-Indexer,
W8A8-Skinny-GEMM, AITER-Triton-Enablement, APU-Memory-Reporting, PLE-Offload
(Zero-Copy auf Unified Memory), GLM-MTP-Dispatch. Strategie, Messwerte,
Spark-Referenzziele und Messplan: **`docs/PERFORMANCE.md`**.
