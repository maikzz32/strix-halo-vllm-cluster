# vLLM-Cluster für 4× AMD Strix Halo (gfx1151) über 25 GbE RoCE

Eigene Build-Pipeline + Cluster-Orchestrierung für vLLM auf 4 Strix-Halo-Nodes
(Ryzen AI Max+ 395, iGPU gfx1151 / RDNA 3.5, je 128 GB Unified Memory),
verbunden über 25 GbE RDMA (RoCEv2). Ziel: immer aktuelle, lauffähige Images
(Stable-Kanal: letztes vLLM-Release, Dev-Kanal: vLLM main für Day-0-Modelle)
und maximaler Durchsatz, entschieden durch eigene Benchmarks.

## Aktueller Versuch: Qwen3.8 Flash Next, eine laufende Antwort

Für die vorhandenen vier Fedora-45-Systeme gibt es jetzt einen separaten
[nativen Startpfad](scripts/native/README.md) mit vLLM `mp`, TP4 und MTP3.
Er verwendet die vorhandenen Container `ray-head`/`ray-worker` und prüft alle
Ranks vor dem Start. Die Modellgewichte liegen auf jedem Node unter
`/home/maik/qwen38_rest`. Der Image-Tag allein enthält nicht alle Änderungen
der bereits installierten Laufzeit; die Voraussetzungen im Runbook beachten.

Dieser Pfad wurde mit `FULL_DECODE_ONLY` getestet. Die ältere allgemeine
Graph-Einschränkung weiter unten gilt daher nicht für diese konkrete Kombination
aus Runtime, Patches und Modell. Für andere Kombinationen bleiben eigene Tests
erforderlich.

Die QSA-Läufe erreichen **49,56–50,30 statt 48,60 Output-Token/s** bei ShareGPT,
48 Anfragen, jeweils einer gleichzeitig. Die reine Generierung verbessert sich
um etwa 2 % durch das Überspringen unsichtbarer QSA-Score-Tiles bei unveränderten Gewichten.
Die 50,30 im abschließenden Neustarttest enthalten zusätzlich kürzere Anlaufzeiten.
Das Ziel einer deutlich schnelleren einzelnen Antwort ist noch nicht erreicht.
[Messungen und verworfene Varianten](docs/2026-09-06-single-stream.md).
Separate kurze Coding-Antworten
erreichten schon vor den neuen Änderungen etwa 67 Decode-Token/s. Diese
unterschiedlichen Workloads dürfen nicht als Vorher/Nachher-Gewinn verglichen
werden. [DGX-Spark-Vergleich](docs/2026-09-06-spark-comparison.md).

`bench/bench_stream.py` speichert vollständige Requests, Antworten und
Streaming-Zeitpunkte; `bench/compare_streams.py` vergleicht passende Läufe
und prüft identische Ausgaben. `bench/bench_sharegpt.sh` wiederholt den
bisherigen C1-Benchmark. Die Optimierungsversuche sind noch nicht abgeschlossen.

Der [kalibrierte HIP/RDMA-Versuch](docs/2026-09-07-hip-rdma-ring4.md)
verkürzt den isolierten 20-KiB-All-reduce auf 56,42 µs gegenüber ungefähr
85,3 µs mit RCCL. Im vollständigen Modelltest entsteht daraus jedoch kein
Gewinn: **49,79 statt 50,49 Output-Token/s** im direkten ShareGPT-Kontrolllauf,
bei 48 identischen Antworten und gleicher MTP-Akzeptanz. Der Versuch wurde
zurückgebaut; der laufende Dienst verwendet die bewährte QSA/RCCL-Konfiguration.
Die [GitHub-Recherche](docs/github-vllm-projects-20260907.md) bewertet acht
Projekte und benennt konkrete nächste Kernel- und Kommunikationsansätze.
Der daraus abgeleitete [AITER-HC-Test](docs/2026-09-07-aiter-hc.md) war bei
allen drei Zielgrößen langsamer als die bereits vorhandenen BF16-Kernel.

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

`tp2`/`tp4` (Tensor-Parallel über Ray/RCCL), `pp4` (Pipeline-Parallel), `tp2pp2`,
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

- GLM-5.3-Flash **läuft** (2026-09-01, tp4, Dev-Image `dev-glm53-flash`):
  upstream gfx950-gated, bei uns via Patch 58 + 61 + Torch-Kpool-Lane
  (Details: `models/registry.yaml`, `docs/PERFORMANCE.md` §e). Bring-up-
  Qualitätscaveat beachten (G1-Repetition auf Kurz-Prompts).
- Graph-**Capture** deadlocked auf gfx1151 (HIP, vllm#32180) — Default ist daher
  `cudagraph_mode NONE` (Inductor-Fusion bleibt aktiv; gemessen +7–9 % vs.
  `--enforce-eager`, 600-s-Soak hang-frei, Stand 2026-08-31).
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
