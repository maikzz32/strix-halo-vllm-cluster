# Performance-Strategie — Strix-Halo-Cluster vs. DGX Spark

Zentrale Strategie-Doku: warum wir welche Optimierungen bauen, welche
Messwerte dafür sprechen, gegen welche Referenz wir uns messen und in
welcher Reihenfolge gemessen wird. Code-Knöpfe (`VLLM_GFX1X_*`,
Registry-env) sind hier nur dokumentiert — gesetzt werden sie in
`models/registry.yaml` und den Patches unter `patches/`.

**Hardware-Grundlage (Stärken/Schwächen gfx1151 vs. GB10, abgeleitete
Design-Regeln): `docs/HARDWARE_GFX1151.md` — dort zuerst lesen.**

## a) Ziel und ehrliche Ausgangslage

**Ziel:** Bei gleicher Modellklasse (Qwen3.8-Flash-Next-Klasse, ~6B aktive
Parameter, MoE) das DGX-Spark-Referenz-Aggregat schlagen.

Ehrliche Ausgangslage pro Node (korrigiert auf **gemessene** Bandbreite):

- DGX Spark (GB10): 273 GB/s spezifiziert, **~205 GB/s gemessen** (~75 %).
- Strix Halo (gfx1151): 256 GB/s spezifiziert, **~215 GB/s gemessen** (~84 %).

Pro Node herrscht bei bandbreitenlimitiertem Decode also **Parität bis
leichter Vorteil Halo** (gemessen, nicht spezifiziert: gpt-oss-120b 50 vs.
53 tok/s, gpt-oss-20b 73 vs. 80 — siehe HARDWARE_GFX1151.md §4). Die Lücke
liegt im **Prefill** (Compute: ~36–40 vs. ~100 BF16-TFLOPS) und im
**Interconnect** (25 GbE vs. ~196 Gbps). Der Sieg kommt über drei Hebel:

1. **Software-Tuning** (MoE-/Attention-Kernels, Compile-Pfad) — auf der
   Spark-Seite größtenteils bereits ausgereizt, bei uns noch nicht.
2. **MTP / Speculative Decoding** — multiplikativ auf alles andere.
3. **4-Node-Aggregat** — die Spark-Referenzwerte stammen von 2 Nodes; wir
   stellen 4 Nodes dagegen.

## b) Hebel-Rangliste (mit belegten Messwerten)

Sortiert nach erwartetem Gesamteffekt. „Kernel-Level" bedeutet: isolierter
Kernel-Benchmark, nicht End-to-End — die Übertragung auf tok/s hängt vom
Anteil des Kernels an der Decode-Zeit ab.

| # | Hebel | Belegter Effekt | Quelle | Status bei uns |
|---|-------|-----------------|--------|----------------|
| 1 | MTP / Speculative Decoding | 1,5–2× Decode; GLM-5.3 MTP-Fix allein: 2632 → 3760 tok/s @ N=64 (gfx950 TP=8) | vllm#53943, vllm.ai AMD-Blog | Registry: `--speculative-config '{"method":"mtp",...}'` für qwen38-flash-next; GLM-Dispatch-Fix = Patch 58 |
| 2 | MXFP4-MoE `opt_flags`-Tune (BM=16 BN=32 BK=256 NW=2 NS=2 WPE=1, kpack=1) | 11,28 → 21,77 tok/s Decode (gfx1151, gemessen) | kyuz0 amd-strix-halo-vllm-toolboxes | `VLLM_GFX1X_MOE_TUNE=1` (Patch 40, Registry-Default) |
| 3 | TileLang Sparse-Indexer MQA-Kernel (Decode-Grid über KV-Tokens, 64-Token-Blöcke, 256 Threads, 56 KB LDS, KV-Bucketing 512/8192 gegen JIT-Stalls) | ~47× Kernel-Level | AlexKGwyn/ds4-vllm | Patch 52 (vendored, Gates `VLLM_GFX1X_TL_*`) |
| 4 | W8A8 Cached-BF16 Skinny GEMM (wvSplitK bei M≤5) | ~24× bei M=1 (Single-Stream-Decode!) | AlexKGwyn/ds4-vllm | Patch 53 |
| 4b | **m=1-Skinny-GEMM-Falle in `rocm_unquantized_gemm_impl`** (shared_expert_gate [1,2048] → pathologischer rocBLAS-Kernel, 137,5 µs statt 6,8 µs) | 13,7 % der Step-Zeit; Fix = **24,53 → 28,32 tok/s (+15,5 % Decode, auf gfx1151 gemessen)**; Caveat: nicht bit-identisch (max abs diff ~2,5e-1) | vllm#52631 | Patch 60 (`VLLM_GFX1X_SKINNY_M1`, default an); falls vllm#53283 merged: Patch selbst-SKIP |
| 5 | Radix-Top-K für Sparse-Indexer | 2,2–8,6× Kernel-Level | AlexKGwyn/ds4-vllm | Patch 51; Risiko: `tl.histogram`-Lowering auf ROCm (s. e) |
| 6 | Getunte MoE-Triton-Configs pro Form (`VLLM_TUNED_CONFIG_FOLDER`, Datei-Schema `E={E},N={N},device_name=AMD_Radeon_8060S[,dtype=...].json`; Tuner: `benchmarks/kernels/benchmark_moe.py --tune`) | +25–63 % (RDNA4-Datenpunkte) | vllm#28649 | Config-Dateien müssen einmalig auf der Hardware getunt werden (s. Messplan) |
| 7 | Kompilieren ohne Graph-Capture: `--compilation-config '{"cudagraph_mode":"NONE"}'` statt `--enforce-eager` — Inductor-Fusion bleibt aktiv, nur der (auf gfx1151 dead-lockende) Capture entfällt; plus `--async-scheduling` | Präzedenzfall vllm#44988; `--enforce-eager` ist wegen vllm#32180 aktuell Pflicht, aber wahrscheinlich überbreit | vllm#44988, vllm#32180 | A/B-Messung, dann Registry-Default ändern |
| 8 | PLE-Offload Zero-Copy auf der APU (`VLLM_PLE_CPU_OFFLOAD` aus PR #53899; pinned host + UVA Triton gather + Prefetch-Stream) | Auf Strix Halo IST „Host-Speicher" GPU-Speicher: volle ~215 GB/s statt PCIe-Flaschenhals wie auf der Spark (die PLE von NVMe/mmap streamen muss) | PR #53899 (peakcrosser7/vllm) | Patch 57 portiert den Pfad nach `amd/ple_layer.py`; Registry setzt `VLLM_PLE_CPU_OFFLOAD=1` für qwen38-flash-next |
| 9 | APU-Memory-Reporting reparieren (HIP meldet ~15,5 GiB VRAM statt ~110+ GiB GTT) | Voraussetzung für korrektes KV-Budget — ohne Fix verschenken wir KV-Kapazität | ROCm/hip#3892, Fix vllm#40963 | Patch 56 |
| 10 | Compile-Cache über identische Nodes verteilen (Triton-Cache ist arch-keyed; vLLM-Compile-Cache hasht die ganze Config) | Präzedenz: Cold-Start 294 s → 82 s; TRITON_STORE_BINARY_ONLY=1 schrumpft den Cache ~77 % | Red Hat Triton-Cache-Analyse, tensorfuse | `scripts/warmup.sh` + `scripts/dist_cache.sh`; Registry als Single-Source gegen Hash-Drift |
| 11 | SCLK-Cap 2400–2500 MHz (Decode flach über Takt, Prefill linear; unkontrollierter Boost = 180 W/111 °C/Lock-Risiko) | ~0 % Decode-Verlust, ~−80 W, Stabilität | strix-halo-guide #24 | ansible `base.yml` Tag `gpu`, `gpu_sclk_cap_mhz: 2500` |
| 12 | gfx1151-only-Build (Arch-Pin gegen vllm#22590, Dep-Schnitt, Platform-Fast-Path) | Build-Zeit Stunden→~30 min Klasse; Bild kleiner; Start schneller | vllm#22590, lemonade-sdk | Dockerfile.fedora, Patch 59 |

Kein Hebel: **FP8 auf RDNA 3.5** — es gibt keine native FP8-Hardware, FP8
läuft emuliert mit BF16-Tempo. AITER-FP4-Pfade liefern außerhalb von
gfx950 still Nullen zurück (Correctness-Hazard) — FP4 bleibt abgeschaltet.

## c) Spark-Referenzwerte (offizielle Benchmark-Ziele)

Quelle für alle drei Zeilen:
[maci0/qwen3.8-flash-next-spark](https://github.com/maci0/qwen3.8-flash-next-spark)
(2× DGX Spark, direkte CX7-RoCE-Strecke, Qwen3.8-Flash-Next — also exakt
unsere Modellklasse). Diese Werte gelten als die zu schlagende Latte und
stehen auch in `bench/report.py` (Abschnitt „Spark-Vergleich").

| Stack (Spark) | Single-Stream | Aggregat | Bemerkung |
|---|---|---|---|
| vLLM TP=2, BF16 KV, MTP 3, 512K YaRN | ~31 tok/s (31,1) | ~74 tok/s (74,3 @ C=8) | PLE per mmap von NVMe |
| SGLang NVFP4 TP=2, 1M ctx | 40–44 tok/s | ~150 tok/s (148–155 @ C=24) | NVFP4-Experten-Quant; für uns mangels FP8/FP4-Hardware kein 1:1-Pfad — Referenz bleibt trotzdem die Latte |
| llama.cpp GGUF + MTP (1× Spark) | ~32 tok/s (32,1, +17 % durch MTP) | — (Single-Node) | Fallback-Referenz |

`bench/report.py` rechnet pro Modell das Verhältnis bestes
Cluster-Ergebnis / Spark-Referenz und markiert `BEATEN` bzw. `NOT-YET`.

## d) Messplan

Reihenfolge (jeder Schritt baut auf dem vorherigen auf):

1. **`tune_moe` / MoE-Tuning** — Triton-MoE-Configs auf der Hardware
   erzeugen (`benchmarks/kernels/benchmark_moe.py --tune` im
   vLLM-Checkout), Ergebnis-JSONs ins `VLLM_TUNED_CONFIG_FOLDER`-Schema
   bringen. Ohne diesen Schritt misst man Default-Configs, nicht das
   Cluster.
2. **`bench/run_matrix.py`** — Modell × Profil × Prompt-Länge ×
   Concurrency-Matrix; liefert die Cluster-Bestwerte für den
   Spark-Vergleich.
3. **`bench/compare_eth_vs_rdma.sh`** — RCCL über RoCE vs. erzwungenes
   TCP; entscheidet, ob das Fabric der Engpass ist.
4. **`bench/iommu_ab.sh`** — IOMMU-Modus-A/B (braucht Reboots), klärt die
   RDMA-Frage `amd_iommu=off` vs. `iommu=pt`.

**Akzeptanzkriterium:** Cluster-Aggregat > Spark-Aggregat bei gleicher
Modellklasse (Qwen3.8-Klasse). Konkret: >74 tok/s Aggregat gegen vLLM
TP=2, ausgerichtet an ~150 tok/s der SGLang-NVFP4-Referenz. Single-Stream
~31 tok/s ist die Sekundär-Latte (hier tragen Skinny-GEMM und MTP die
Hauptlast).

## e) Offene Risiken

- **GLM-5.3 auf gfx1151 ungetestet:** PR #53906 ist offiziell
  gfx950-gated (MI355X); Patch 58 (MTP-Dispatch) ist gegen die
  recherchierte Code-Form geschrieben, aber noch nicht gegen einen
  echten glm-release-Checkout gelaufen. FlashInfer ≥ 0.6.18 nötig.
- **`tl.histogram`-Lowering auf ROCm:** der Radix-Top-K-Kernel hängt an
  Triton-Primitiven, deren ROCm-Lowering auf gfx1151 nicht verifiziert
  ist — im Zweifel fällt Hebel 5 aus oder bricht die Kompilierung.
- **Patch-Schicht insgesamt unvalidiert:** alle Patches sind gegen
  recherchiertes Upstream-Verhalten geschrieben und laufen erst beim
  ersten echten Image-Build / Hardware-Lauf. Exit-42-Meldungen sind das
  dafür vorgesehene Signal (Re-Audit, nicht stilles Überspringen).
- **PLE-Offload ist NVIDIA-seitig implementiert** (`nvidia/ple_layer.py`
  im PR; `amd/ple_layer.py` ohne Offload) — die Zero-Copy-These auf der
  APU muss erst durch einen echten Lauf belegt werden.
- **Referenzwerte in Bewegung:** die Spark-Seite optimiert weiter (im
  maci0-Repo bereits SGLang-spec-Varianten >80 tok/s Single). Die Latte
  in Abschnitt c) ist der Stand von 2026-08-30 und muss periodisch
  nachgezogen werden.
