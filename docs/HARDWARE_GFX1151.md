# Hardware-Profil: AMD gfx1151 (Strix Halo) — Stärken, Schwächen, Design-Regeln

Zweck: dieses Repo baut ein vLLM, das **nur** auf gfx1151 laufen muss. Dieses
Dokument ist die Hardware-Grundlage, von der alle Patches, Configs und
Playbooks abgeleitet sind. Jede Zahl ist mit Quelle belegt; Schätzungen sind
markiert. Stand: 2026-08-30. Vergleichsreferenz: NVIDIA GB10 (DGX Spark).

## 1. Compute-Komplex

- **40 CUs = 20 WGPs**, 2 Shader Engines × 2 Arrays, 2× SIMD32 pro CU,
  **wave32**, max. 32 Waves/CU, Workgroup max 1024 Threads.
  ([rocminfo, ROCm#6027](https://github.com/ROCm/ROCm/issues/6027))
- **Boost 2900 MHz**, aber: unter LLM-Last regelt er **nicht selbst herunter** —
  ~180 W / ~111 °C, vereinzelt Hard-Locks. SMU ignoriert Software-Powerlimits
  (ryzenadj wirkungslos); einziger wirksamer Hebel ist ein harter **SCLK-Cap**
  via `pp_od_clk_voltage`. Sweet Spot gemessen: **2400–2500 MHz** (~100 W,
  ~70 °C, Heat-Soak erst ab ~2500–2600).
  ([strix-halo-guide #24](https://github.com/hogeheer499-commits/strix-halo-guide/issues/24))
- **Decode ist bandbreitenlimitiert und flach über dem Takt; Prefill skaliert
  ~linear mit dem Takt** (117→134 t/s für 2200→2600 MHz). Konsequenz: Cap
  kostet fast nichts an Decode, verhindert aber Thermal-Instabilität.
  → `ansible/playbooks/base.yml` (Tag `gpu`), `gpu_sclk_cap_mhz: 2500`.
- Rechenleistung: **FP32 ~29,7 TFLOPS, FP16/BF16 ~59,4 TFLOPS** (packed +
  WMMA, theoretisch @2,9 GHz; real ~36–40 TFLOPS gemessen, thermisch/dispatch
  begrenzt, [TheRock#5314](https://github.com/ROCm/TheRock/issues/5314)).
- **VGPR-File 192 KB/SIMD** (großes RDNA3-Registerfile; gfx1150 hat nur
  128 KB) — hohe Occupancy mit fetten Kernels möglich.
  ([Chips and Cheese](https://chipsandcheese.com/p/amd-rdna-3-5s-llvm-changes))

## 2. Speichersystem — der entscheidende Teil

- **256-bit LPDDR5X-8000, 256 GB/s theoretisch, ~212–215 GB/s real gemessen**
  (~83–84 % Effizienz, mehrere unabhängige Messungen). Decode hängt hier.
- **MALL/Infinity-Cache: 32 MB** (nicht 128!), ~1 TB/s Hit-Bandwidth,
  ~700 GB/s gemessen bei ≤12 MiB Working Set, ~228 GB/s ab ≥48 MiB.
  **Wichtige Caveats:** (a) die CPU kann den MALL **nicht** nutzen;
  (b) Cache-Politik ist Firmware-gesteuert, nicht garantiert; (c)
  Host-Pointer-Zero-Copy-Buffer **umgehen** den MALL komplett.
  ([C&C Memory Subsystem](https://chipsandcheese.com/p/strix-halos-memory-subsystem-tackling),
  [hello-gpu ch4](https://github.com/datawhalechina/hello-gpu/blob/main/docs/part1-hardware-rocm/chapter4/index.md))
- Restliche Hierarchie: L0 32 KB/CU, L1 256 KB/Shader-Array, **L2 nur 2 MB**,
  **LDS 64 KB/CU** (harte Kernel-Constraint: block_N ≥ 256 läuft über).
- GTT/Unified Memory: GPU-Allokationen landen physisch im selben LPDDR5 —
  "CPU-Offload" ist eine Speichercontroller-Kopie, **kein** PCIe-Transfer.
  Konfiguration: `amdgpu.gttsize=126976 ttm.pages_limit=32505856`;
  `amd_iommu=off` = gemessen 5–12 % schneller (kyuz0 #66), RDMA-Kompatibilität
  per `bench/iommu_ab.sh` klären.
- **CPU↔GPU-Bandbreiten-Sharing:** Zen5-CCDs ziehen bis 124–175 GB/s — jede
  CPU-Last während Decode stiehlt direkt GPU-Bandbreite. Regel: CPU-Seite
  (Tokenisierung, Sampling, Parsing) während Decode schlank halten.
  ([C&C](https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix))

## 3. WMMA / ISA — was das Silizium kann

- Nativ per WMMA: **f16, bf16, i8, i4** (`v_wmma_*_16x16x16_*`, Tile immer
  16×16×16). **Kein FP8/FP6/FP4-Matrix-Pfad** — FP8 läuft emuliert auf
  BF16-Tempo, FP8-Checkpoints sind hier nur Speicherformat.
  ([GPUOpen](https://gpuopen.com/learn/wmma_on_rdna3/))
- Triton senkt `tl.dot` für f16/bf16/i8 auf WMMA ab (Dims als Vielfache von
  16 wählen!). **iu4 wird nirgends abgesenkt** — INT4-WMMA-Silizium existiert,
  aber kein Software-Pfad nutzt es: **die größte unerforschte Chance**.
- Triton-Fallen: triton#9175 (WMMA + skalare Loads → Compiler-Crash),
  triton#9815 (Pipeliner num_stages=4), INT8-`tl.dot`-Deckel
  BLOCK_M/N ≤ 32 / BLOCK_K ≤ 64 auf gfx1151.

## 4. GB10 (DGX Spark) zum Vergleich

| Achse | GB10 | gfx1151 | Fazit |
|---|---|---|---|
| Decode-Bandbreite/Node | 273 GB/s spec, **~205 gemessen** | 256 spec, **~215 gemessen** | **Parität bis leicht Halo** |
| Last-Level-Cache | 24 MB L2 + 16 MB SLC (nicht additiv) | 2 MB L2 + 32 MB MALL | leicht Halo |
| BF16-Compute | ~100–125 TFLOPS | ~59 TFLOPS theo., ~36–40 real | GB10, ~2–3× |
| Quant-Pfad | NVFP4 nativ (~500 TFLOPS dense) | BF16/INT8 WMMA, FP8 emuliert | GB10 |
| CPU↔GPU-Link | NVLink-C2C ~600 GB/s | on-package Fabric, gleicher Speicher | GB10 (Link), Halo (Zero-Copy) |
| Inter-Node | 196 Gbps real (NCCL ~40 GB/s) | 25 GbE | **GB10, ~8×** |
| Cluster-Reife | offiziell nur 2 Nodes, Treiberbugs | ROCm fragil, aber 4 Nodes belegt | beide fragil |
| Strom | ~100–140 W, Firmware-Bugs | ~100–140 W, SCLK-Cap nötig | Parität |

Quellen: [C&C GB10](https://chipsandcheese.com/p/analyzing-nvidia-gb10s-gpu),
[nvfp4bench-Thread](https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618),
[StorageReview Cluster](https://www.storagereview.com/review/nvidia-dgx-spark-cluster-review-distributed-inference-on-dell-gigabyte-and-hp).

**Gemessene Decode-Parität existiert bereits:** gpt-oss-120b llama.cpp:
Halo 50 vs Spark 53 t/s; gpt-oss-20b: 73 vs 80. Prefill ist die Lücke
(788 vs 1689 t/s). ([ihower.tw](https://ihower.tw/blog/13294-framework-desktop))

## 5. Abgeleitete Design-Regeln (die "perfekte Software" für diesen Chip)

1. **Decode: Bytes sind alles.** Bei ~215 GB/s zählt nur Gewichtsgröße pro
   Token: MXFP4/INT8-Gewichte, KV-Cache-Quantisierung, MTP (mehr Tokens pro
   Gewichts-Durchlauf). Kein FP8-Compute erwarten — FP8 = BF16-Tempo, aber
   halbe Bytes → trotzdem als *Speicherformat* nützlich.
   → Patches 50/53, Registry `dtype`, MTP extra_args.
2. **Skinny-GEMM ist der Decode-Feind Nr. 1.** rocBLAS fällt bei m=1 auf
   pathologische Kernel (137,5 µs statt 6,8 µs; 13,7 % der Step-Zeit;
   Fix = +15,5 % Decode, gemessen auf gfx1151, vllm#52631). wvSplitK ab
   M≤5 (Patch 53) ist dieselbe Baustelle. → Patch 60 (geplant).
3. **Prefill: WMMA + Takt.** Mehr SCLK = mehr Prefill; Tile-Dims als
   Vielfache von 16; RDNA4-Tuningmuster (BK=128, kpack=2, waves_per_eu 4–8,
   `.cg`) als Startpunkt für eigene Sweeps. → `bench/tune_moe.py`,
   `patches/configs/`.
4. **LDS 64 KB/CU und 256 Threads sind harte Gesetze** für eigene Kernel
   (TileLang-Indexer: threads=256 zwingend, block_N<256). → Patch 52.
5. **MALL-Budget 32 MB als Designgröße:** KV-Dtype/Block-Size so wählen, dass
   heiße Working Sets unter ~32 MB bleiben können (3,2× DRAM-Tempo).
   Evidenz für A/B fehlt öffentlich — selbst messen (UNKNOWN → eigenes
   Experiment in `bench/`).
6. **Eager ist (vorerst) Pflicht, aber Launch-Overhead angreifen:**
   `--async-scheduling` (+3,3 % Decode auf GB10 gemessen, auf gfx1151 mit
   5–6 µs Launch × hunderte Kernel/Step vermutlich mehr), dazu A/B
   `cudagraph_mode: NONE` (Inductor-Fusion ohne Capture).
   → `bench/cudagraph_ab.sh`.
7. **Kernels kompilieren einmal, überall nutzen:** Triton/Compile-Caches sind
   arch-keyed und über identische Nodes portierbar; Drift in irgendeiner
   Config/env forked den Hash still. Registry = einzige Wahrheitsquelle.
   → `scripts/warmup.sh`, `scripts/dist_cache.sh`, `TRITON_STORE_BINARY_ONLY`.
8. **Cluster: 25 GbE ist der Flaschenhals — also Verkehr minimieren.** RoCE
   ~5 µs Latenz (E810-Referenz), aber 8× weniger Bandbreite als Spark.
   PP vor TP bei Concurrency; Modelle, die auf einen Node passen, gar nicht
   shard-en. → `serve.sh`-Profile, `bench/compare_eth_vs_rdma.sh`.
9. **NIC-Praxis:** PCIe Gen4 x4 reicht für 25 GbE, aber Framework-AGESA-Bug:
   Mellanox CX-5 trainiert nur Gen3; Intel E810 ok. Gen3 x4 = 3,9 GB/s =
   25 GbE ohne Reserve. → `docs/RUNBOOK.md`.
10. **INT4-WMMA ist unbebautes Land:** `v_wmma_i32_16x16x16_iu4` existiert im
    Silizium, kein Compiler senkt darauf ab. Wer hier zuerst einen
    W4A16-MoE-Pfad baut, halbiert die Decode-Bytes nochmal gegenüber MXFP4.
    Langfrist-Härtetest, kein Sprint.
11. **Takt-Cap setzen** (2400–2500 MHz): kostet ~0 % Decode, spart ~80 W,
    verhindert Locks. → ansible base.yml.

## 6. Errata / Landminen (alle im Runbook verlinkt)

- ROCm#6165: Hang unter Dauer-vLLM, linux-firmware ≥ 20260410 (MES 0x86) Pflicht.
- ROCm#5750: Low-Power-State-festgefahren (885 MHz) auf altem Stack.
- vllm#32180: HIP-Graph-Capture-Deadlock → `--enforce-eager`.
- TheRock#4552: Triton "invalid device ordinal" im Fork → Patch 10/59.
- hip#3892: HIP meldet 15,5 GiB statt GTT → Patch 56.
- vllm#52631: m=1-Skinny-GEMM-Falle (13,7 % Step-Zeit) → Patch 60 (geplant).
- triton#9175/#9815: WMMA-Compiler-Crashes (Konfigurationen meiden).
- Framework AGESA: PCIe-Gen3-Training mit Mellanox-NICs.
- Patch-20-Konflikt (RTLD_GLOBAL vs. LOCAL, LucRoot) → `VLLM_GFX1X_ROCM_SMI_RTLD`.
