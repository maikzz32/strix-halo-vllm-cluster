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
| 7 | Kompilieren ohne Graph-Capture: `--compilation-config '{"cudagraph_mode":"NONE"}'` statt `--enforce-eager` — Inductor-Fusion bleibt aktiv, nur der (auf gfx1151 dead-lockende) Capture entfällt; plus `--async-scheduling` | Präzedenzfall vllm#44988; **gemessen 2026-08-31 auf qwen38-27b-ablit/tp4: 44,3 → 47,4 tok/s @C=1 (+7 %), 104,3 → 114,2 @C=16 (+9,4 %), 600-s-Soak ohne Hang** | vllm#44988, vllm#32180, bench/results/20260831T204158Z_cudagraph_* | **Registry-Default seit 2026-08-31** (nur dieses Modell/Profil verifiziert — bei neuen Modellen nachverifizieren); `--async-scheduling` gemessen 2026-09-03 auf qwen38-flash-next-int4/tp4: 29,27 → 29,83 tok/s (greedy, c=1), jetzt Registry-Default für diesen Eintrag |
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

**Toolbox-C-Referenz (kyuz0, 1 Node, Qwen3.8-27B, MTP an)** — zweite Latte,
diesmal gegen die offizielle Strix-Halo-Toolbox statt gegen die Spark.
Werte sind tok/s **pro Request** (nicht Aggregat); daneben der Kontrolllauf
derselben Toolbox, der die Messtreuung der offiziellen Zahlen zeigt:

| Concurrency | Toolbox C | Kontrolllauf | Abweichung |
|---|---:|---:|---:|
| C=1 | 43,55 tok/s | 43,44 tok/s | 0,3 % |
| C=8 | 16,84 tok/s | 15,10 tok/s | 10 % |
| C=32 | 7,46 tok/s | 7,99 tok/s | 7 % |

Konsequenz für die Bewertung: die offiziellen Zahlen streuen zwischen zwei
Läufen um bis zu ~10 % (C=8) — ein `BEATEN` knapp über 1,00× ist noch im
Rauschen, belastbar ist der Sieg erst ab ~1,1×. `bench/report.py` vergleicht
pro Concurrency-Stufe die beste Cluster-Zelle als Aggregat/C gegen diese
Werte.

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

- **GLM-5.3 auf gfx1151: LÄUFT inkl. MTP (2026-09-01, tp4, dev-Image `dev-glm53-flash`):**
  PR #53906 ist offiziell gfx950-gated, MTP upstream als „not supported on
  ROCm" markiert — bei uns trägt die Kette: Patch 58 (Triton-Sparse-MLA-Lane,
  re-audiert auf echtem glm-Checkout, geometry-keyed MTP-Dispatch) + Patch 61
  (AWQ-Namensremap + int4-Dequant der BF16-KDA/MLA-Projektionen) + Patch 62
  (MTP-Draft-Layer quant-frei/BF16 — der lokale int4-Requant hält Layer 45
  komplett in BF16, der Draft würde sonst die int4-Methode erben und beim
  Laden sterben; NICHT gegen den upstream-FP8-Checkpoint anwenden) +
  Kpool-Indexer-Lanes: Torch v1.5 (batched-head logits, page-granular gather)
  + Triton-Hotpaths (`patches/runtime_glm53_kpool_torch.py` /
  `runtime_glm53_kpool_triton.py`, Gate `VLLM_GFX1X_KPOOL_TRITON`; Gather
  bit-exakt, Logits max. rel. Diff 3,4e-7, Integer-Outputs exakt; die Torch-
  Lane findet einen OOB-Gather-Bug in der Upstream-Triton-Referenz) +
  z3-solver/libz3 (Dockerfile). Zahlen (Prompt 512, Aggregat C=1/8/32):
  9,9/20,7/48,2 tok/s ohne Spec; **mit MTP (num_spec=5, Rezept-Default,
  gewinnt das A/B gegen 3 = 11,6/44,3/100,4): 14,1/61,5/102,5 tok/s** —
  +42 % / ~3,0× / ~2,1×, Acceptance 1,0, Greedy-Output auf einem G1-Prompt
  bit-identisch zur Nicht-MTP-Baseline (Draft+Verify+Rejection rekonstruieren
  die Ziel-Greedy-Wahl exakt). Der Indexer-Anteil ist bei Kurzkontext klein;
  die Triton-Lane zahlt erst bei 8k+ Kontext ein (standalone bis 38× auf
  Decode-Logits vs. Torch v1.4). Offen: Qualitätscaveat (G1-Repetition auf
  Kurz-Prompts, int4-Requant-Verhalten; Needle-Test @4,5k besteht), die
  Runtime-Patches (61/62, Kpool-Lanes) sind noch nicht im Image gebacken —
  Container-Neubau braucht sie erneut.
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

### mp-Executor über Nodes (2026-09-03)
`scripts/serve_mp_node.sh`: vLLM-mp-Executor mit `--nnodes 4 --node-rank N --headless` statt Ray. qwen38-flash-next-int4/tp4, greedy c=1, MTP k=3: 29,9 → **31,27 tok/s** (+4,5 %). Follower ohne `--headless` scheitern mit „collective_rpc should not be called on follower node". Das DGX-Spark-Rezept (MiaAI-Lab, 2 Nodes: 36,4 Prosa / 55,8 Code) nutzt denselben Executor plus FULL_DECODE_ONLY-Graphen — die auf gfx1151 hängen (vllm#32180, in beiden Modi reproduziert).

### HIP-Graphen auf gfx1151 funktionieren doch (2026-09-03)
Der Deadlock aus vllm#32180 ist kein Capture-Problem: Capture läuft in allen Modi durch, der **Replay** hängt, sobald eager RCCL-Kollektive (MTP-Draft) und graph-aufgezeichnete Kollektive (Target-Verify) auf demselben Communicator abwechseln (py-spy: alle Ränge in `rocr::BusyWaitSignal` hinter `short_conv_attn.py:333`). Ohne MTP laufen Graphen sofort (15,0 → 25,9 tok/s). Mit MTP: Patch 65 (`VLLM_GFX1X_SPEC_CUDAGRAPH=0`, Speculator eager) **plus** `NCCL_GRAPH_MIXING_SUPPORT=1` **und** `NCCL_LAUNCH_MODE=GROUP` (RCCL 2.30.4 kennt beide; einer allein hängt) → **41,0 tok/s greedy, TPOT 22,4 ms, 600-s-Soak ×4 ohne Fehler** (eager mp-Verbund: 31,3). Speculator-Prefill-Graph (`=prefill`) läuft mit den Envs ebenfalls (~42 tok/s). Referenz Dual-DGX-Spark: 36,4 (Prosa).

### ROCm 10.0 (2026-09-03)
Image `dev-20260903-rocm10` (= `:dev-rocm10`, ID 4ed662efe674) aus `build-dev.yml` mit `rocm_index=https://stable.repo.amd.com/rocm/whl-next`, `tag_suffix=rocm10`: torch 2.13.0+rocm10.0.0, Triton 3.8.0, RCCL 2.30.x. Zwei Build-Fixes: Marker-Konvention der Patches 63/64 und amdsmi-Filter in den vLLM-Build-Requirements (PyPI-amdsmi 7.0.2 bindet gegen libamd_smi.so.27 nicht). Auf Hardware (mp-Verbund, Graphen, MTP k=3, Patch 65 + GEMV v1.4 zur Laufzeit): **40,7 tok/s greedy, TPOT 21,8 ms, 48/48** — Parität zu ROCm 7.14 (41,0). Kosmetik: `vllm.__version__` meldet `0.1.dev1+g33898f832` (setuptools-scm ohne Tags im Build-Checkout).

### Flash-Next auf zwei Nodes: 31,6 auf 39,8 tok/s durch Requantisierung (06.09.2026)

Der ausgelieferte Checkpoint laesst genau die Gewichte in BF16, die bei jedem Token
vollstaendig gelesen werden. Von 175 GiB Modell sind das rund 9 GiB je Forward, davon
7,7 GiB dichte Teile. Sie selbst zu quantisieren ist der einzige Hebel, der die
Groessenordnung aendert.

| Stufe | tok/s | TPOT | Akzeptanz |
|---|---|---|---|
| Ausgangszustand | 31,56 | 28,46 ms | 51,6 % |
| dichte Projektionen in 4 Bit (`tools/requant_dense.py`) | 35,52 | 25,05 ms | 52,1 % |
| zusaetzlich LM-Head (`tools/requant_lmhead.py`, Patch 69) | 39,51 | 22,56 ms | 53,3 % |
| zusaetzlich Entwurfs-Prefill-Graph | **39,78** | 22,4 ms | 53,3 % |
| dieselbe Konfiguration mit 262k Kontext | 38,77 | 22,78 ms | 53,3 % |

Wichtig: **Gruppengroesse 32.** `moe_intermediate_size` (640) geteilt durch die Rangzahl muss
durch die Gruppengroesse teilbar sein. Alle fertigen W4A16-Varianten auf HuggingFace nutzen 128
und laden deshalb weder bei TP2 noch bei TP4.

Beim LM-Head zwei Stolpersteine: der Layer heisst intern `language_model.lm_head` (Ziel muss
`re:.*lm_head$` sein), und der MTP-Kopf hat einen zweiten ParallelLMHead.

**Gemessene Sackgassen** (alle auf zwei Nodes, Basis 31,56 bzw. 39,78):

| Versuch | Ergebnis |
|---|---|
| Entwurfsgraphen (`VLLM_GFX1X_SPEC_CUDAGRAPH=1`) | 17,1 tok/s, 45 % langsamer |
| RCCL Protokoll LL, zwei Kanaele | 30,7 |
| GDN-Kernelgeometrie (BV 16, zwei Waves) | 31,95 / 31,76 |
| `HIP_FORCE_DEV_KERNARG` + hipBLASLt | 31,77 |
| `HSA_USE_SVM=0` | 31,72 |
| MTP-Kopf quantisieren | 38,79 (Akzeptanz faellt auf 50,6 %) |
| `num_speculative_tokens=7` | QSA ring capacity 12 teilt Blockgroesse 1648 nicht |
| Hyper-Connections quantisieren | dreifach verdrahtet: Merge, packed_modules_mapping und synthetisches `_input_mix_padding` |
| Expert-Parallelitaet | Deadlock beim alten, unnoetig beim neuen Checkpoint |

Der HIP-Kernel `wvSplitK_int4_g` **ist** im Image vorhanden (`torch.ops._rocm_C` laedt lazy),
der dichte 4-Bit-Pfad laeuft also bereits ueber HIP.

### Vier Nodes mit dem requantisierten Checkpoint (06.09.2026)

| Konfiguration | tok/s | TPOT | KV-Cache |
|---|---|---|---|
| vier Nodes, 32k | 47,41 | 18,96 ms | 2,27 M Token |
| vier Nodes, 262k (Produktion) | 45,30 | 18,81 ms | 3,50 M Token |
| zwei Nodes, 32k | 39,78 | 22,4 ms | -- |
| zwei Nodes, 262k | 38,77 | 22,78 ms | 0,69 M Token |
| vorher, vier Nodes, alter Checkpoint | 41,0 | 22,4 ms | 3,44 M Token |

Zeitbudget bei vier Raengen: 6,4 ms Gewichte + 9,1 ms Kollektive + 13,5 ms Fixanteil = 29 ms,
mal 2,6 akzeptierte Tokens ergibt die gemessenen 47,4 tok/s. Fuer 60 tok/s muesste die Iteration
auf 23 ms sinken; die beiden grossen Posten (Kollektive, Fixanteil) sind mit vLLM auf dieser
Hardware nicht weiter senkbar.

**node4 hatte 97 GB im Treiber gebunden** -- weder Container-Neustart, Container-Neuaufbau noch
`rocm-smi --gpureset` (auf APUs nicht unterstuetzt) halfen. Erst ein Maschinenneustart gab den
Speicher frei.

### Zwei-Node-Marke erreicht: 40,34 tok/s (06.09.2026)

Letzter Schritt: Experten-Router und QSA-Indexer mitquantisieren (0,154 GiB je Token).
vLLM erzeugt beide fest mit `quant_config=None`; Patch 72 reicht die Konfiguration durch,
gesteuert ueber `VLLM_GFX1X_GATE_QUANT=1`. Die Qualitaet leidet nicht -- die Akzeptanzrate
des Entwurfskopfs steigt von 53,3 auf 54,0 Prozent.

| Konfiguration | tok/s | TPOT | Ziel |
|---|---|---|---|
| zwei Nodes, 32k | **40,34** | 22,15 ms | 40, erreicht |
| vier Nodes, 32k | 48,51 | 18,73 ms | 60, verfehlt |
| vier Nodes, 262k (Produktion) | 47,25 | 18,73 ms | 60, verfehlt |
| Ausgangszustand, vier Nodes | 41,0 | 22,4 ms | -- |

Weg auf zwei Nodes: 31,56 -> 35,52 (dichte Teile) -> 39,51 (LM-Head) -> 39,78 (Prefill-Graph)
-> 40,34 (Router und Indexer). Insgesamt +27,8 Prozent.

Budget je Iteration bei vier Raengen: 6,2 ms Gewichte + 9,1 ms Kollektive + 13,5 ms Fixanteil
= 28,8 ms, mal 2,62 Tokens = 48,5 tok/s (trifft die Messung). Fuer 60 waeren 23 ms noetig.

**Tree-Kollektive bei vier Raengen geprueft und verworfen:** Der Ring braucht sechs
Netzwerk-Spruenge, ein Baum nur vier -- rechnerisch 95 auf 59 us je AllReduce. Gemessen mit
`RCCL_OVERRIDE_ALGO=TREE`, `PROTO=LL`, zwei Kanaelen: **41,02 statt 48,51 tok/s**, TPOT 21,29
statt 18,73. Der Ring bleibt auf dieser Strecke klar besser.
