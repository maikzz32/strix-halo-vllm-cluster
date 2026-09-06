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

### Durchsatz bei paralleler Nutzung (vier Nodes, 262k Kontext, 06.09.2026)

Alle Optimierungszahlen oben sind Einzelanfrage-Latenz (Konkurrenz 1). Im Agentenbetrieb mit
mehreren gleichzeitigen Anfragen liefert derselbe Cluster deutlich mehr:

| gleichzeitige Anfragen | Gesamtdurchsatz | Zeit je Token |
|---|---|---|
| 1 | 47,33 tok/s | 18,73 ms |
| 2 | **61,91 tok/s** | -- |
| 3 | **71,05 tok/s** | -- |
| 4 | 82,51 tok/s | 40,23 ms |
| 8 | 94,80 tok/s | 74,05 ms |
| 16 | 96,12 tok/s | 71,57 ms |
| 32 | 97,65 tok/s | 72,48 ms |

Saettigung zwischen acht und sechzehn gleichzeitigen Anfragen bei knapp 98 tok/s.

### Der Fixanteil gemessen: Speicherkopien dominieren (06.09.2026)

Torch-Profiler (in 0.29 ueber `--profiler-config`, nicht mehr ueber eine Umgebungsvariable),
acht Decode-Schritte, zwei Raenge:

| Operation | Gesamtzeit | Aufrufe |
|---|---|---|
| aten::copy_ | 309,3 ms | 1016 |
| aten::to | 309,0 ms | 800 |
| aten::_to_copy | 308,7 ms | 568 |
| kompilierter Graph | 49,0 ms | 32 |
| QSA-Attention | 22,3 ms | 16 |
| MoE mit Shared Expert | 16,3 ms | 16 |
| Kollektive | 10,3 ms | 104 |

**Korrektur nach Messung mit Aufrufstapeln:** die Kopien sind NICHT der Engpass. Von 630
Kopieroperationen dauern nur sechs laenger als 0,2 ms; der Rest sind Skalare und
Ein-Element-Tensoren (int, bool) -- Zaehler, Flags, Indizes. Die grossen Zahlen oben sind
verschachtelt gezaehlt (aten::to enthaelt aten::_to_copy enthaelt aten::copy_) und ergeben
aufsummiert 122 Prozent der Schrittzeit.

Der Fixanteil ist GPU-Rechenzeit vieler kleiner Kernel, die der Profiler nicht aufloest, weil sie
in HIP-Graphen gekapselt sind. Das Modell hat je Layer sieben getrennte GDN-Projektionen,
Hyper-Connection-Mischung ueber vier Stroeme, gruppierte Normen, Router und Experten-Gather --
bei 48 Layern mehrere hundert Kernelaufrufe je Token. Der einzige Hebel waere Kernel-Fusion;
vLLMs fusionierte Bausteine greifen auf gfx1151 nicht (AITER ist CDNA-only, der fusionierte
GDN-Decode-Kernel wird nur fuer CUDA gebaut).

Naheliegender Verdacht geprueft: `--mamba-ssm-cache-dtype bfloat16` statt des vom Modell
gesetzten float32 ergibt **39,06 statt 40,34 tok/s** -- die Akzeptanzrate faellt von 54,0 auf
52,4 Prozent, der ungenauere Zustand kostet mehr als die gesparten Konversionen bringen.

### RCCL-Kanalzahl bei vier Raengen: vier ist das Optimum (06.09.2026)

Bei zwei Raengen war die Kanalreduktion schlechter, bei vier Raengen isoliert nachgemessen
(ohne Baum-Algorithmus, Standardprotokoll):

| Kanaele | tok/s |
|---|---|
| 2 | 32,49 |
| **4** | **49,41** |
| 8 | 48,73 |
| Standard (12) | 48,51 |
| 4 + PROTO=LL | 47,04 |

`NCCL_MAX_NCHANNELS=4 NCCL_MIN_NCHANNELS=4` bringt 1,9 Prozent. In der Produktion mit 262k
Kontext: 48,04 statt 47,33 tok/s.

### Warum vier Nodes nicht doppelt so schnell sind (06.09.2026)

| Posten je Iteration | zwei Nodes | vier Nodes |
|---|---|---|
| Gewichte lesen | 12,4 ms | 6,2 ms |
| Kollektive | 3,7 ms | 9,1 ms |
| Kernel-Startlatenz | 13,5 ms | 13,5 ms |
| Summe | 29,6 ms | 28,8 ms |

Nur das Gewichte-Streaming wird durch mehr Raenge kleiner. Die Kollektive wachsen, weil der Ring
bei vier Raengen sechs Netzwerkspruenge braucht statt zwei.

**Kollektiv-Latenz isoliert gemessen** (5 KB AllReduce, vier Raenge, 400 Wiederholungen):
Standard 90,7 us / 2 Kanaele 85,4 / 1 Kanal 87,8 / 4 Kanaele 96,4 / HSA_NO_SCRATCH_RECLAIM 90,5 /
NTHREADS 128 bzw. 512: 92,1 / 90,8 / NCHANNELS_PER_NET_PEER=1 90,7 / QPS=1 ohne Split 90,0.
Kein Parameter bewegt etwas.

**GPU-Direct-RDMA waere der Hebel** (etwa 40 us, das ergaebe 59 tok/s), scheitert aber
reproduzierbar an `ibv_reg_mr_iova2 failed with error Invalid argument` -- getestet mit
NCCL_NET_GDR_LEVEL 3/5/SYS und NCCL_DMABUF_ENABLE=1. Auf dieser APU laesst sich GPU-Speicher
nicht fuer RDMA registrieren. Plattformeigenschaft, keine Einstellung.

### Korrektur des Zeitbudgets: der Entwurf kostet 40 Prozent (2026-09-06)

Die Tabelle oben war aus Einzelposten rekonstruiert und beschreibt nur den Ziel-Forward.
Direkt gemessen (vier Nodes, `qwen38_rest`, FULL_DECODE_ONLY, `SPEC_CG=prefill`, ShareGPT c=1):

| Konfiguration | Durchsatz | TPOT | Tokens je Iteration | Iterationszeit |
|---|---|---|---|---|
| `MTP_K=0` (ohne Spekulation) | 29,25 tok/s | 31,80 ms | 1 | **31,8 ms** |
| `MTP_K=3` (Produktion) | 48,2 tok/s | 20,6 ms | 2,607 | **53,7 ms** |

Der Ziel-Forward kostet also 32 ms fuer 48 Layer. Die drei Entwurfsschritte kosten die
Differenz, rund **20 ms, also ~6,6 ms je Schritt fuer einen einzigen Layer** -- elfmal
so viel je Layer wie im Zielmodell.

**Ursache:** Der MTP-Entwurf ist kein leichter Kopf, sondern ein vollstaendiger
`Qwen4ExpDecoderLayer` mit `layer_type="full_attention"` (`models/qwen4_exp/amd/mtp.py:205`):
QSA-Attention mit Indexer (Budget 2048), komplette MoE ueber 512 Experten, dazu
`fc_embedding`/`fc_hidden` (beide `gather_output=True`) und ein eigener Kopf ueber
248320 Vokabeleintraege. Und er laeuft **eager**: `speculator.py:145` setzt den
`decode_cudagraph_manager` auf NONE, sobald `VLLM_GFX1X_SPEC_CUDAGRAPH` auf `0` oder
`prefill` steht (Patch 65). Kollektive erklaeren davon nur ~2,3 ms.

Zweite Bremse im selben Pfad: `use_fused_multi_step_decode` ist aus, weil unser
QSA-Backend `QWEN4_EXP_EXP_QSA_STATE` kein `supports_draft_decode_metadata_update`
meldet. Deshalb baut `_multi_step_decode` (`speculator.py:505`) zwischen jedem Schritt
die Attention-Metadaten in Python neu auf, ausserhalb jedes Graphen. Beim Triton-Backend
ist diese Update-Methode schlicht leer -- das Flag ist nur die Zusicherung, dass die
Metadaten auf persistente Puffer zeigen. Der QSA-Bauer nutzt solche Puffer bereits
(`qsa_cache.py:590-612`) und baut per Triton-Kernel.

**Folge fuer das Ziel:** Entwurfsschritte auf 2 ms brachten 38 ms je Iteration und damit
68 tok/s; schon eine Halbierung auf 3,3 ms reicht fuer 43 ms und **61 tok/s**. Die
60-tok/s-Marke haengt an diesem Pfad, nicht an RDMA und nicht an den Kollektiven.


## Messmethodik: der erste Lauf nach einem Serverstart zaehlt nicht (2026-09-06)

Vier Benchmarks auf **einem** Serverprozess, ohne Neuladen dazwischen (TP4,
qwen38_rest, k=3, SPEC_CG=prefill, ShareGPT c=1, seed 42, T=0):

| Lauf | Governor | tok/s | TPOT | TTFT |
|---|---|---|---|---|
| 1 (kalt) | powersave | 46,97 | 18,46 ms | 557,6 ms |
| 2 | performance + C3 aus | 49,21 | 17,94 ms | 556,1 ms |
| 3 | powersave | **49,21** | 17,96 ms | 554,0 ms |
| 4 | performance + C3 aus | 49,14 | 17,96 ms | — |

Der Erstlauf liegt **4,8 % zu niedrig**, die drei warmen Laeufe streuen 0,14 %.
Konsequenz fuer alle frueheren Zahlen: die k-Serie (k=0 29,25 / k=3 48,2 /
k=4 46,68) besteht aus Erstlaeufen und ist systematisch zu niedrig. Untereinander
bleibt sie gueltig — die Steigung `31,8 ms + 7,25 ms x k` haelt —, aber der
**Ist-Stand ist 49,2 tok/s warm**, nicht 48,2.

Ab sofort: je Konfiguration ein Aufwaermlauf (`bench_k.sh`, startet den Server) und
danach der Messlauf auf demselben Prozess (`bench_only.sh`). A/B nie ohne
Rueckschalt-Lauf, also immer **A/B/A**.

## Governor und C-States: widerlegt (2026-09-06)

Die Tabelle oben ist zugleich der A/B/A-Test des CPU-Governors. Unter laufender
Decode-Last betreten die Nodes C3 rund 8.500–11.500 Mal je Sekunde und verbringen
81–86 % der Kernzeit dort, bei 350 µs Austrittslatenz und 2,3 GHz statt moeglichen
5,2 GHz. Das sieht nach einem grossen Hebel aus und ist keiner: `performance` +
`epp=performance` + C3 deaktiviert (verifiziert: C3-Eintritte fallen auf **0**)
liefert **denselben** Durchsatz wie `powersave`. Der scheinbare Gewinn von +4,8 %
war die Aufwaermung des Erstlaufs.

Damit sind auch `idle=poll`, `processor.max_cstate=1` und `pm_qos_resume_latency_us`
erledigt — sie adressieren dieselbe Ursache. Die Entwurfsschritte haengen an
Kernel-Startlatenz und GPU-Arbeit, nicht an CPU-Rechenzeit.

## Zielrechnung, aus dem warmen Stand (2026-09-06)

Aus Durchsatz, TTFT und TPOT folgt eine mittlere Ausgabelaenge von 227 Token. Bei
unveraendertem TTFT (556 ms) verlangen **60 tok/s**:

| Groesse | jetzt (warm) | noetig |
|---|---|---|
| TPOT | 17,95 ms | **14,28 ms** (−20,4 %) |
| Iterationszeit | 47,0 ms | **37,4 ms** |
| davon je Entwurfsschritt | ~6,3 ms | **~3,2 ms** |

Es bleibt bei einer Halbierung der Entwurfsschritte: der bessere Ausgangswert
verschiebt die Marke kaum, weil die konstante TTFT mitgetragen werden muss.
Der Hebel dafuer ist Patch 73 (fusionierter Mehrschritt-Entwurf), zusammen mit
Patch 65 auf `VLLM_GFX1X_SPEC_CUDAGRAPH=1`, damit alle k Schritte in **einen**
Graphen fallen.


## Patch 73 gemessen: der Entwurfs-Overhead ist NICHT die Ursache (2026-09-06)

Patch 73 hebt den fusionierten Mehrschritt-Entwurf (`supports_draft_decode_metadata_update`
am `QSAMetadataBuilder`). Auf allen vier Nodes angewandt, Anker exakt einmal getroffen,
Marker in jedem Container verifiziert. Er **wirkt** — die Zeile „Fused multi-step draft
decode is not supported" verschwindet aus dem Startlog — und er ist **korrekt**: die
Akzeptanz ist in jedem Lauf identisch (4505 Entwuerfe, 54,0 %, 2,620 Token je Iteration);
bei seed 42 / T=0 bedeutet das bitgleiche Ausgaben. Waere die In-Place-Aktualisierung
falsch, liefen die Entwurfsschritte auf veralteten Sequenzlaengen und die Akzeptanz braeche
ein.

Nur schneller ist er nicht (alle Werte warm, also zweiter Lauf):

| Konfiguration | tok/s | TPOT |
|---|---|---|
| ohne Patch, `SPEC_CG=prefill` | 49,21 | 17,95 ms |
| p1: Patch 73, fusioniert | 49,20 | 17,97 ms |
| p2: Patch 73 + `SPEC_CG=1` | 49,32 | 17,93 ms |

Bei p2 werden **79 „decode CUDA graphs (FULL)"** aufgezeichnet (Graph-Speicher 2,78 -> 4,69
GiB), alle drei Entwurfsschritte laufen also als ein Replay statt als einzeln gestartete
Kernel. Ergebnis: 0,25 % Streuung, kein Effekt.

**Damit sind beide Overhead-Erklaerungen einzeln widerlegt:**
- der Python-Metadatenaufbau zwischen den Schritten kostet nichts (p1),
- die Kernel-Startlatenz der Entwurfsschritte kostet nichts (p2).

Die 7,25 ms je Entwurfsschritt sind **echte Arbeit im MTP-Entwurfsmodell**. Das erklaert
rueckblickend, warum die alte „Startlatenz"-Bilanz nie zu den Messwerten passte. Ein
Entwurfsschritt ist nicht „ein Layer": dazu kommen `VocabParallelEmbedding`,
`fc_embedding`/`fc_hidden` (beide `gather_output=True`, je ein AllGather), der LM-Head ueber
248320 Eintraege und das Sampling — was das Zielmodell **einmal je Iteration** macht, macht
der Entwurf **je Schritt**.

**Folge:** Kein weiterer Overhead-Hebel am Entwurfspfad. Der naechste Ansatz muss die Arbeit
verkleinern oder vermeiden — `method: "ngram"` / `"suffix"` laeuft ohne Modell-Forward und
braucht laut Kostenmodell nur **1,95 Token je Iteration** (~32 % Akzeptanz) fuer 60 tok/s,
gegenueber 2,620 bei MTP. Patch 73 bleibt dennoch nuetzlich: er macht `SPEC_CG=1` erstmals
ohne Hang lauffaehig (frueher 17,1 tok/s). Produktion weiter mit `SPEC_CG=prefill`, weil die
Entwurfsgraphen 1,9 GiB kosten und nichts einbringen.


## MTP-Entwurfskopf nach INT4: gelungen und wirkungslos (2026-09-06)

Der MTP-Entwurfskopf war im Checkpoint komplett **BF16, 5,214 GB**, waehrend das Zielmodell
durchgehend INT4 ist -- er steht ausdruecklich auf der ignore-Liste. Groesste Posten:
`mtp.layers.0.mlp.experts.gate_up_proj` [512,1280,2560] = 3355 MB und `...down_proj`
[512,2560,640] = 1678 MB. Da der Entwurfsschritt 5,08 ms kostet (TP4) und fast linear mit der
Rangzahl skaliert (TP2: 8,74 ms), lag die Bandbreitenhypothese nahe.

Quantisiert (compressed-tensors pack-quantized, W4A16, group 32, asymmetrisch), dabei vom
gebuendelten 3D-Layout ins per-Experte-Layout ueberfuehrt -- das ist Pflicht, weil
`build_expert_params_mapping` Fused-Eintraege nur fuer den unquantisierten Namen
`experts.w13_weight` kennt, nicht fuer `weight_packed`. Ergebnis: 6144 Tensoren,
**1,455 GB statt 5,033 GB**.

| | BF16-Kopf | INT4-Kopf |
|---|---|---|
| Durchsatz | 49,21 tok/s | 49,34 tok/s |
| TPOT | 17,95 ms | 18,22 ms |
| Akzeptanz | 54,0 % | 53,1 % |
| Token je Iteration | 2,620 | 2,592 |
| Iterationszeit | 47,03 ms | 47,23 ms |
| **je Entwurfsschritt** | **5,08 ms** | **5,14 ms** |
| Modellspeicher je Rang | 45,13 GiB | 44,33 GiB |

Die Quantisierung greift nachweislich (0,80 GiB weniger je Rang, erwartet 0,89 GB), aber die
Entwurfskosten bleiben unveraendert. **Die Bandbreitenhypothese ist damit experimentell
widerlegt.** Der BF16-MoE-Pfad liest sparse -- nur ~10 von 512 Experten, ca. 98 MB --, weil der
INT4-GEMV-Patch `use_int4_w4a16` und `B.dtype == uint8` verlangt (`fused_moe.py:853-860`) und
sonst der stock-Triton-Kernel uebernimmt. Die vorherige Rechnung ("5 GB je Schritt, passt auf
248/288 GB/s") traf nur zufaellig plausible Werte.

**Damit sind alle drei Erklaerungen fuer die Entwurfskosten widerlegt:** Metadatenaufbau
(Patch 73), Kernel-Startlatenz (volle Graphen ueber alle Entwurfsschritte) und Gewichts-
bandbreite (dieser Versuch). Die 5,08 ms bleiben unerklaert, obwohl sie mit der Rangzahl
skalieren waehrend das Zielmodell das nicht tut.

**Zwei Fallen fuer eine Wiederholung:** (1) Die ignore-Liste enthaelt neben 30 expliziten
mtp-Eintraegen den Regex `re:mtp\..*`, und dieser traegt den ganzen Schutz -- die expliziten
Namen greifen nach `_remap_ignored_layers` (mtp.layers.0 -> .48) NICHT. Nur den Regex zu
entfernen laesst den Router quantisiert erwarten und den Start scheitern; richtig ist ein
Lookahead `re:mtp\.(?!layers\.\d+\.mlp\.experts\.).*`. (2) `/home/maik` ist im Container
schreibgeschuetzt -- Ergebnis nach /tmp schreiben und mit `podman cp` herausholen.

Produktion bleibt auf `qwen38_rest`. Der quantisierte Checkpoint `qwen38_mtpq` liegt auf allen
vier Nodes und kostet dank Hardlinks auf die unveraenderten Shards nur ~2,7 GB je Node.


## Die Iteration, vollstaendig aufgeschluesselt (2026-09-06)

Erste DIREKTE Messung statt Ableitung aus dem Durchsatz. Beide fertigen Profiler sind hier
unbrauchbar -- der Torch-Profiler stirbt beim Export, rocprofv3 haengt (zweimal
reproduziert, Exit 124 schon bei einem trivialen Matmul; ein Altlauf aus einer frueheren
Sitzung hing 18 h unbemerkt). vLLMs eingebauter `StepTimingCollector` hilft ebenfalls
nicht: er ist nur im `_dummy_run` vollstaendig verdrahtet, im Produktivpfad fehlen
`forward_end` und die drafter-Marker (model_runner.py:1731/1734 gegen 777/782/825).
Also eigene HIP-Events, ohne synchronize im heissen Pfad -- Patch 74 (Phasen im
Entwurfsschritt) und Patch 76 (Zielmodell-Forward nach Batchgroesse). Die Messung selbst
kostet nichts: 49,31 gegen 49,21 tok/s.

**Zielmodell-Forward nach Batchgroesse** (n=100 je Tabelle, ueber 11800 bzw. 4600 Forwards):

| Batchgroesse | Forward | Aufschlag |
|---|---|---|
| 1 Token (k=0) | **30,86 ms** | — |
| 4 Token (k=3, Verify) | **39,0 ms** | +8,14 ms → **2,71 ms je zusaetzlichem Token** |

**Entwurfsschritt** (ueber 9200 Schritte, Streuung < 3 %):
forward 1,22 ms (63 %) + sample 0,70 ms (36 %) + update_inputs 0,009 ms = **1,93 ms**.

**Beide Iterationen gehen exakt auf:**

| | Forward | Entwurf | Rest | Summe | gemessen |
|---|---|---|---|---|---|
| k=0 | 30,86 | — | 0,96 | 31,82 | 31,82 ms |
| k=3 | 39,00 | 3 x 1,93 = 5,79 | 1,99 | 46,78 | 46,78 ms |

Anteile bei k=3: **Verifikation 82 %, Entwurf 12 %, Rest 5 %.**
Kostenmodell: **Iteration(k) = 31,82 + 4,98 k ms**, wobei 4,98 = 1,93 Entwurf
+ 2,71 Verifikationsaufschlag + ~0,34 Rejection.

**Damit ist die bisherige Zerlegung widerlegt.** „Iteration = Zielmodell + k x Entwurf"
schlug alles, was mit k waechst, dem Entwurf zu -- auch die Verifikation. Die Entwurfskosten
schienen deshalb bei 5,08 ms zu liegen, wo tatsaechlich 1,93 ms stehen. Das erklaert
rueckwirkend, warum drei gut begruendete Entwurfsoptimierungen wirkungslos blieben (Patch 73
fusionierter Metadatenpfad; volle CUDA-Graphen ueber alle Entwurfsschritte; INT4-Quantisierung
des MTP-Kopfs): alle drei zielten auf 12 % der Iteration.

**k=3 ist nachweislich optimal.** Ein zusaetzlicher Entwurfsschritt kostet 4,64 ms. Bei
k=3→4 bringt er 0,225 Token = 0,048 Tok/ms gegen einen Durchschnitt von 0,056 -- daher war
k=4 schlechter (46,7 gegen 49,3 tok/s). Bei k=2→3 bringt er ~0,36 Token = 0,078 Tok/ms.

**Folge fuer das 60-tok/s-Ziel.** Noetig waeren 36,5 ms je Iteration bei 2,615 Token. Allein
der Ein-Token-Forward kostet 30,86 ms und ist latenzgebunden -- TP2 und TP4 liefern dort
dasselbe (31,94 gegen 31,80 ms), mehr Nodes helfen nicht. Selbst wenn Entwurf UND
Verifikationsaufschlag vollstaendig verschwaenden, blieben 32,9 ms, also ca. 67 tok/s als
theoretische Obergrenze. Der einzige grosse verbleibende Hebel ist die Akzeptanz: bei
unveraenderter Iteration braeuchte es 3,35 statt 2,615 Token (+28 %) -- eine Frage der
Qualitaet des Entwurfskopfs, nicht der Ausfuehrung.

Messwerkzeuge bleiben installiert und sind per Env aus (Default 0):
`VLLM_GFX1X_DRAFT_TIMING=N` (Patch 74), `VLLM_GFX1X_TARGET_TIMING=N` (Patch 76).
Patch 75 (Aktivierung des eingebauten Collectors) ist wirkungslos und bleibt inert.
