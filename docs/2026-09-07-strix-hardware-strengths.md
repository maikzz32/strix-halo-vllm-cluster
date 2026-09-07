# Strix Halo: Hardwarestaerken fuer den aktuellen vLLM-Pfad

Stand 2026-09-07. Quellen-/Dateianalyse, keine Remote-Benchmarks und keine
Serving-Aenderung. Ziel bleibt >=60 Output-Token/s ShareGPT48/C1 bei gleichen
Gewichten und gleicher Rechenqualitaet. Die zuletzt gemeldeten 54.53 Token/s
sind Kontext aus dem Haupttask; dieser Agent hat sie nicht neu gemessen.

## Belastbare Hardware-Einordnung

- **Kapazitaet und Speicherinterface:** Max+395 hat 16 Zen5-Kerne, Radeon
  8060S mit 40 RDNA3.5-CUs und LPDDR5X-8000 am 256-bit-Interface. Das ergibt
  theoretisch 256 GB/s je Node, nicht garantierte nutzbare Modellbandbreite.
  Vier Nodes haben vier getrennte Speichercontroller; daraus wird kein
  gemeinsamer 1-TB/s-Adressraum. [AMD Produktdaten](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html),
  [AMD Halo-Daten](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html).
- **UMA:** CPU/GPU teilen physischen RAM; GPUVM/GTT sind Abbildungen und
  Limits, keine separate schnelle Speicherbank. Unsere existierende
  GPU-direkte UMA-Tuerklingel plus CPU-RDMA nutzt diese Staerke bereits.
  Nochmals Hostkopien durch UMA zu ersetzen waere kein neuer Versuch.
  Daraus folgt jedoch keine pauschale Cache-/Kohaerenzgarantie fuer beliebige
  Hostpointer. [AMD RDNA3.5-Systemoptimierung](https://rocmdocs.amd.com/en/latest/reference/system-optimization/rdna3-5.html),
  [lokaler Nachweis](2026-09-07-hip-uma.md).
- **RDNA-Ressourcen:** WGPs sind CU-Paare, Wave32 ist fuer die gfx115x-
  Computeanalyse die typische Betrachtung; Wave64 existiert ebenfalls.
  HIP meldet hier 20 Multiprozessoren, nicht fehlende 20 CUs. Die aktuelle
  AMD-Spezifikation nennt fuer gfx1151 2 MiB L2 und 32 MiB L3; Angaben zu
  VGPR/LDS duerfen nicht ohne ihre CU/WGP/SIMD-Granularitaet verglichen werden.
  AMDs develop-Dokumentation ist eine bewegliche Quelle, kein eingefrorener
  Nachweis unserer installierten Runtime.
  [AMD WGP-Modell](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/conceptual/rdna/wgp.html),
  [AMD Spezifikation](https://rocmdocs.amd.com/en/develop/reference/gpu-specs.html).
- **WMMA:** RDNA3.5 hat 16x16x16-Matrixinstruktionen. BF16-Operanden mit
  FP32-Akkumulator passen zum existierenden Dequantisierungspfad. Integer-
  WMMA verlangt Integer-Eingaenge fuer A und B: IU4 allein ist kein Ersatz
  fuer W4A16. BF16-Aktivierungen in INT4 umzuwandeln verletzt die Vorgabe.
  Die Matrixfragmente benoetigen definierte Lane-Verteilung; eine verlustfreie
  Speicherpermutation kann deshalb wertvoller sein als neue Quantisierung.
  [RDNA3.5 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna35_instruction_set_architecture.pdf),
  [AMD WMMA-Intrinsics und Fragmente](https://gpuopen.com/learn/wmma_on_rdna3/).
- **Zen5/AVX512:** Die eigenen F/BW-Intrinsics liefern bereits exakt
  gerundete BF16-Reduktion. Lokale Messung ist hier staerker als ein
  generisches TOPS-/CPU-Datenblatt: etwa 3.5 statt 9 us CPU-Reduktion.
  Ein weiteres pauschales AVX512-Flag ist kein neuer Hebel.
  [Nachweis](2026-09-07-uma-avx512.md).
- **NPU:** XDNA2 ist ein separater Beschleuniger mit eigenem Runtime-/Modell-
  Pfad. AMD beschreibt Linux-NPU-Inferenz, aber daraus folgt keine vLLM-
  Unterstuetzung fuer unser Qwen-Modell, dessen GDN/MTP-Zustaende und
  asymmetrisches INT4. Keine belastbare Drop-in-NPU-Entlastung gefunden.
  [AMD Linux-NPU-Flow](https://ryzenai.docs.amd.com/projects/WinML/en/stable/llm_linux.html).

## Drei neue, priorisierte Hypothesen

### 1. Verlustfreies W1-Packlayout passend zum WMMA-Eingang

Der aktuelle Kernel liest gepackte K-Bytes, trennt beide Nibbles, interleaved
sie, expandiert GS32-Skalen/Zero-Points und transponiert nach BF16. Der neue
Versuch waere eine **einmalige bijektive Umordnung gepackter Gewichte und
Metadaten beim Laden**, die diese Lane-/Layoutarbeit vereinfacht; kein
vollstaendiges BF16-Entpacken des Modells. Arithmetic bleibt unveraendert:
FP32 `(nibble-zp)*scale`, BF16-Rundung, gleiche Dot-Kacheln/K-Reihenfolge,
gleiche Split-Reduktion. Nicht mit laufendem W1-Workgroup-Reordering oder
dem bereits negativen HC-Transpose-Test verwechseln.

Quellanker: `research/moe-w1-order-base.py:749-790`, besonders `interleave`
und `trans`; [Original-Triton-MoE](https://github.com/vllm-project/vllm/blob/33898f832/vllm/model_executor/layers/fused_moe/fused_moe.py),
WMMA-Fragmente oben. Zuerst kompiliertes ISA/TTGIR lesen: Falls bereits
keine teure Konvertierung/Shuffle/LDS-Runde existiert, Hypothese verwerfen.
Triton-Layoutaenderungen muessen die Ausgabeparitaet trotz gleicher tl.dot-
Schreibweise separat beweisen.

**Budget:** W1+W2 lagen im aelteren Profil zusammen bei etwa 8 ms pro
Targetrunde; W1 allein ist nur ein Teil davon. Arbeitsziel fuer den
Entscheidungstest: wiederholbar mindestens 5 us pro W1-Aufruf bei breit
verteilten Experten; 48 Aufrufe ergeben rechnerisch 0.24 ms, keine TPS-
Prognose. Groessere Vorteile erst nach tatsaechlicher Messung ansetzen.

**Kleinster Test:** ein kompiliertes aktuelles W1 M4/N320/K2560 plus ein
verlustfrei umgeordnetes Pendant; 24 rotierende volle E512-Baenke mit
reuse10/overlap25/disjoint40, Daten-Rekonstruktion bitgenau, alle Outputs
bitgenau, Graph-Replay mit geaenderten Inputs. Packkosten separat erfassen.
Erst nach Gewinn und Paritaet zum Modellvergleich.

### 2. Shared-Expert-Epilog und Output-Alias innerhalb des MoE-Ops

Die aktuelle opaque MoE-Grenze verhindert einige automatische Fusionen.
Der Shared-Zweig berechnet skalaren Gate, Sigmoid und Broadcast-Multiplikation;
der Routed-Zweig kopiert sein fertiges Ergebnis wegen der derzeitigen
Output-Alias-Policy. Ein gfx1151-spezifischer Adapter kann den **fertigen**
Routed-Output direkt in den vorgesehenen Ausgabepuffer schreiben und die
Shared-Gate-Epilogarbeit zusammenfassen. Alle bisherigen BF16-Zwischenrundungen
und Reduktionsreihenfolgen muessen als explizite Operationen erhalten bleiben.
Keine pauschale Lockerung der globalen AITER-Guards.

Quellanker: `research/moe-profiler-map.md`, Schritte 5/6/13/14 und Abschnitt
Output-Copy; [vLLM modular_kernel.py](https://github.com/vllm-project/vllm/blob/33898f832/vllm/model_executor/layers/fused_moe/modular_kernel.py).
Das ist **nicht** die schon getestete W2-Splitreduktion+Expertensumme-Fusion.

**Budget:** Zunaechst aktuelle Graphknoten zaehlen; sind diese Operationen
bereits fusioniert/eliminiert, abbrechen. Bei zwei vermeidbaren 2-4-us-
Operationen pro Block waeren 48 Bloecke rechnerisch 0.19-0.38 ms. Diese
Spanne ist ein Testbudget, keine Messung. 20-KiB-Kopien sparen allein kaum
DRAM-Zeit; relevant ist Dispatch/Abhaengigkeitslatenz auf nur 20 WGPs.

**Kleinster Test:** extrahierten aktuellen Shared-/Finalize-Pfad gegen
Adapter mit M1/M4/M8, realen Strides und separaten Eingabe-/Ausgabecanaries
vergleichen, inklusive Graphen und Input-Aliasing-Pruefung. Zuerst
GPU-Zeit/Knoten; kein voller Modellneustart ohne messbaren Gewinn.

### 3. Zen5-Fortschrittsthread gezielt platzieren

Der AVX512-RDMA-Fortschrittsthread spinnt; die verbleibende CPU-seitige
Collective-Wartezeit war rund 35 us. Noch kein lokaler Bericht zeigt ein
kontrolliertes CPU-Affinitaets-A/B. Ein opt-in Thread-Affinitaetsparameter
koennte Migrationen oder SMT-Konkurrenz vermeiden. Das ist keine Behauptung,
Node16 sei thermisch gedrosselt: die vorhandene Hardwareanalyse belegt
gerade keine solche Ursache. CPU- und GPU-Arbeit teilen zudem Ressourcen;
weitere CPU-GEMMs parallel zur GPU sind deshalb nicht automatisch hilfreich.

Quellanker: `tools/hip_uma_backend.hip` Fortschrittsthread,
`research/cpu-rdma-tp4-results-20260907.md` (ohne Affinitaetsoverride),
`research/hardware-audit-20260906.md`; existierender AVX512-Bericht oben.
Topologie zuerst aus den echten CPU-cache-shared_cpu_list/SMT/NIC-sysfs-
Daten ableiten; keine willkuerliche CPU-ID und keine globale IRQ-Umschaltung.

**Budget:** 1-5 us weniger je 97 Target-Collectives waeren 0.10-0.49 ms.
Der obere Rand ist nur eine Entscheidungsspanne. Die Netzserialisierung
verschwindet nicht: drei 20-KiB-Sends brauchen bei idealen 25 Gbit/s schon
rund 19.7 us je Sender, noch ohne Protokoll und Peer-Abstimmung.

**Kleinster Test:** isolated 32-bank UMA/PyNccl-Paritaet, dann alternierend
ungebunden/ruhiger physischer Kern/anderer LLC-Bereich mit gleichem Backend,
Median UND Tail pro Rank sowie Thread-Migrationen erfassen. Gleiche Last,
keine konkurrierenden Modellanfragen; Affinitaet nach Test vollstaendig
zuruecksetzen. Nur nach kollektivem Gewinn Modellvergleich.

## Was nicht erneut als neue Optimierung zaehlt

Diese konservativen Einzelbudgets beweisen zusammen noch keinen Weg von
54.53 auf 60 Token/s. Dafuer sind rund 10.0% mehr Gesamtdurchsatz bzw.
9.1% weniger Gesamtlaufzeit erforderlich; Prefill/TTFT zaehlen im Benchmark
mit. Eine neue kritische-Pfad-Messung nach dem W1-Test muss entscheiden,
welcher groessere Restblock die Luecke tatsaechlich schliessen kann.

TP2, EP4, W1-Split-/Tile-Sweeps, W1-Cachehints, W2-Serialisierung, AVX512,
direkte UMA, HC-Transpose/verlustlose Packung, GDN-Fusion und AITER-HC wurden
bereits untersucht. W1-Workgroup-Reordering ist aktuell im Modelltest und
soll zuerst abgeschlossen werden. Weder AITER-Masterflag noch CDNA/gfx950-
Configs noch Integer-WMMA sind automatische gfx1151-Beschleuniger.

Das historische HARDWARE_GFX1151.md wurde gezielt bei Eager-Pflicht,
256-Thread-Grenze und INT4-WMMA/W4A16 korrigiert. Andere dortige Community-
Zahlen sind nicht durch diese Recherche neu validiert. Insbesondere kein
Takt-Cap oder IOMMU-/Firmware-Wechsel ohne konkretes neues Messergebnis.


## Lokale Topologie und abgeschlossener W1-Test

Die read-only sysfs-Abfrage aller vier Hosts bestaetigt dieselben zwei CPU-L3-Gruppen: `0-7,16-23` und `8-15,24-31`. Die SMT-Paare sind den gespeicherten `thread_siblings_list` zu entnehmen; diese Gruppen sind CPU-Caches, keine GPU-Caches. Alle logischen CPUs0-31 sind fuer den abgefragten Hostprozess erlaubt. Das belegt noch keine Affinitaet des Container- oder Kommunikationsthreads.

Die RDMA-Geraete melden `numa_node=-1` und `local_cpulist=0-31`. Daraus ergibt sich keine bevorzugte CPU-L3-Gruppe fuer die NIC. Es wurde keine Affinitaet geaendert. Vor einem Test muessen der konkrete Progress-Thread und dessen erlaubte CPUs festgestellt werden.

Der W1-Modellvergleich ist inzwischen abgeschlossen: 54.9197 / 56.0396 / 54.8789 Output-Token/s vor / Kandidat / nach, alle48 Antworten exakt. Details und Einschraenkung der zeitlichen Kontrollnaehe stehen in [W1-Bericht](2026-09-07-moe-w1-order.md). Damit verbleiben gegenueber dem Kandidaten rund7.1% mehr Gesamtdurchsatz bis60 Token/s.
