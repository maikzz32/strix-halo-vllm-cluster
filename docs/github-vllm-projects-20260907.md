# GitHub-Projekte für den Strix-Halo-vLLM-Cluster

Recherche am 07.09.2026. Nur öffentliche Primärquellen gelesen; keine SSH-Verbindung, Installation, GPU- oder Netzwerklast. Auswahl für vier gfx1151/APUs mit Intel-E810/E830-RoCE, Qwen3.8-Flash-Next INT4 **asymmetrisch, Gruppe 32**, BF16-Aktivierungen und TP4/MTP3. Ziel bleibt die Geschwindigkeit einer Antwort bei gleichen Gewichten und gleicher Rechenqualität.

## Ergebnis und Reihenfolge

Nach Abschluss dieser Quellenrecherche wurde der konkrete AITER-Tile lokal
isoliert geprüft: alle drei HC-Zielformen waren langsamer als der vorhandene
Kernel. [Messung und unveränderte Produktionskonfiguration](2026-09-07-aiter-hc.md).
Auch der [kalibrierte RDMA-Backendversuch](2026-09-07-hip-rdma-ring4.md) brachte
im direkten Modellvergleich keinen Gewinn. Die folgenden Bewertungen der
übrigen Projekte sind weiterhin Quellenbefunde, keine lokalen Leistungsmessungen.

Es gibt brauchbare Bausteine, aber in den geprüften Quellen keinen nachgewiesenen, direkt einsetzbaren Ersatz, der genau diesen Qwen-Checkpoint auf vier Strix Halo über Intel-RoCE schneller ausführt. Der aussichtsreichste neue Architekturhinweis ist ein GPU-Kernel, der direkt auf registrierte UMA-Puffer zugreift und einen CPU-verbs-Progress-Thread steuert. Die gefundenen fertigen Implementierungen schließen HIP-Graph-Capture allerdings aus. Sie ersetzen unseren bereits graphfähigen Versuchsaufbau daher nicht ohne erhebliche Anpassung.

| Rang | Projekt | Konkreter Nutzen | Nächste sinnvolle Prüfung | Aufwand / Entscheidung |
|---|---|---|---|---|
| 1 | AlexKGwyn/ds4-vllm | Fused GPU staging + CPU verbs + GPU reduction auf Strix Halo | UMA-Systematomics und GPU-direkter Zugriff auf unsere bereits registrierten Hostpuffer isoliert prüfen | Hoch; Architektur übernehmen, Backend nicht kopieren und aktivieren |
| 2 | vllm-project/vllm | Aktuelle GDN-/Gate-Fusionen; Warnung vor falscher asymmetrischer INT4-Behandlung | Aktiven Modellpfad gegen die unten genannten PRs abgleichen | Niedrig bis mittel je lokalem Operator |
| 3 | ROCm/aiter | Explizite gfx1151-BF16-GEMM-Konfiguration und portable Triton/FlyDSL-Kernel | HC-GEMMs mit unseren tatsächlichen M=1/4-Formen testen | Mittel; einzelne Operatoren, kein globales AITER-Umschalten |
| 4 | ROCm/FlyDSL | RDNA3-WMMA und explizite Layouts für eigene Fusionen | Eine unveränderte BF16-HC-Operation als kontrollierten Microbenchmark portieren | Mittel bis hoch; Compiler-/Kernelfundament |
| 5 | ROCm/rocm-systems, RCCL | DIRECT_A2A-Referenz für 2–4 gfx1151-Ranks; neuere Navi-Arbeit | Numerik und Graph-Ausschluss des A2A-Codes als Vergleich prüfen | Hoch; kein fertiger Graphersatz |
| 6 | kyuz0/amd-strix-halo-vllm-toolboxes | Gepflegtes gfx1151-Patchmanifest, Intel-RoCE-Provider-Erfahrung | Nur die Differenz zu unserer bereits funktionierenden Runtime prüfen | Niedrig; Wartung/Fehlervermeidung, kein belegter TPS-Gewinn |
| 7 | hec-ovi/vllm-awq4-qwen | Native gfx1151-INT4-WMMA- und vLLM-Dispatch-Beispiele | Nur Layout-/Dispatch-Ideen prüfen | Hauptpfad ablehnen: andere Architektur und zusätzliche Aktivierungsquantisierung |
| 8 | ROCm/mori | Aktuelle GPU/RDMA-Bausteine, MORI-CCL | Intel-irdma-Unterstützung zuerst am Quellcode nachweisen | Zurückstellen; dokumentierte IBGDA-NICs passen nicht |

## 1. AlexKGwyn/ds4-vllm: der wichtigste neue Fund

[Repository](https://github.com/AlexKGwyn/ds4-vllm), geprüft auf `94f32c3ead3f597f4e7ce05f723f4ad37b7403fc` vom 04.09.2026. Der frühere Name `ds4-vllm-public` wird weitergeleitet. Originalcode Apache-2.0; separate Drittanbieterhinweise für unter anderem OdinLink-Kernelcode. [Lizenzabgrenzung](https://github.com/AlexKGwyn/ds4-vllm/blob/94f32c3ead3f597f4e7ce05f723f4ad37b7403fc/THIRD_PARTY_NOTICES.md).

Die veröffentlichten Modellzahlen sind **DeepSeek-V4-Flash, zwei Strix Halo, Thunderbolt/OdinLink, DSpark**, 300-Token-Antworten mit Temperatur 0: bei 512 Kontext ungefähr 23 Token/s Prosa und 32 Token/s Code. Sie belegen weder Qwen-Unterstützung noch einen Gewinn gegenüber unserem Test. [Benchmarkbedingungen](https://github.com/AlexKGwyn/ds4-vllm/blob/94f32c3ead3f597f4e7ce05f723f4ad37b7403fc/README.md).

Der native [`ib_ar2.hip`](https://github.com/AlexKGwyn/ds4-vllm/blob/94f32c3ead3f597f4e7ce05f723f4ad37b7403fc/container/native/ib_ar2.hip) ist interessanter als die Modellzahlen:

- Ein einziger GPU-Block schreibt die Eingabe in doppelt gepufferte `hipHostMalloc`-Sendeslots, veröffentlicht sie mit System-Fence und Doorbell, wartet auf das vom Peer geschriebene Flag und addiert direkt aus dem Host-Recvslot. Keine SDMA-Kopie und kein HIP-Hostcallback.
- Ein CPU-Progress-Thread ruft normale `ibv_post_send`/`ibv_poll_cq` auf. Das ist **kein GPU-initiierter NIC-Doorbell-/IBGDA-Pfad**.
- Tatsächlich nur ein Peer, eine QP, zwei Ranks. BF16-Arithmetik: beide Werte zu FP32, einmal addieren, nach BF16 zurück. Keine Aussage über die richtige vierfache RCCL-Summierungsreihenfolge.
- HCA-Präfix konfigurierbar, aber QP-Adressierung ausdrücklich native InfiniBand-LIDs, `is_global=0`, keine RoCE-GID-Aushandlung. Ein anderes HCA-Präfix allein macht das nicht Intel-RoCE-fähig.
- Sequenznummer und Slot werden beim Host-Enqueue bestimmt. Graph-Replay würde diese Werte wiederholen.
- Kein Destroy-API im geprüften File. Nach GPU-Timeout wird der Additionsabschnitt trotzdem ausgeführt; der Fehler wird erst beim nächsten Python-Aufruf abgefragt. Damit nicht unsere gewünschte Fehlerbehandlung.

Der [Python-Wrapper](https://github.com/AlexKGwyn/ds4-vllm/blob/94f32c3ead3f597f4e7ce05f723f4ad37b7403fc/container/rootfs/opt/venv/lib/python3.12/site-packages/ib_ar2.py) schließt Stream-Capture explizit aus. Der [vLLM-Patch](https://github.com/AlexKGwyn/ds4-vllm/blob/94f32c3ead3f597f4e7ce05f723f4ad37b7403fc/container/patches/vllm-upstream.patch) nutzt deshalb einen Eager-Break zwischen Graphsegmenten. Für unseren FULL_DECODE_ONLY-Pfad wäre das eine Architekturänderung, deren zusätzliche Launchkosten mitgemessen werden müssten.

Der [Standalone-Test](https://github.com/AlexKGwyn/ds4-vllm/blob/94f32c3ead3f597f4e7ce05f723f4ad37b7403fc/host/bench/ib_ar2_test.py) enthält standardmäßig 8 KiB, 48 KiB und 1 MiB, 100 Warmups und 2000 synchronisierte Eager-Aufrufe. Die im Printtext genannten RCCL-59-µs sind keine beigefügten Messergebnisse des neuen Backends. Keine vergleichbare TP4/20-KiB/Graph-Latenz gefunden.

**Nächster Test:** nach dem bereits laufenden Ring4-Numerikversuch eine getrennte GPU-UMA-Doorbell-Probe mit wechselnden Epochen, Datenmustern und Fehlerfällen. Unsere funktionierende Intel-RoCE-Verbindung behalten. Erst danach wäre der Verzicht auf die beiden GPU-Kopien sinnvoll zu bewerten.

## 2. vLLM upstream: gezielte PRs statt pauschalem Update

Apache-2.0, betrachteter Main `58ad1f3b8973b23943107b51230d594050b42ec3`. Qwen3.8-Flash-Next-Modellsupport [PR #53896](https://github.com/vllm-project/vllm/pull/53896) ist seit 31.08.2026 gemergt; dies allein ist kein Grund, unsere neuere gepatchte Runtime auszutauschen.

- [#55522: RDNA3 W4A16 MoE](https://github.com/vllm-project/vllm/pull/55522), offen, Head `389d099779b04e660c3b3bf17f9ff26d2f944b8c`: **asymmetrische ZeroPoints werden vom bisherigen HIP-Kernel ignoriert**. Neuer Dispatcher fällt dafür auf Triton zurück. Testbeleg ist gfx1100/TP2/symmetrischer Qwen3.6-35B-G32, nicht unser asymmetrischer Checkpoint. Diesen Kernel nicht blind erzwingen.
- [#53623: flaches GDN-QKVZ](https://github.com/vllm-project/vllm/pull/53623), offen, Head `25991b17f181b12af00c022e354ebbffe3528e66`: vier Launches werden zwei, wenn die AITER-Signatur das Layout versteht. Publizierte +2–3 % stammen von MI355X/TP8/Qwen-2.4T. Erst prüfen, ob unser qwen4_exp-Pfad überhaupt dieselbe GDN-Klasse nutzt; MTP-Mehrtokenform separat validieren.
- [#54185: fused Shared-Expert Gate](https://github.com/vllm-project/vllm/pull/54185), offen, Head `06cbbebc9b0288519da8a235daf7a1135cb7e49b`: ersetzt `F.linear` durch Plattformdispatcher. Relevant nur bei diesem konkreten FSE-Pfad; unsere quantisierten Router nicht damit verwechseln.
- [#55292: Metadatenfähigkeiten](https://github.com/vllm-project/vllm/pull/55292), offen: erfasst auch QSA, aber nur CPU-Tests, keine lokale ROCm-End-to-End-Validierung. Wartbarkeit, kein gemessener Beschleuniger.

## 3. AITER: eine konkrete BF16-Hypothese

[ROCm/aiter](https://github.com/ROCm/aiter), MIT, Main `30fc180f2b9290ad9075c13756b379bb7e92086f`. gfx1151 ist experimentell; portable Triton/FlyDSL- und einige HIP-Operatoren, nicht generell CK/ASM. A8W4/FP4-Marketingwerte betreffen andere Arithmetik/Hardware.

[PR #4814](https://github.com/ROCm/aiter/pull/4814), offen, Head `df9d0a8a6fd2151eaf8bf5f74e8cd0c983fb8320`, liefert konkrete gfx1151-A16W16-Tiles. Testhardware Radeon 8060S, ROCm 7.2/Torch 2.12.1/Triton 3.7.1. Kleine M nutzen `(32,16,256)`, zwei Warps, einen Wave/EU, ohne Split-K. Autor misst zum Beispiel `(1,128,2880)` mit 21,96 µs statt 78,24 µs `torch.addmm`; das ist **nicht** der Vergleich gegen unseren bereits spezialisierten HC-Kernel.

Nächster Test: diese Tilewahl für die tatsächlichen HC-Formen M=1/4 mit identischem BF16-Rundungsverhalten, warmem Cache und Graph-Replay gegen unseren bestehenden Kernel. Kein globales `VLLM_ROCM_USE_AITER=1`.

## 4. FlyDSL: eigene Fusionen auf Wave32

[ROCm/FlyDSL](https://github.com/ROCm/FlyDSL), Apache-2.0 (Lizenzdatei geprüft; GitHub-Metadaten melden unspezifisch NOASSERTION), Main `b81e99d69fcb2cd6100502b6360d643be898d669`. Explizites RDNA3-F16/BF16-WMMA-Beispiel und [RDNA-GEMM-Tests](https://github.com/ROCm/FlyDSL/blob/b81e99d69fcb2cd6100502b6360d643be898d669/tests/kernels/test_rdna_gemm.py). Der gfx11-Pfad unterscheidet sich von gfx120x und den CDNA-MFMA-Kernels.

Nutzen: Kontrolle über Layouts, Speicherbewegung und Fusion statt ausschließlich Triton-Autotuning. Keine gefundene Messung für Qwen-Flash-Next asym-G32 auf vier APUs. Daher eine einzelne HC-/Norm-Fusion untersuchen, nicht das ganze Servingframework umstellen. WMMA kann die Reduktionsreihenfolge ändern; gleiche Dtype ist kein Numeriknachweis.

## 5. RCCL: DIRECT_A2A existiert, aber ohne Graphen

Die Entwicklung liegt nun in [ROCm/rocm-systems/projects/rccl](https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl); das frühere Einzelrepo ist als verschoben markiert. RCCL hat eine BSD-artige Drei-Klausel-Lizenz mit separaten Drittanbieterhinweisen; Monorepo-Metadaten allein sind keine Lizenzangabe.

[PR #10043](https://github.com/ROCm/rocm-systems/pull/10043), Head `4e053ceabe2c5ad364472d16f9abf301df5e3a24`, wurde am 14.08. **ohne Merge geschlossen**. Der tatsächliche Codevorschlag weicht vom breiteren [Issue #10044](https://github.com/ROCm/rocm-systems/issues/10044) ab: gruppierte Send/Recv-Aufrufe plus lokaler GPU-Reduce, 2–4 gfx1151-Ranks/eine GPU je Node, bis 64 KiB TP4-One-shot, darüber Two-shot. **Kein Graph-Capture und keine äußere Group.** Die bis 2,44-fache Latenzverbesserung im Issue ist USB4-Full-Mesh und kein Beweis für unseren einzelnen 25G-Switch-Uplink. Volles Fan-out teilt sich hier dieselbe NIC.

[PR #11035](https://github.com/ROCm/rocm-systems/pull/11035), offen: Navi-Channel-/Ring-Tuning, aber laut Beschreibung gfx110x/gfx120x und P2P/SHM; keine veröffentlichte gfx1151-RoCE-20-KiB-Messreihe. Keine Übernahme allein wegen des Namens Navi.

## 6. Kyuz0: nützlicher Wartungsvergleich

[Repository](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes), Main `63afa25d59b70b994232c2ef01c3547fe7cdf788`, 02.09.2026. Aktuell ROCm10/Ubuntu; TP2-Ethernet/RoCE funktional geprüft, aktuelle Performance explizit **noch nicht** gemessen. Historische Zahlen sind hohe Parallelität.

Das [Patchmanifest](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes/blob/63afa25d59b70b994232c2ef01c3547fe7cdf788/docs/VLLM_PATCH_MANIFEST.md) nennt besonders einen passenden rdma-core/libirdma-v62-Overlay gegen `Unknown completion` und begrenzte AITER-Guards. Sinnvoll: gegen unsere Paketstände vergleichen, bei funktionierendem Transport nicht prophylaktisch austauschen. Keine Repository-Lizenz von GitHub erkannt; vor tatsächlicher Codeübernahme Dateiherkunft/-lizenz klären. Dokumentierte Konfigurationen lassen sich unabhängig prüfen.

## 7. hec-ovi: INT4-WMMA-Beispiel, nicht unser Decodepfad

[vllm-awq4-qwen](https://github.com/hec-ovi/vllm-awq4-qwen), Unlicense für Eigenanteile, Main `1452a9ff2bac1e422c9c1667dd5a8b4045e6d8b9`, 10.05.2026. Qwen3.6-27B, AWQ, DFlash, einzelner Strix. Native INT4-MMQ-Prefillpfade ab M≥32, Decode unverändert; HIP-Graphen werden dort ausgeschaltet. Aktivierungen werden für INT8-WMMA verarbeitet, deshalb keine unveränderte BF16-Rechenqualität für unseren Hauptpfad. Hilfreich sind Wave32-/Dispatch-Beispiele, nicht die genannte Spitzenrate 24,8 Token/s.

## 8. MORI und weitere verworfene Abkürzungen

[ROCm/mori](https://github.com/ROCm/mori), MIT, `879983bdbd8c65c52e9f79ad836a61cbffef98b6`, 07.09.2026. Die dokumentierten IBGDA-DV-Libraries sind `libionic`, `libmlx5`, `libbnxt_re` für Pollara, ConnectX und Thor2. Kein Beleg für Intel `irdma` in diesem Pfad. Host-RDMA/MORI-IO ist nicht automatisch ein passender kleiner BF16-Allreduce. Vor irgendeinem Build müsste das konkrete Intel-Backend und gfx1151-Modul nachgewiesen werden.

[ROCm/ATOM](https://github.com/ROCm/ATOM) (MIT, `5a9c2068bac9cf620425c673be5faee8e0062fc5`) hat ein vLLM-Plugin und frische Kommunikation/Fusionsarbeit. Die geprüfte Modellliste belegt nicht unseren qwen4_exp-Checkpoint auf gfx1151; Instinct-/FP8-Resultate sind nicht übertragbar. Architekturquelle, kein jetzt gerechtfertigter Frameworkwechsel.

CUDA-spezifische FlashInfer/CUTLASS/Marlin/DGX-Rezepte sind kein ausführbarer ROCm-Kernelpfad. FP8/NVFP4-/GGUF-Wechsel und die leonyurko-FP8-Emulation ändern den vorgegebenen Gewichts-/Arithmetikpfad. Die bekannten DGX-Spark-Flash-Next-Rezepte bleiben Benchmarkkontext; große Mehrbenutzer-TPS sind nicht die Antwortgeschwindigkeit einer Sitzung.

## Lokale Nachvollziehbarkeit

Gezielt heruntergeladene Originalquellen liegen unter `research/sources/alex-*` sowie `research/sources/kyuz0-patch-manifest-20260907.md`. GitHub-API-Abfragen haben Default-Branch-Commit, PR-Zustand, Merge-Status und Lizenzdateien geprüft. Keine dieser neuen Quellen wurde in den Modelldienst integriert. Die oben genannten Mikrotests sind Empfehlungen, keine bereits ausgeführten Leistungsnachweise.
