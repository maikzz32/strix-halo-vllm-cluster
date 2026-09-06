# Qwen3.8 Flash Next: Messungen vom 6. September 2026

Ziel ist eine einzelne laufende Antwort deutlich über dem bisherigen
ShareGPT-Ergebnis von etwa 48 Token/s. Die bisherigen Änderungen erreichen
dieses Ziel noch nicht. Alle hier verglichenen Modellläufe verwenden dieselben
INT4-Gewichte unter `/home/maik/qwen38_rest`, vier TP-Ranks und MTP3.

## Vergleichbare Modellmessungen

ShareGPT48, Seed 42, Temperatur 0, eine Anfrage gleichzeitig, reservierter
Kontext 262144. Die Output-Rate umfasst auch Prompt-Verarbeitung und
Request-Overhead. TPOT beschreibt die Zeit je Ausgabetoken nach dem ersten Token.

| Variante | Output-Token/s | Mittlere TPOT | Output-Tokens | MTP-Akzeptanz |
|---|---:|---:|---:|---:|
| Unveränderte Ausgangslaufzeit | 48,60 | 18,08 ms | 11839 | 54,14 % |
| MTP Local Argmax | 48,44 | 18,12 ms | 11839 | 54,14 % |
| QSA unsichtbare Tiles überspringen | 48,92 | 17,69 ms | 11839 | 54,14 % |
| QSA, aufgewärmter Wiederholungslauf | 49,56 | 17,75 ms | 11839 | 54,14 % |
| TP4 + Expert Parallelism + QSA + K640-Kernel | 46,22 | 19,09 ms | 11871 | 53,73 % |
| QSA, normale Produktion nach Neustart (7. September) | 50,30 | 17,73 ms | 11839 | 54,14 % |

Alle 48 Requests waren erfolgreich. Die vollständigen ShareGPT-Ausgaben der
QSA- und Argmax-Läufe einschließlich der QSA-Wiederholung sind identisch. Für die ursprüngliche Ausgangsmessung
liegen Benchmark-Log und Metriken vor; detaillierte Texte wurden erst im
nachfolgenden Argmax-Lauf gespeichert.

Sechs separate Streaming-Tests mit kurzen Prompts und jeweils 512 Ausgabetokens
liefern vor/nach QSA exakt gleiche Texte. Der Median der gepaarten Verbesserung
der Decode-Rate ist **2,01 %**: Coding etwa 66,9 auf 68,2, Deutsch 55,0 auf 56,2,
Analyse 64,3 auf 65,6 Token/s. Diese höheren Raten stammen von anderen Prompts
als ShareGPT und sind kein Sprung von 48 auf 68 Token/s durch die Änderung.

Der erste Coding-Request nach dem QSA-Neustart brauchte 2,27 Sekunden bis zum
ersten Token; spätere vergleichbare Requests etwa 0,3–0,4 Sekunden. Deshalb
werden Decode-Rate und Anlaufzeit getrennt dokumentiert. Der erste vollständige
QSA-ShareGPT-Lauf enthält diese Anlaufkosten. Die Wiederholung erreicht
49,56 Token/s bei 609 ms mittlerer TTFT und bestätigt den kleinen Gewinn.
Ein vollständig alternierender A/B/A-Nachweis liegt noch nicht vor.

Der normale Neustart ohne Profiler bestätigte die Konfiguration mit 50,30
Output-Token/s, 542,87 ms mittlerer TTFT und erneut exakt denselben 48 Antworten.
Die TPOT bleibt mit 17,73 ms nahe am vorigen QSA-Lauf. Der zusätzliche Anstieg
von 49,56 auf 50,30 geht vor allem mit geringerer Anlaufzeit einher; er ist kein
weiterer nachgewiesener Decode-Kernelgewinn. Gegenüber der ursprünglichen TPOT
von 18,08 ms bleibt die reine Generierungsverbesserung ungefähr 2 %.

Eine anschließende Streaming-Kontrolle enthielt eine vorübergehende Episode
von ungefähr 16 Sekunden: Die letzten Tokens einer deutschen Antwort und die
ersten Tokens der folgenden Analyse kamen langsamer, bei identischen Texten.
Diese beiden Antworten erreichten insgesamt nur 40,80 bzw. 41,41 Decode-Token/s;
die Analyse hatte zudem 5,99 Sekunden TTFT. Alle sechs Antworten der nächsten
Kontrolle lagen wieder bei etwa 56/65/68 Token/s. Die Ursache ist bisher offen;
der Ausreißer wird nicht aus dem Versuchsprotokoll entfernt. Die gleichzeitige
Hardware-Telemetrie der Wiederholung kann die frühere Episode nicht erklären.
Die [beiden vollständigen Streaming-Vergleiche](../bench/records/2026-09-07-production-streams.json)
enthalten auch diese langsamen Antworten.

Expert Parallelism war im vollständigen Modelltest langsamer. Die sechs kurzen
Streaming-Ausgaben änderten sich; ihre Decode-Rate sank gegenüber QSA ohne EP
im Median um 10,37 %. Alle 48 ShareGPT-Anfragen gelangen, aber bei veränderten
Texten und Tokenzahlen. Der dafür getestete K640-Kernel wurde auf allen vier
Nodes exakt auf seine gesicherte Originaldatei zurückgesetzt. Die laufende
Konfiguration verwendet wieder gewöhnliches TP4.

Eine kleine zusätzliche JSON-/Logikprobe bestand 11 von 12 Aufgaben. Das Modell
antwortete auf die Rekurrenz `x=2; x=3*x+1` nach drei Schritten mit 79 statt 67.
Diese Probe wurde nicht vor der Änderung ausgeführt; der Fehler kann damit
nicht der Änderung zugeordnet werden. Die erwartete Lösung bleibt unverändert.

## Ergebnisse isolierter Versuche

| Versuch | Befund | Konsequenz |
|---|---|---|
| QSA unsichtbare Score-Tiles | 32 GPU-Fälle bitgleich; kurzer Kontext etwa 95 auf 11,5 µs pro Scorer | Opt-in-Patch, vollständiger Modellvergleich oben |
| QSA Radix TopK | Bei kurzen Kontexten etwa doppelt so langsam; Gleichstände können andere Sets liefern | Nicht übernehmen |
| QSA vollständige Sparse Attention | Etwa 38–43 µs je Layer, zwölf Target-Layer | Kein großer Restengpass belegt |
| HC BF16 über BLAS | Warmer Cache täuschte Gewinn vor; wechselnde Gewichte langsamer | Nicht übernehmen |
| HC BF16 mit vorab transponierten Gewichten | Down 42,16 auf 37,38 µs, Up langsamer; Down allein grob 0,46 ms je Verify | Kleiner Kandidat, nicht im Modell getestet oder übernommen |
| HC W8 | Verlustbehaftet; isolierte Einsparung grob 1,54 ms pro Verify, zusätzlicher Speicher | Nicht übernehmen |
| GDN Verify-Geometrie | Beste Variante spart insgesamt nur etwa 17 µs über 36 Layer | Nicht übernehmen |
| Routed MoE W1-Geometrie | Fünf Varianten in vier Routerverteilungen langsamer; 60 Numerikfälle bestanden | Bestehende Geometrie beibehalten |
| Routed W2 als SIMD-GEMV pro Route | Bei disjunkten Experten bis 1,29× schneller, bei hoher Wiederverwendung etwa 2× langsamer; auch mit rotierenden Gewichten geprüft | Kein allgemeiner Ersatz; nicht übernehmen |
| RCCL Tree/LL, vier Ranks | 8/16 KiB etwa 19/37 % langsamer; Korrektheit und Teardown bestanden | Automatische Auswahl beibehalten |
| RCCL bei tatsächlichen 20 KiB | Auto/LL4 stabil 85,25–85,29 µs; LL2, LL1 und Simple langsamer | Bestehende Auswahl beibehalten |
| MTP Local Argmax | Texte gleich, kein belegter Gesamtgewinn | Default bleibt aus |

K5/K6 können den Vorteil zusätzlicher akzeptierter Tokens durch mehr Draftarbeit
und den Verlust des Dense-INT4-Skinny-Pfads bei mehr als fünf Target-Tokens
aufbrauchen. Die gemessene positionsweise Akzeptanz ist 73,66/51,57/37,20 %.
Ein großer Gewinn allein durch längere Entwürfe ist daraus nicht belegt.

## Vorhandener Lucebox-Fork

Zusätzlich wurde der lokal vorhandene Cluster-Port mit Source-Stand `60f51d1`
geprüft. Er unterstützt Qwen4Exp, native MTP und RDMA tatsächlich. Derselbe
gebaute Server sowie vollständige Q3_K_M-Referenzgewichte und das MTP-Modell
liegen auf allen vier Hosts. Seine früher dokumentierten warmen Raten betragen
33,2 / 30,6 / 25,7 Token/s für einen / zwei / vier Nodes. Die zugehörigen finalen
Rohmessungen wurden nicht gefunden; dies sind frühere Protokollangaben, keine
neuen ShareGPT-Gegenmessungen. Die verwendete Q3_K_M-Quantisierung und MTP1
unterscheiden sich von der hier gemessenen INT4-/MTP3-Konfiguration.

Der Fork lädt QSA-Indexergewichte, verwendet sie jedoch nicht im Attention-Graph;
stattdessen ruft er gewöhnliche volle Attention auf. Die kurze Funktionsprobe
belegt deshalb keine gleichwertige Sparse-Attention-Semantik bei langen Kontexten.
Diese vorhandene Alternative wurde nicht erneut als Leistungsversuch gestartet.
Die aktuellen Messungen und Änderungen bleiben auf vLLM beschränkt.

## Laufzeit und Wiederherstellung

Die native Steuerung wurde mit erfolgreichen Starts und Stopps auf allen vier
Nodes geprüft. 22 lokale Lifecycle-/Deployment-Tests bestehen. Logs, genaue
Startkonfigurationen und wiederherstellbare Deployment-Dateien liegen unter
`/home/maik/strix-halo-next` auf den Nodes.

Auf Node 1 ist die vorhandene Container-Laufzeit zusätzlich als lokales Image
`localhost/strix-halo-runtime:20260906-qsa-gated` gesichert:

```
718fdd7ceb8ad3fa889361f1cec132d966ed3eb21059fbf2898e2a51352cefb1
```

Dieses Image enthält die installierten Binärpakete und Python-Patches. Es ist
kein getesteter Ersatzcontainer und enthält keine Sicherung der extern
eingebundenen Modelle, Volumes oder Host-RDMA-Konfiguration. Das ursprüngliche
Container-Inspect und die Image-ID liegen auf Node 1 im `snapshots`-Verzeichnis.
Die vier ursprünglichen Container wurden nicht ersetzt.

Die QSA-Änderung wird mit `patches/qsa_invisible_tiles.py --apply` installiert
und erst durch `VLLM_QSA_SKIP_INVISIBLE_TILES=1` aktiv. Ohne Flag bleibt der
alte Rechenpfad erhalten. Der Patcher bewahrt eine SHA-gekennzeichnete Originaldatei
und unterstützt `--restore --apply`; Modellgewichte werden nicht verändert.

## Gemessenes GPU-Zeitbudget

Nach zwei separat reproduzierten ROCm-Profilerfehlern gelang die vollständige
Aufzeichnung auf allen vier Nodes. Je Rank sind 16 vollständige Target-Graphen
mit 2694 Kernels ausgewertet. Ein Target-Graph dauert unter Instrumentierung
etwa 40,1 ms, eine vollständige Runde einschließlich Draftarbeit etwa 48,9 ms.
Im Target entfallen ungefähr 8,7–9,1 ms auf 98 RCCL-Kernels, 8,0–8,3 ms auf
die beiden gerouteten MoE-Projektionen und 7,1 ms auf HC-Projektionen.
RCCL-Zeiten enthalten Wartezeit. Die Zeitspannen der vier Target-Ranks liegen
innerhalb von 0,15 %; ein einzelner klar langsamer Node ist damit nicht belegt.

Weitere 6,4–6,6 ms liegen zwischen Kernels. Der Profiler verändert den Queue-Pfad;
diese Lücken sind deshalb kein nachgewiesenes, vollständig nutzbares
Optimierungspotenzial. Für 20 % mehr Durchsatz müssten bei gleichbleibender
Token-Akzeptanz grob 8,1 ms pro Runde entfallen. Keine der bisher geprüften
Einzeländerungen erreicht diese Größenordnung.

Details stehen im [GPU-Zeitbudget](2026-09-06-gpu-budget.md) und in der
[Profiling-Anleitung](2026-09-06-profiler.md). Leistungsgewinne werden ohne
Profiler geprüft. Die normalen Laufzeitvorgaben enthalten keine Profiler-Flags.
Die [maschinenlesbaren ShareGPT-Messungen](../bench/records/2026-09-06-sharegpt.json)
bewahren Kennzahlen, Quellen-Hashes und Hashes der gespeicherten Antworten.
