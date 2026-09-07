# AITER #4814: konkreter HC-Microtest, nur Quellcodeprüfung

Die anschließende [isolierte lokale Messung](2026-09-07-aiter-hc.md) ist separat
dokumentiert. Dieser Bericht hält die vorausgehende Quellenprüfung fest.

Stand 07.09.2026; kein SSH, Build, Import auf GPU oder Benchmark ausgeführt. Geprüfter PR-Head: `df9d0a8a6fd2151eaf8bf5f74e8cd0c983fb8320` ([PR](https://github.com/ROCm/aiter/pull/4814)). Der PR ändert ausschließlich `aiter/ops/triton/configs/gfx1151/triton/gemm/gemm_a16w16/DEFAULT.json`. Diese Tile-Konfiguration kann isoliert getestet werden; ein Upgrade der gesamten Produktionsinstallation ist dafür nicht erforderlich, sofern die vorhandene Kernelsignatur kompatibel ist.

## Dateien und Aufruf

- Öffentlicher Wrapper: [`aiter/ops/triton/gemm/basic/gemm_a16w16.py`](https://github.com/ROCm/aiter/blob/df9d0a8a6fd2151eaf8bf5f74e8cd0c983fb8320/aiter/ops/triton/gemm/basic/gemm_a16w16.py), `gemm_a16w16(x, w, bias=None, dtype=torch.bfloat16, y=None, config=None, activation=None, skip_reduce=False, kernel_type="bandwidth_bound", backend=None)`.
- Tatsächlicher gfx1151-Kernel: [`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py`](https://github.com/ROCm/aiter/blob/df9d0a8a6fd2151eaf8bf5f74e8cd0c983fb8320/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py), `_gemm_a16_w16_kernel`.
- Konfigurationsberechnung: `aiter/ops/triton/utils/gemm_config_utils.py`, `compute_splitk_params(config, K)`. Wichtig: der explizite `config`-Pfad des Wrappers ruft diese Funktion nicht selbst auf. Die fehlende `SPLITK_BLOCK_SIZE` muss ergänzt werden.

Beispiel für vorab angelegte Tensoren, noch nicht ausgeführt:

```python
import torch
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params

# x: BF16 [M,K], w: BF16 [N,K], y: BF16 [M,N], alle auf demselben GPU-Gerät.
# Zunächst original row-major weights verwenden, nicht w.T übergeben:
# der Wrapper transponiert intern als View.
cfg = dict(
    BLOCK_SIZE_M=32, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256,
    GROUP_SIZE_M=1, num_warps=2, num_stages=2, waves_per_eu=1,
    matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1,
)
cfg = compute_splitk_params(cfg, x.shape[1])  # hier SPLITK_BLOCK_SIZE == K
result = gemm_a16w16(
    x, w, bias=None, dtype=torch.bfloat16, y=y,
    config=cfg, activation=None, skip_reduce=False, backend="triton",
)
assert result.data_ptr() == y.data_ptr()
```

Konfiguration und Tensoren außerhalb der Messung anlegen; einmal kompilieren/wärmen, erst dann Graph-Capture. Der Wrapper ist über `torch_compile_guard` als Custom-Op abgesichert, aber das ist kein Beleg, dass genau unsere Torch-/Triton-Version und diese Formen erfolgreich capturebar sind.

## Unsere Formen und die tatsächliche Tile-Arbeit

| Fall | X | W | Ergebnis | Grid mit PR-Tile | K-Schritte |
|---|---|---|---|---|---|
| Merged HC down | `[4,10240]` | `[336,10240]` | `[4,336]` | 21 Programme | 40 × 256 |
| Final HC down | `[4,10240]` | `[320,10240]` | `[4,320]` | 20 Programme | 40 × 256 |
| HC up | `[4,320]` | `[10240,320]` | `[4,10240]` | 640 Programme | 256 + 64 gültige Werte |

N336 ist **320 Lowrank + 4 Injection + 12 Nullpadding**, kein auf320 abrundbares Padding. 96 Merged-down-Module plus ein finales N320-Modul; 97 Up-Module. Der M-Wert bleibt4. Im Kernel wird der BM32-Block bei M4 über `% M` auf vorhandene Zeilen abgebildet und nur die vier gültigen Ausgabezeilen gespeichert. Dadurch berechnet der generische Tile mehr Zeilen als nötig. Die 20/21 Down-Programme bieten zudem weniger Programme als40CUs; ein Gewinn gegenüber unserem skinny Kernel ist keineswegs sicher. Up hat einen maskierten K-Tail, der explizit mitgetestet werden muss. Das Gleiche gilt separat für M1-Draftformen.

## Numerik: keine Bitgleichheit aus dem Datentyp ableiten

Dieser Pfad lädt BF16-Operanden, verwendet einen **FP32-`tl.dot`-Akkumulator**, addiert K-Blöcke der Reihe nach und konvertiert erst beim Store nach BF16. Mit `NUM_KSPLIT=1` gibt es keinen separat materialisierten Split-K-Partialtensor. Kein INT8/FP8 und keine Gewichtsquantisierung nötig. Auf gfx1151 ist die konkrete WMMA-Lowering/Reihenfolge compilerabhängig und erst im kompilierten Code belegbar.

Unser Baseline-Aufruf ist `vllm.model_executor.layers.utils.rocm_unquantized_gemm_impl(x,w)` (wvSplitK). Der neue dot-Tile hat keine bewiesene identische Reduktionsreihenfolge. Quellenchecks können daher gleiche BF16-Eingaben/Gewichte bestätigen, **nicht gleiche Outputbits**. Vor jedem Timing pro Matrix bitweise Abweichungen, Max-/RMS-Fehler gegen FP32-/FP64-Referenz, Endlichkeit und die12 Nullpadding-Ausgaben erfassen. Bias auslassen: der öffentliche AITER-Benchmark verwendet Bias und startet den Akkumulator damit; das entspricht unseren HC-GEMMs nicht. `activation=None` lassen, HC-SiLU nicht unbemerkt in FP32 vor die bisherige BF16-Rundungsgrenze fusionieren.

Die veröffentlichten AITER-Tests akzeptieren u.a. `atol=0.1, rtol=0.01`; sie sind kein strenger Bitparitätsnachweis. Ein schneller Microtest mit anderen Outputbits darf nur als Kandidat mit offener Qualität gelten, nicht als erfüllte Nutzeranforderung.

Zusätzliche Quellcodeauffälligkeit: der Kernel enthält `tl.assume(stride_ck > 0)`, während der Wrapper bei `NUM_KSPLIT=1` dafür0 übergibt. Das kann bei einer Spezialisierung ohne aktiven Split-K wirkungslos eliminiert sein, ist aber ein widersprüchlicher Optimierungshinweis. Vor Verwendung mit unserer Triton3.8-Version einen kleinen begrenzten Smoke-Test und generierten IR-Code prüfen; Erfolg älterer PR-Tests nicht als Kompatibilitätsgarantie behandeln.

## Cold-weight- und Graphvergleich

Den vorhandenen Ansatz aus `tests/test_hc_w8_gpu_cold.py` übernehmen:24 verschiedene, identische BF16-Gewichtsmatrizen je Methode, gleiche Eingabe, getrennte Ausgabe je Matrix, zyklisch in einem Graph. Down-Pool157,5MiB, Up150MiB. Finale N320-down-Form separat messen. Das überschreitet den Einzelmatrix-Cache-Arbeitssatz; ohne Countermessung lediglich **rotierender großer Pool**, kein Nachweis konkreter Cachemiss-Raten.

Baselines jeweils aktuell neu messen: original `rocm_unquantized_gemm_impl`, AITER mit obiger Tilewahl. Optionale zweite Referenz ist der vorhandene transponierte BF16-down-Kandidat; Layoutumwandlung dabei außerhalb Timing und ihre Speichermehrkosten separat. Alte Zahlen42,163µs down/35,302µs up sind Kontext, kein gleichzeitiger Kontrolllauf.

Keine mehrfache Wiederholung derselben einzelnen Weight-Matrix als alleinigen Beleg verwenden. Vorallokation/JIT ausschließen, warmen Eager-Smoke und Graphoutput prüfen, danach fünf Samples mit20Replays ×24Matrizen wie im vorhandenen Harness. Nur eine Form ändern, keine globale Tuninginstallation. Die AITER-Benchmarkquelle benutzt `do_bench(...,warmup=25,rep=100)` bzw. optional `do_bench_cudagraph`; diese Zahlen sind Triton-Zeitbudgets in Millisekunden, **keine festen25/100Iterationszahlen**, auch wenn die PR-Beschreibung sie so bezeichnet.

## Versionen und Abhängigkeiten

PR-Messung: ROCm7.2, Torch2.12.1+rocm7.2, Triton3.7.1. Unser Stack: ROCm10/HIP7.15, Torch2.13, Triton3.8; hier noch ungetestet. Die PR-Head-Requirements pinnen weder Torch noch Triton, sondern erwarten passende ROCm-Imagepakete. `flydsl==0.3.1` steht im allgemeinen Requirementsfile; **dieser ausgewählte GEMM-Pfad importiert oder verwendet FlyDSL nicht**. Gluon ist in dem Wrapper nur für gfx1250 freigegeben und sollte explizit mit `backend="triton"` ausgeschlossen werden. Ein FlyDSL-Upgrade ist für diesen Test nicht begründet.

MIT-Lizenz laut Wrapperheader. Heruntergeladene Quellen unter `research/sources/aiter4814-*`; der misslungene Abruf von `_triton/config_utils.py` ist ein404-Artefakt und keine benötigte Datei. Die tatsächliche Konfigurationsdatei ist `gemm_config_utils.py`.
