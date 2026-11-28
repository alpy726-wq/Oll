---
license: apache-2.0
tags:
  - time-series
  - forecasting
  - probabilistic-forecasting
  - continuous-time
  - tokenizer-free
  - streaming
pipeline_tag: time-series-forecasting
---

# Oll-1

A continuous machine for real-valued signals, from the paper *Sequence Modelling Without a Vocabulary* (A. Keser).
Every value is encoded by a frozen prime-frequency basis b(x)_k = p_k^{-1/2} e^{2πi x√p_k} (no tokenizer, no embedding
table); a reversible stack of continuous-time oscillator blocks advances by the real time gap between events; an
associative phase memory is written in one pass; the output is a probability density with a closed-form normaliser.

- Parameters: 17,508,438 (R = 1024, 8 reversible blocks, 8 oscillator modes, memory bank 512)
- Training: 200 minutes on one NVIDIA H100, real data only (weather, electricity, markets, speech, books, FineWeb-Edu,
  ten public forecasting collections)
- File: `oll-1.pt` (fp16 weights + config)

## What it does

- **Forecast with uncertainty** for any univariate series: median and 10–90 % band of sampled futures. Irregular
  timestamps and missing readings are allowed; each window is seen relative to its own recent past, so any unit or
  level works.
- **Per-reading surprise** (−log predicted density). *Experimental*: as an anomaly detector it is at chance level on NAB.
- **Streaming** one reading at a time with constant memory; streaming equals the full pass to 2×10⁻⁶.

## Measured on held-out real data

| | result |
|---|---|
| CRPS vs a no-learning Gaussian | weather 0.038 vs 0.210, electricity 0.094 vs 0.230, speech 0.039 vs 0.068, held-out series 0.054 vs 0.087, markets 0.093 vs 0.092 |
| ETTh2 (standard long-horizon protocol, adapted) MSE, H = 96/192/336/720 | 0.299 / 0.338 / 0.362 / 0.435 (repeat-last 0.362–0.559, one-day repeat 0.346–0.529) |
| ETTh1 / ETTm1 | below repeat-last, above one-day repeat |
| 80 % band coverage (ETTh1 oil temperature) | 73 % |
| text, held-out | 2.11 bits per byte (books), 2.03 (FineWeb-Edu) |

Protocol deviation for ETT: test windows every 24 h instead of every step. Full tables, negative results and code:
the accompanying repository (`paper/oll_paper.pdf`).

## Use

```bash
pip install torch numpy
python -m oll forecast my_sensor.csv --horizon 96   # -> forecast_my_sensor.csv (median, q10, q90)
python -m oll alarms my_sensor.csv                  # experimental
```

```python
from oll.core import load, forecast
m = load("models/oll-1.pt")
f = forecast(m, values, timestamps, horizon=96)     # dict(median, q10, q90)
```

CSV: `timestamp,value` or one value per line. CPU is enough (about 45 s for a 48-step forecast with 8 paths).

## Limits

Small scale (17.5 M parameters, 200 GPU-minutes); daily rhythm is flattened in long fed-back roll-outs; markets are not
predicted better than persistence; one-step surprise is not an anomaly detector.

# Oll-Reader (`oll-reader.pt`, 6.6 MB, CPU)

Answers questions from your documents. The documents are written once into a phase memory addressed by meaning
(frozen prime basis, no training); a small linear reader trained on SQuAD v1.1 (45 min, CPU) picks the answer.
SQuAD v1.1 validation (10,570 questions): exact match 33.2 %, F1 42.9 %; confidence >= 0.6: 71 % correct, >= 0.8: 84 %.
Below confidence 0.3 it says it is not sure. English only; the answer must be written in the documents.

```bash
python -m oll ask notes.txt                    # interactive
python -m oll ask notes.txt --q "When does the new office open?"
```
