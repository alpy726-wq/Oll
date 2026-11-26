# Oll-1

A continuous-time machine (frozen prime-frequency basis, reversible continuous-time trunk, phase memory, phase-amplitude
density output; 17.5 M parameters; trained 200 min on one H100 on real data only) packaged as a tool for sensor and
time-series data.

**What it does**
- **Forecast with uncertainty**: median and 10–90 % band of 16 imagined futures. Irregular timestamps and missing
  readings are allowed. Any unit or level works: every window is seen relative to its own recent past (sensory adaptation).
- **Anomaly alarms**: per reading, how unexpected it was (surprise = −log predicted density), alarm when surprise is
  unusually high relative to its own recent surprise.
- **Streaming**: one reading at a time, constant memory.

**Use**
- Web: https://ssdf--oll-web-web.us-east.modal.run (upload a CSV or try the real NAB examples; first request ~1 min cold start)
- Command line (CPU is enough):
  ```
  pip install torch numpy
  python -m oll forecast my_sensor.csv --horizon 96     # -> forecast_my_sensor.csv
  python -m oll alarms   my_sensor.csv                  # -> alarms_my_sensor.csv
  ```
  CSV: `timestamp,value` or one value per line.
- Python: `from oll.core import load, forecast, surprise` (weights: `models/oll-1.pt`).

**Measured** (held-out real data; see runs/gpu/main/*.json)
- ETT long-horizon protocol, ETTh2 with adaptation: MSE 0.30 / 0.34 / 0.36 / 0.44 (H = 96/192/336/720),
  below repeat-last (0.36–0.56) and one-day seasonal repeat (0.35–0.53).
- ETTh1 / ETTm1: below repeat-last; above one-day seasonal repeat (daily rhythm is a known weakness).
- NAB real anomaly series: see runs/nab_rule_selection.json (rule chosen on half the series, scored on the other half).
