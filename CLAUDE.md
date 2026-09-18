# DriveSentinel — working notes for Claude

KONE Elevate'26, PS-06: real-time drive health monitoring from various-stage
voltage and current waveforms.

**Read `docs/workflow_v2.md` first.** It is the build plan; §13 holds the decisions
taken after the pre-build audit and supersedes anything above it that conflicts.
`docs/analysis_report.md` is the audit itself — the measured facts about the four
datasets live there, and measured facts override the plan.

## Non-negotiables

1. **Every number carries its protocol name.** Leaky numbers are labelled
   `(leaky reference)`, in the heading as well as the body. `docs/claims_audit.md`
   is the register; a number that is not in it does not go on a slide.
2. **Results markdown is generated from JSON**, never hand-typed. Anything not run
   reads `NOT RUN`.
3. **No tuning against test folds.** Declare the recipe before the run. The existing
   sweeps already violated this and are declared as such in `claims_audit.md` §1.2 —
   do not add to that list.
4. **Group splits only.** Every honest number is a group-holdout number. Route folds
   through `drivesentinel.common.splits.iter_checked`, which asserts group separation
   and refuses an undeclared leak.
5. **The fusion confidence floor is pre-registered** (`config.FUSION_CONFIG`,
   fixed 2026-09-18T07:19:23Z, before any multi-stage branch existed). ≥3 independent
   validation groups **and** honest macro-F1 ≥ 0.75 to raise `Fault` alone. Do not
   move it after seeing results.
6. **The v5 bearing pipeline is frozen.** `drivesentinel/{config,dataset,dsp,export,
   features,folds,model,quantize,train}.py` and `scripts/01..05` get no behaviour
   changes. Extend around them.

## Layout

```
drivesentinel/            v5 bearing pipeline (frozen)
  common/schema.py        the Recording every branch sees
  common/splits.py        group-holdout splits + leak assertions
  adapters/               the ONLY code that knows dataset quirks
  branches/               one per drive stage
scripts/01..05            the frozen bearing pipeline
scripts/branches/         one per branch
docs/workflow_v2.md       the plan (§13 = post-audit decisions)
docs/analysis_report.md   the audit: measured dataset facts
docs/claims_audit.md      every headline number, its protocol and its source
```

## Environment

`.venv/` — Python 3.12, **CPU torch** (`torch 2.14.0+cpu`). This machine has **no
NVIDIA GPU** (Intel UHD only), so `torch.cuda.is_available()` is `False` and the
"minutes on a 3050" timings in `docs/` and `folds.py:19` do not apply here. Re-time
rather than quoting them.

```bash
.venv/Scripts/python.exe -m pytest -q
```

## Dataset traps that will cost you a day each

- **D2 KAIST** — channel names vary per file (8 files use `ai1`, not `ai2`).
  Current-probe **polarity varies per file** in three patterns; uncorrected, the
  negative-sequence ratio separates acquisition batch, not fault severity. The six
  `*_0_00_*` files are **three** byte-identical pairs. Vibration filenames use
  `coil` where current uses `intercoil`. Amplitudes are probe volts, not amps
  (`NI_SensorSensitivity = 1.0`) — ratios only.
- **D3 Bacha** — never use `converted_dataset*.csv`; both apply the ACS712 2.5 V
  mid-rail offset to a DC-bus divider. `VDC`/`IDC`/`VD` are dropped: their per-class
  means span <0.8 ADC counts. `SETPOINT*`/`temperature_data`/`thermistor_calib_data`
  are thermistor calibration, not fault runs.
- **D4 Thomas** — `.mat` is MATLAB **v5** (scipy handles it; no h5py). Files 1–5 are
  the healthy motor, 6–10 the faulty one, same five scenarios in the same order. The
  label vector needs reconstruction (1000-sample window, step 500, over the ten files
  merged in order) and **must be validated** against the measured phase-current
  collapse boundaries; the collapse rule is primary either way.

## Style

Match the surrounding code: module docstrings explain *why* a choice was made and
what the alternative cost, comments cite measurements, and constants live in
`config.py` with the reasoning attached. Keep that.
