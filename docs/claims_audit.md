# Claims audit

Every headline number, its protocol, the file that produced it, and the command that
regenerates it. Per `docs/workflow_v2.md` §9 rule 5.

Leaky numbers are labelled **(leaky reference)**. Anything not yet run reads `NOT RUN`.

---

## 1. Pre-registered decisions

Decisions fixed **before** the runs they govern, so that they cannot be chosen after seeing
results. Changing any of these later invalidates the metric it governs.

### 1.1 Fusion branch confidence floor

| Field | Value |
|---|---|
| **Fixed at** | **2026-09-18T07:19:23Z** |
| Repo state | `7344a436b724a08058ca6496d9d1d5936e17226b` (`main`) |
| Branches trained at this time | **none** — S4, S1, S2/S3 had not been implemented |
| Authority | `docs/workflow_v2.md` §13.6 |

A branch may raise `Fault` on its own **only if both** conditions hold:

1. it has **≥ 3 independent validation groups**, and
2. its honest (non-leaky) **macro-F1 ≥ 0.75**.

Otherwise the branch is **indicative**: it may raise `Warning` and contribute to the evidence
string, but may never raise `Fault`.

Implemented in `drivesentinel/config.py` as `FUSION_CONFIG["fault_authority"]`. The dashboard
renders evidence tiers and keeps waveform and spectrum panels live for indicative branches.

**This floor was set before any multi-stage branch was trained and must not be changed after
seeing branch results.** If it is ever changed, the change, its date and its reason go here,
and every metric measured under the old floor is re-labelled.

Expected consequences at the time of fixing (recorded so that the outcome cannot be
retrofitted):

| Branch | Groups available | Expectation |
|---|---|---|
| S5 bearing | 29 bearings | meets the floor |
| S4 winding | 3 motors | meets the group test; macro-F1 unknown |
| S1 supply | 2 motors | **fails the group test** → indicative |
| S2/S3 telemetry | 1 run per condition | **fails the group test** → indicative |

### 1.2 Declared selection-on-test

`artifacts/runs/sweep4_v1.json` and `artifacts/runs/sweep5_v2.json` are recipe sweeps scored
**on the leave-one-bearing-out folds themselves**. The configuration in
`drivesentinel/config.py:110-144` (feature set v2, `val_bearings_per_class: 0`, strong
augmentation, `n_seeds: 1`) was chosen from those results.

Under `docs/workflow_v2.md` §9 rule 3 this is **selection on test**, and it is declared as
such here rather than presented as an independent validation.

**Mitigation, and why the effect is believed small:** the differences the sweep was used to
choose between sit inside the measured run-to-run noise floor. `drivesentinel/config.py:131-137`
records the three-way comparison (v2+strong 0.8018, v1+strong 0.7823, v2+weak 0.7611) against
a ±0.0103 three-repeat standard deviation, and `docs/accuracy_ceiling.md` records that an
earlier two-draw comparison against a ±0.11 noise floor reached the **opposite** conclusion.
The one change that is larger than the noise floor — removing the inner validation split
(0.7283 → 0.7823, std 0.108 → 0.0018) — is a variance fix with a mechanical explanation
(`drivesentinel/folds.py:64-80`), not a tuned hyperparameter.

**How to read the LOBO number in light of this:** 0.8018 ± 0.0103 is an optimistic estimate
by an amount not separately measured. A clean estimate would need an outer loop, or a
recipe frozen before any sweep. Neither was done.

---

## 2. Bearing pipeline (S5, frozen)

| Claim | Value | Protocol | Produced by | Recorded in | Regenerate |
|---|---|---|---|---|---|
| Window accuracy | **0.8018 ± 0.0103** | LOBO, 29 folds, pooled, 3 repeats | `scripts/experiments/repeat.py` | `artifacts/runs/repeat_v2.json` → `summary.window_acc` | `python scripts/experiments/repeat.py` |
| Macro-F1 | **0.7944 ± 0.0114** | same | same | same → `summary.macro_f1` | same |
| Per-recording accuracy | **0.8332 ± 0.0056** | same | same | same → `summary.recording_acc` | same |
| Per-bearing mean | 0.7983 | LOBO, single run | `scripts/02_train_lobo.py` | `artifacts/runs/lobo_summary.json` → `per_bearing_acc_mean` | `python scripts/02_train_lobo.py` |
| Pooled window accuracy | 0.7979 | same | same | same → `pooled.window_acc` | same |
| Aggregation @ 40 windows | 0.8626 | LOBO + pooling | same | same → `aggregation_curve` | same |
| Majority baseline | 0.4153 | — | same | same → `pooled.majority_baseline` | same |
| v1 (2-channel) comparison | 0.7823 | LOBO, 3 repeats, feature set v1 | `scripts/experiments/repeat.py` | `artifacts/runs/repeat_noval.json` | `DRIVESENTINEL_FEATURE_SET=v1 python scripts/experiments/repeat.py` |
| Accuracy **(leaky reference)** | **0.9865 ± 0.0024** | stratified 5-fold over **windows**, feature set **v2** | `scripts/experiments/shuffled_benchmark.py` | `artifacts/runs/shuffled_benchmark.json` → `accuracy` | `python scripts/experiments/shuffled_benchmark.py` |
| Macro-F1 **(leaky reference)** | 0.9847 | same | same | same → `macro_f1` | same |
| Window accuracy **(leaky reference)** | **0.9184** | shuffled windows, feature set **v1** | `scripts/experiments/task_variants.py:38` | `artifacts/runs/task_variants.json` → `3class_shuffled_LEAKY.window_acc` | `DRIVESENTINEL_FEATURE_SET=v1 python scripts/experiments/task_variants.py` |
| Binary LOBO window accuracy | 0.7982 | LOBO, binary healthy/damaged | `scripts/experiments/task_variants.py` | `artifacts/runs/task_variants.json` → `binary_lobo.pooled.window_acc` | same |
| Speed-estimate error | median 0.022 %, p99 0.208 % | current-derived vs tachometer, all recordings | `scripts/01_build_cache.py` | `artifacts/cache_report.json` | `python scripts/01_build_cache.py` |
| INT8 vs float argmax agreement | **99.90 %** (n=2,048) | calibration-disjoint windows | `scripts/03_export_int8.py` | `artifacts/int8_export/verification.json` → `agreement.argmax_agreement` | `python scripts/03_export_int8.py` |
| INT8 vs float agreement, full cache | 99.88 % | all 16,127 windows | `scripts/04_verify_golden.py` | `artifacts/int8_export/verification_full.json` | `python scripts/04_verify_golden.py` |

### 2.1 Reconciling 0.9865 and 0.9184

**Both are random window splits. Both are leaky by the same mechanism.** Windows from one
recording — and all 560 windows of one bearing — appear on both sides of the split.

The ~6.8-point gap between them is **not** a protocol difference. It is the feature set:

| | 0.9865 | 0.9184 |
|---|---|---|
| Feature set | **v2 — 5 channels** | **v1 — 2 channels** |
| Split | stratified random over windows | unstratified random permutation |
| Evidence | `artifacts/runs/shuffled_benchmark.json` → `"feature_set": "v2"` | `artifacts/runs/task_variants.log` line 1: `cache (16127, 2, 512) \| feature set v1` |

Corroborated by the LOBO pair measured on the same two feature sets: v2 0.8018 vs v1 0.7823.

Honest presentation:

> Under a random window split — the protocol behind most published Paderborn results — the
> 5-channel model scores 0.9865 ± 0.0024 and the 2-channel model 0.9184. Both are **leaky
> references**. Under leave-one-bearing-out the same 5-channel model scores 0.8018 ± 0.0103
> window / 0.8332 ± 0.0056 per recording. The LOBO figure is the honest one.

### 2.2 Known reporting gaps

| Gap | Status |
|---|---|
| `artifacts/runs/shuffled_benchmark.json` records `feature_set` but **not** `n_seeds` or `tta_shifts`, and there is no `shuffled_benchmark.log`. The 0.9865 headline is therefore not fully reproducible from recorded artefacts. | **OPEN** — re-run with the config dumped into the JSON |
| `artifacts/runs/task_variants.json` records no `train_config` or `feature_set`; the feature set is known only from the `.log`. | **OPEN** — same fix |
| The INT8 calibration and verification windows are drawn from the same cache the deployment model trained on (`scripts/03_export_int8.py:35-38`). Disjointness between calibration and verification is real; neither set is unseen data. | **DECLARED** — the figure quoted is agreement, not accuracy |

---

## 3. Multi-stage branches

| Branch | Dataset | Groups | Honest metric | Fault authority (§1.1) | Status |
|---|---|---|---|---|---|
| S5 `bearing` | D1 Paderborn | 29 bearings | 0.7944 macro-F1 (LOBO) | **may raise Fault** | frozen |
| S4 `winding` | D2 KAIST | 3 motors | NOT RUN | NOT RUN | NOT RUN |
| S1 `supply` | D4 Thomas | 2 motors | NOT RUN | **indicative** — fails the ≥3-group test | NOT RUN |
| S2/S3 `inverter_telemetry` | D3 Bacha | 1 run/condition | NOT RUN | **indicative** — fails the ≥3-group test | NOT RUN |
| B-SIM `inverter_waveform` | simulator | — | NOT RUN | out of MVP (§13.8) | NOT RUN |

### 3.1 Required companion measurements

| Measurement | Why | Status |
|---|---|---|
| **B-S4 batch control** — train a classifier to predict the D2 acquisition batch from the same features | The polarity-corrected negative-sequence ratio separates acquisition batches perfectly at 1000 W (batch A max 0.0775 < batch B min 0.0875) and is non-monotonic in severity on 5 of 6 motor × fault-type combinations. If batch is predicted better than fault class, the fault metric is not measuring what it claims. | NOT RUN — **required beside every B-S4 metric** (§13.8) |
| **B-S1 threshold rule** — documented zero-current phase detection + its detection latency | One recording per motor × class makes a learned metric indefensible. The rule is the deliverable; the learned model is the leaky reference. | NOT RUN |
| **B-S2/S3 electrical-only ablation** — OC/SC classification with the temperature channels removed | `over_temp` is separable by a single NTC threshold, so a 4-class headline would be a thermometer in a classifier costume. The electrical-only number is the one that matters. | NOT RUN |
| **D4 label-alignment validation** — reconstructed 1000/500 sliding-window labels vs measured phase-current collapse boundaries | §13.3. If validation fails, the current-collapse rule stands alone and that must be stated. | NOT RUN |

---

## 4. Claims that must never appear

| Number | Why not |
|---|---|
| 99.2 % / 0.9927 / 0.9920 as an **accuracy** | `verification_full.json` figures are training-set accuracies — the deployment model trained on all 16,127 windows. Quote `argmax_agreement` instead. |
| 0.9865 or 0.9184 without the **(leaky reference)** label | Both are random window splits over near-identical windows of the same recordings. |
| "Verdict within a single AC cycle" | Replaced by "per-trip verdict with evidence accumulation" (`docs/workflow_v2.md` §12). |
| Any D2 claim of f/f_e **invariance** | f_e ≡ 200.00 Hz across all 48 D2 recordings; the axis is a fixed rescale on this dataset (§13.8). |
| Any D2 **absolute current amplitude** in amps | `NI_SensorSensitivity = 1.0` with the channel sensor declared `Voltage`: the `A` unit is a label on an identity scale. Ratios only. |
| Any D3 claim resting on `VDC`, `IDC` or `VD` | Per-class means span < 0.8 ADC counts against std 1.2–1.4. Dropped in §13.5. |
| A D3 **9-class** accuracy | Three classes have ~8–10 test windows under the block split (§13.8). |
| A D4 **bearing-fault** capability claim | n = 1 motor per class; bearing fault is perfectly confounded with motor identity. Documented negative result only. |
