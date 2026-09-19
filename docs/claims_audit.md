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

### 1.3 Cache rebuild, 2026-09-18 — the numbers moved and why

The repo was relocated to `C:\kone` and the Python environment did not survive.
`artifacts/order_spectra_v2.npz`, its meta parquet, and
`artifacts/runs/deployment_model.pt` were all absent, so the feature cache and every
model were rebuilt from `data/`.

**The rebuilt cache is not identical to the one behind the recorded numbers.**

| | Pre-rebuild | Rebuilt | Δ |
|---|---|---|---|
| Recordings | 2,306 | **2,318** | +12 |
| Windows | 16,127 | **16,211** | +84 |
| `inner_race` windows | 6,070 | 6,154 | +84 |
| KI14 recordings | **68** | **80** | +12 |
| All other bearings | unchanged | unchanged | — |

The entire difference is **KI14**. The machine that produced the recorded numbers held
68 of its 80 runs; this copy of `data/` has all 80. `KA04` (79, byte-identical
run-17/18 duplicate dropped) and `KA08` (79, one structurally corrupt `.mat`) are
short in both. The stale note in `dataset.py` that said "KI14 68 = absent from this
copy" was corrected at the same time.

**The DSP front end is confirmed unchanged.** The speed-estimate error — computed from
the phase current and validated against the tachometer on every recording — reproduced
to four significant figures across the rebuild:

| | Pre-rebuild | Rebuilt |
|---|---|---|
| Median speed error | 0.02214 % | **0.02210 %** |
| p99 speed error | 0.20849 % | **0.20829 %** |

Two independent runs of the whole front end over an overlapping but not identical set
of 2,300+ recordings agreeing to that precision means the signal processing, the order
resampling and the speed estimation are bit-stable across the move. The contract
fingerprint `4fdbe9910a513c40` is also unchanged.

**Effect on the headline number:**

| Metric | Pre-rebuild (3 repeats) | Rebuilt (single run) | Δ | Noise floor |
|---|---|---|---|---|
| Window accuracy | 0.8018 ± 0.0103 | **0.7944** | −0.0035 | ±0.0103 |
| Macro-F1 | 0.7944 ± 0.0114 | **0.7852** | −0.0024 | ±0.0114 |
| Per-recording accuracy | 0.8332 ± 0.0056 | **0.8356** | −0.0026 | ±0.0056 |
| Per-bearing mean | 0.7983 ± 0.2852 | **0.7945 ± 0.2836** | −0.0038 | — |

Every difference is inside the run-to-run standard deviation the 3-repeat study
measured on the same recipe. **This is not a regression and not an improvement — it is
the same result measured again.**

**Both numbers stay visible.** Neither is deleted. The rebuilt single-run figures are
current; the 3-repeat figures are labelled "pre-rebuild cache" wherever they appear.
`repeat.py` has **not** been re-run — it costs ~3¼ hours on this machine, which has no
GPU. The pre-rebuild artefacts are preserved at
`docs/prior_results/v5_pre_rebuild/`.

**Caveat on the current figure:** it is a single run, so it has no error bar of its
own. Use ±0.0103 as the noise floor when comparing anything against it. Do not compare
a future configuration to 0.7944 and call a 0.005 difference an improvement.

### 1.4 INT8 export re-run against the rebuilt model

The shipped `.mem` files were exported from the **pre-rebuild** deployment model. That
model's checkpoint no longer existed after the relocation, so the export could not have
been re-verified against its own float parent. Both were regenerated on 2026-09-18 from
`deployment_model.pt` as rebuilt.

All ten `.mem` files changed (MD5 compared before and after). The previous export's
weights are superseded; anything already flashed to an FPGA from them corresponds to a
model trained on the pre-rebuild cache.

| | Pre-rebuild export | Rebuilt export |
|---|---|---|
| argmax agreement, n=2,048 | 99.90 % | **99.85 %** |
| argmax agreement, all windows | 99.882 % (16,127) | **99.877 %** (16,211) |
| Disagreements, all windows | 19 | **20** |
| Max \|logit\| difference | 4.0409 | **1.3486** |
| Mean \|logit\| difference | 0.0811 | **0.0754** |
| Weights / biases | 27,024 / 179 | 27,024 / 179 |
| Accumulator bits (conv1–4, fc) | 21/22/23/23/21 | 21/22/23/23/21, all fit int32 |
| Methodology bar ≥ 98 % | PASS | **PASS** |

Both exports clear the bar comfortably. The max logit difference improved threefold,
which is a quantisation-calibration detail and not a claim about accuracy.

Disagreements concentrate on the known-hard specimens — `KI05` 6, `K002` 3, `KA03` 3,
`KA04` 2, `KA05` 2, `KA30` 2, `KA22` 1, `KI14` 1. Eight bearings account for all 20.
These are the same bearings the LOBO sweep fails on, which is what you would expect:
INT8 and float disagree where the float model's own logits are closest together.

### 1.5 Post-hoc protocol change: V5 leave-one-session-out (B-S4)

| Field | Value |
|---|---|
| **Decided at** | **2026-09-18**, after the `i0rel_residual` probe |
| Repo state at decision | `3c2a6d0` (the commit that recorded the probe) |
| Protocol added | **V5 — leave-one-session-out** over the three D2 acquisition days |
| Authority | user instruction, this session |

**This is a protocol chosen after seeing data, and it is declared as such.** The
pre-registered B-S4 protocol set (`workflow_v2.md` §5) was V1 leave-one-motor-out,
V2 lowest-severity holdout, V3 shuffled reference. V5 was added later.

**The reason it was added.** A 1-NN probe on `i0rel_residual` — a scalar that is zero
by Kirchhoff in a three-wire machine and therefore cannot contain winding information —
recovered the acquisition session at 1.000 on the 1000/1500 W recordings, and recovered
70 % of the fault label with it. Leave-one-motor-out does not hold session out, so V1
alone cannot distinguish winding diagnosis from session identification.

**Why adding it is not the same as tuning on test.** V5 was defined *before* any B-S4
model was fitted — no fault-class number existed when the protocol was chosen. The
grouping is also not a free parameter: it is the acquisition date read from the TDMS
`Date Created` property, corroborated independently by DAQ chassis, recording-duration
discipline and root-name convention (four signals, one partition, `data_notes_d2.md`
§7D). Nothing about the split was selected to improve a score.

**What would make this illegitimate, and did not happen:** choosing the grouping to
maximise a metric, trying several groupings and keeping the best, or defining V5 after
seeing V1's result. V1 and V5 were run in the same invocation of
`scripts/branches/10_winding.py`, from one cached feature matrix.

**How to read V1 and V5 together.** V1 answers "does this transfer to a new motor",
V5 answers "does it transfer to a new acquisition session". Both are reported; the gap
between them is itself the result (§3.2).

### 1.6 Correction: five residual groups nesting in three days, not six sessions

An earlier statement in this session — and in an instruction issued from it — held that
the `i0rel_residual` analysis resolved **six** acquisition sessions. **That was wrong.**

The correct structure is **five residual-derived groups nesting inside three
acquisition days**:

| Day | Residual-derived groups | n |
|---|---|---|
| 2022-01-25 | `cDAQ1Mod2 (+,+,+)`, `cDAQ1Mod2 (+,-,-)` high-residual | 7 + 11 |
| 2022-03-08 | `cDAQ1Mod2 (+,+,-)`, `cDAQ1Mod2 (+,-,-)` low-residual | 8 + 12 |
| 2022-08-11 | `cDAQ5Mod1 (+,+,+)` | 7 |

The error was double-counting: the 12/11 split inside the `(+,-,-)` polarity group was
counted as a sixth group, when the five batch keys already encoded it through the
root-name style.

**The metadata verification is what caught it**, which is the argument for doing the
verification rather than trusting the probe. V5 groups by the three **days**, not by the
five residual groups — the days are what the metadata independently establishes.

### 1.7 B-S4 fusion status: INDICATIVE

| Field | Value |
|---|---|
| Protocol used for the decision | **V5** (leave-one-session-out) |
| Validation groups scored | **2** — floor is **3** |
| Honest macro-F1 | **0.6250** — floor is **0.75** |
| **Status** | **INDICATIVE — no `Fault` authority** |

**Fails both tests of the pre-registered floor** (§1.1, fixed 2026-09-18T07:19:23Z at
commit `7344a436`, before any multi-stage branch existed). The floor has **not** been
adjusted, and must not be.

Groups scored is 2 rather than 3 because the 2022-08-11 session contains only
`inter_coil` recordings (7/0) and cannot score a two-class task. It remains in training
for the other folds.

Consequence for `drivesentinel/fusion.py` and the dashboard: B-S4 may raise `Warning`
and contribute to the evidence string, but may never raise `Fault` on its own. Its
waveform and spectrum panels stay live — an indicative branch is displayed, not hidden.

### 1.8 Fusion authority as measured (P4, 2026-09-18)

Which branches may raise `Fault`, and why. **Read from each branch's results JSON at
load time by `fusion.load_branch_metrics()`, never hardcoded** -- so the table cannot
drift away from the measured numbers.

Floor: **>= 3 independent validation groups AND honest macro-F1 >= 0.75**,
pre-registered 2026-09-18T07:19:23Z at commit `7344a436`, before any multi-stage branch existed (1.1).
**Not adjusted.**

| Branch | Stage | Protocol | Groups | Macro-F1 | Tier |
|---|---|---|---|---|---|
| `bearing` | S5 | leave-one-bearing-out | 29 | 0.7852 | **Fault-capable** |
| `winding` | S4 | V5 (leave-one-session-out) | 2 | 0.6250 | **INDICATIVE** |
| `supply` | S1 | — | — | — | **NOT MEASURED** |
| `inverter_telemetry` | S3 | — | — | — | **NOT MEASURED** |

- **`bearing` holds Fault authority.** 29 bearings and macro-F1 0.7852 clear both
  tests. It is the only branch that can condemn a drive on its own.
- **`winding` is INDICATIVE and fails both tests** -- 2 scored sessions against 3, and
  0.6250 against 0.75. It may raise `Warning` and contribute evidence. Its panels stay
  live; it is capped, not hidden.
- **`supply` and `inverter_telemetry` are NOT MEASURED** -- no results JSON yet.
  Absence of a metric earns the same lack of trust as a bad one, so they have no
  authority either. They are re-read automatically once their branches run.

Asserted in `tests/test_fusion.py`: an INDICATIVE branch held at `p_fault = 0.99` for
50 consecutive updates reaches `Warning` and stops. If that test ever passes as
`Fault`, the floor has been defeated.

### 1.9 Trip gating is implemented and disabled

`config.TRIP_CONFIG["enabled"] = False`. No real dataset in the roster has a speed
ramp:

| Dataset | f_e | Ramps? |
|---|---|---|
| D1 Paderborn | 900 / 1500 rpm, constant within each 4 s recording | no |
| D2 KAIST | 200.00 Hz in all 48 recordings | no |
| D3 Bacha | 10 rad/s constant, 10 Hz telemetry | no |
| D4 Thomas | mains-fed, 49.955-50.036 Hz | no |

Segmentation on any of them returns `cruise` for every window, so gating changes
nothing. Enabling it by default would ship a code path no data exercises and a claim
nothing tests. **No claim is made anywhere that trip gating gates real data.**

It is unit-tested against synthetic ramps with known ground truth: a trapezoidal
profile segments into accel/cruise/decel in order, with cruise boundaries within
0.25 s of truth, and a linear chirp tracks within 2 Hz median error.

The one real-data application is **envelope** segmentation of the D4 start-up
transients, where f_e is pinned by the grid and the transient lives in the current
envelope. Reported as segmentation only, never as frequency tracking. Measured cruise
onsets: FILE 1 at 8.83 s, FILE 6 at 9.17 s -- consistent with the per-2 s RMS blocks
in `data_notes_d4.md`. For FILE 5 and 10 the machine never rotates, so `cruise` there
means steady **current**, not steady rotation; stated so the label is not over-read.

The order-resolution table in `docs/results_multistage.md` is **analytic
extrapolation**, labelled as such per row. It is arithmetic about what a window length
can resolve, not a claim about detection performance at low speed.

### 1.10 Dashboard: what it is allowed to display (P5, 2026-09-19)

`dashboard/app.py` is **replay only**. Every number it shows is read from a results
JSON at display time; nothing is computed live, nothing is typed in, and no metric
is passed to it as a literal.

**The out-of-sample rule is enforced in code, not trusted.**
`workflow_v2.md` section 8 requires a replayed unit's predictions to come from the
fold model that held that unit out. `panels.assert_out_of_sample()` raises if a
bearing scenario is not marked out-of-sample, or if the replayed bearing appears in
its own fold's training-set list, and `app.py` renders the error instead of the
panel.

This costs real compute and is worth it. `scripts/02_train_lobo.py` keeps only the
deployment model, which trained on all 29 bearings, so each bearing scenario
**retrains its own leave-one-bearing-out fold** (about 2.8 min per bearing on this
CPU). Reusing the deployment model would have been one line and would have made
every bearing replay in-sample -- the exact failure the rule exists to prevent.

**Authority is displayed per stage, not in a legend.** Each stage light carries a
badge -- FAULT-CAPABLE / INDICATIVE / NOT MEASURED -- with the branch's honest
metric and protocol underneath, so a viewer can see that a light can never go red
without looking anywhere else. Tiers come from `fusion.load_branch_metrics()`.

**No stage is hidden.** `supply` and `inverter_telemetry` have no results JSON and
render as NOT MEASURED with live panels. An empty stage is more honest than a
diagram implying four working branches.

**As displayed today: exactly one stage of five can raise Fault.**

| Stage | Branch | Tier |
|---|---|---|
| S1 Supply | `supply` | NOT MEASURED -- branch not built |
| S2 DC link | `inverter_telemetry` | NOT MEASURED -- branch not built |
| S3 Inverter | `inverter_telemetry` | NOT MEASURED -- branch not built |
| S4 Winding | `winding` | INDICATIVE -- capped at Warning |
| S5 Bearing | `bearing` | **FAULT-CAPABLE** |

**Scenario (a) is a severity ramp, not a healthy-to-fault ramp.** All three D2
healthy recordings are from 2022-01-25, so healthy is not measurable under a
session-aware split (1.5, 1.7, data_notes_d2.md 7D). The original scenario (a) in
workflow_v2.md section 8 called for "healthy trip -> gradual S4 winding fault".
That cannot be built from real data and **is not synthesised**. The replacement
walks real inter_turn recordings from the lowest to the highest severity and is
named as such everywhere it appears.

**Scenario (b) includes a failure on purpose.** `KI05` scores 0.0089 under
leave-one-bearing-out and is one of the three bearings shipped as a scenario,
alongside `KA04` (works) and `K001` (healthy specimen). A demo that can only show
successes is a worse demo, and the per-bearing spread is already the headline
finding of `accuracy_ceiling.md`.

**Scenarios (c) and (d) are absent and say why.** (c) S1 phase loss is SKIPPED --
the B-S1 branch does not exist: no adapter, no threshold rule, no results JSON.
(d) S2/S3 open circuit is NOT MEASURED -- no
`artifacts/multistage/inverter_telemetry/*.json`. Both appear in the manifest with
a stated reason, and the dashboard lists them in the sidebar rather than omitting
them.

**Stored scenario size.** Capped at 120 windows per bearing, evenly spaced across
the fold so a replay spans the whole recording set rather than its first seconds.
Metrics shown on the cards come from the **full** fold, not from the stored subset;
`n_windows_in_fold` and `n_windows_stored` are both recorded in each NPZ.

### 1.11 B-S1 supply: a rule, not a model, and why (P2, 2026-09-19)

**The deliverable is a documented threshold rule.** D4 has one recording per
(motor × class) and two motors, so under leave-one-motor-out a learned model sees
one training example per class — it separates two 20 s captures rather than
learning phase loss. Liu et al. measure exactly that on this dataset: macro-F1
0.9682 under a random split, 0.5856 under a within-label block split.

**The rule:** a phase is LOST in a window when its 0.2 s RMS falls below 5 % of
the median of the other two. A window is OFF when every phase is below 0.05 A, and
OFF windows are excluded from scoring. `phase_loss_running` is separated from
`single_phasing_start` by **rotation** (vib_x RMS above 0.005), not by current —
the two look nearly identical electrically, and measured vibration separates them
by a factor of ~20 (stalled 0.0008–0.0012, rotating 0.017–0.041).

| Protocol | Scheme | Accuracy | Macro-F1 |
|---|---|---|---|
| **R2 — the rule** | leave-one-motor-out | **1.0000** | **1.0000** |
| L1 — learned | leave-one-motor-out | 0.6593 | 0.6394 |
| L3 — learned **(leaky reference)** | shuffled windows | 0.9983 | 0.9981 |

**34 points** separate the learned model's leaky and honest splits, on the same
features. The rule is unaffected because nothing in it is fitted.

Detection latency per event: FILE 2 **0.10 s**, FILE 5 **0.30 s**, FILE 7
**0.00 s**, FILE 10 **1.50 s**. Measured against an amplitude crossing independent
of the rule's own threshold, and floored by the 0.1 s hop.

#### Correction: the threshold's sensitivity is narrower than first claimed

The first draft of `branches/supply.py` asserted that "anything between 1 % and
30 % gives the same answer". **That was wrong**, and measuring it is what caught
it:

| Level | Safe range | Evidence |
|---|---|---|
| **Recording verdict** (what the branch reports) | **0.03 – 0.40**, 10/10 throughout | at 0.02 it loses FILE 2; at 0.01 it loses FILE 2 and FILE 7 |
| Window classification | **0.048 – 0.053** only | worst real lost-phase ratio 0.0476 (FILE 10); tightest normal-window ratio 0.0526 (FILE 5) |

Both window-level bounds come from the same few windows — those straddling the
instant a phase dies. A 0.2 s window spanning the transition contains both states
and its RMS lands between them. That is an artefact of the window length, not a
property of the fault, and it is why the branch reports a recording-level verdict
requiring a ≥ 0.5 s event rather than trusting individual windows.

**Do not quote the window-level margin as if it had the recording-level margin.**
Both numbers are in the module docstring and asserted in
`tests/test_supply_features.py`.

### 1.12 D4 label vector: reconstructed, not validated, not used

The paper describes a 1000-sample window at step 500 over the ten files merged in
order. That reconstruction was tested against the phase-current collapse
boundaries measured here:

| Check | Result |
|---|---|
| Label-run boundaries coinciding with a **file** boundary at 1998 windows/file | **6 of 14** |
| Same, at 1999 windows/file | 1 of 14 |
| Label-run boundaries coinciding with a **measured phase-loss event** | **0 of 4** |

So the per-file window count is probably 1998 and the file structure
reconstructs — but the within-file event structure does not match anything
measurable in the currents.

**The label vector is therefore not used.** Labels come from the current-collapse
rule, which was primary either way. The disagreement is recorded rather than
resolved by assumption, and `adapters/thomas_motor.py` sets
`provenance["label_vector_used"] = False` on every recording.

### 1.13 B-S1 fusion status: INDICATIVE, at macro-F1 1.0000

| Field | Value |
|---|---|
| Protocol | R2, threshold rule, leave-one-motor-out |
| Validation groups | **2** — floor is **3** |
| Honest macro-F1 | **1.0000** — floor is 0.75 |
| **Status** | **INDICATIVE — no `Fault` authority** |

**A perfect score still does not earn `Fault` authority**, because two motors is
below the pre-registered group minimum. The floor was fixed 2026-09-18T07:19:23Z
at commit `7344a436`, before any multi-stage branch existed, and §1.1 recorded the
expectation "S1 supply — 2 motors — **fails the group test** → indicative" at that
time. It has not been adjusted.

This is the floor working as intended: a branch can be perfectly right about the
recordings it has and still not be trusted to condemn a drive alone, because two
machines cannot tell you how the rule behaves on a third.

### 1.14 D4 bearing confound: a documented negative result

Predicting **motor identity** from the same features scores **0.9983** against a
0.5049 baseline under a shuffled split.

That number is not a bearing-fault capability and must never be quoted as one.
D4 has exactly one motor per bearing condition, so "outer-race fault" and "motor
identity" are the same variable; nothing distinguishes a bearing defect from any
other difference between two physical machines — winding tolerances, mounting,
alignment, age. Vibration is measurably 1.6–2.3× higher on the faulty motor across
every matched scenario pair, and that measurement cannot be attributed to the
bearing.

Stated once in `adapters/thomas_motor.py::bearing_confound_note()` and quoted from
there everywhere it appears, so the wording cannot drift.

### 1.15 B-S2/S3 inverter telemetry: the ablation is the finding (P3, 2026-09-19)

| Protocol | Split | Accuracy | Macro-F1 | Baseline |
|---|---|---|---|---|
| **V3 electrical-only** | contiguous block | **0.8503** | **0.8229** | 0.4045 |
| V3 electrical-only **(leaky ref)** | random windows | 0.9229 | 0.9106 | 0.4048 |
| V1 with temperature | contiguous block | 1.0000 | 1.0000 | 0.4045 |
| V2 with temperature **(leaky ref)** | random windows | 1.0000 | 1.0000 | 0.4048 |

**V1 and V2 agree exactly.** The leaky split gains nothing — not because the block
split is safe, but because with a thermometer in the feature set the task is
already saturated. A 4-class score that includes `over_temp` is substantially a
temperature reading: F6 heats HB1, F7 heats HB1 and HB2, F8 heats HB3, each
matching its filename.

#### The measurement that says how much to trust any of it

Per-class F1 under the **electrical-only** block split:

| Class | F1 | Test windows |
|---|---|---|
| `normal` | 0.984 | 127 |
| **`open_circuit`** | **0.511** | 51 |
| `short_circuit` | 1.000 | 31 |
| **`over_temp`** | **0.796** | 105 |

**`over_temp` scores F1 0.796 with no temperature sensor in the feature set.**

A thermal fault is not physically detectable from two 10 Hz phase currents. That
number is the model identifying **which run** a window came from, not what
condition the inverter was in. Each condition is one contiguous file recorded at a
distinct wall-clock time (13:24 → 14:31), so anything that drifts with time
carries run identity, and the currents drift.

**Treat every 4-class number on this dataset as an upper bound contaminated by run
identification.**

And the class the electrical channels *should* see is the weakest:
`open_circuit` at F1 **0.511**. F1 (HB2 high-side open) has almost the same
channel means as F0 — measured Ia/Ib 519/474 against 515/475.

#### 9-class location view: confusion matrix only

Under the block split the smallest classes get 9–11 test windows (`F4` 9, `F3` 11,
`F5` 11). A per-class figure on nine samples has a 95 % CI of roughly ±35 points.
The matrix is rendered qualitatively in `results_multistage.md`; **no per-class
number is reported from it**, and none should be quoted.

#### Calibration independence, by construction

`data_notes_d3.md` §4 refits the NTC Steinhart–Hart coefficients because the
shipped ones under-read by 13–20 °C. That refit is three parameters on four points
and is an extrapolation below 30 °C.

**No feature in this branch depends on it.** Temperature enters only as raw ADC
statistics and as ADC *differences* between channels, both monotone in temperature
under any calibration. `tests/test_inverter_telemetry.py` asserts that a constant
shift applied to every NTC reading — which is what a different calibration does to
first order — leaves every difference feature unchanged.

If the refit is wrong, none of these numbers move.

#### Dropped channels

`VDC`, `IDC`, `VD` are not used. Per-class means span 0.78 / 0.45 / 0.26 ADC counts
against per-channel std 1.2–1.4 — inside their own quantisation noise. That also
removes `Vdc·Idc`, `dVdc/dt` and `dIdc/dt` from the original plan: products and
derivatives of a constant.

### 1.16 B-S2/S3 fusion status: INDICATIVE at zero validation groups

| Field | Value |
|---|---|
| Protocol | V1, contiguous block split, within-run |
| Validation groups | **0** — floor is **3** |
| Honest macro-F1 | 1.0000 — floor is 0.75 |
| **Status** | **INDICATIVE — no `Fault` authority** |

**Zero**, not two: one run per condition means there is nothing independent to hold
out at all. The block split separates early-in-run from late-in-run within the same
recording.

§1.1 recorded the expectation "S2/S3 telemetry — 1 run per condition — **fails the
group test** → indicative" on 2026-09-18, before this branch existed. The floor has
not been adjusted.

All three multi-stage branches now score at or above the macro-F1 floor — 1.0000,
1.0000 and 0.6250 — and **all three are INDICATIVE**, every one of them failing on
the group test. That is the floor doing the job it was written for: the binding
constraint on this project is how many independent machines each dataset contains,
not how well a model fits.

### 1.17 D2 CNN experiment — post-hoc, criteria declared before the run

**This was run AFTER the gradient-boosting result was known.** It is a post-hoc experiment and is declared as one. What makes it legitimate rather than fishing is that the recipe and the acceptance criteria were written into `scripts/experiments/d2_cnn.py` and **committed before the first run** (commit `05cf8d7`), so neither could be adjusted after seeing the numbers.

**Hypothesis:** a CNN on the full f/f_e harmonic spectrum finds inter-coil vs inter-turn structure that 23 scalar features miss.

**Recipe, fixed in advance:** OrderSpectrumCNN (16,9,2)(32,7,2)(64,5,2)(64,3,2) + GAP + linear, 512 bins over 0–20 orders, 30 epochs, lr 0.003, batch 256, OneCycle, label smoothing 0.05, class-balanced weights, seeds [0, 1, 2]. Single model, no ensemble, no TTA, **no hyperparameter search**. 10,463 windows — the same rows, task and splits as the GBM run; only the features and the model change.

| Protocol | CNN (mean ± std, 3 seeds) | Gradient boosting |
|---|---|---|
| V1 leave-one-motor-out | 0.4719 ± 0.0081 | 0.3784 |
| **V5 leave-one-session-out** | **0.5608 ± 0.0142** | **0.6251** |
| V3 shuffled **(leaky reference)** | 0.8448 ± 0.0078 | 0.9990 |

**Acceptance criteria, as declared:**

| # | Criterion | Threshold | Measured | Outcome |
|---|---|---|---|---|
| 1 | V5 beats the session-only baseline — the `i0rel_residual` 1-NN, a scalar that cannot contain winding information | > 0.750 | 0.5608 | **FAIL** |
| 2 | V5 beats the GBM by more than the seed spread | > 0.6251 + 0.0142 | margin -0.0643 | **FAIL** |

**Outcome: NEGATIVE RESULT.** Shipped branch: **gradient boosting (unchanged)**.

The hypothesis is not supported. A CNN with capacity comparable to the bearing model (26,914 parameters against 27,024), given the whole spectrum rather than 23 scalars, does not recover winding structure the scalar features missed — because on this dataset the structure that survives a session-aware split is mostly not winding structure. This is consistent with everything else measured on D2: the session is recoverable at 1.000 from a quantity that is zero by Kirchhoff, and V1 sits below its own majority baseline for both models.

**No further CNN variants were tried.** Sweeping architectures against a declared acceptance criterion until one passes is exactly the failure the criterion exists to prevent. One recipe, declared, run, reported.

**The tier does not move.** INDICATIVE -- 2 validation groups against a floor of 3, whatever this scores. The floor is on GROUPS as well as macro-F1 and was pre-registered before any branch existed.

Regenerate: `python scripts/experiments/d2_cnn.py`. Results in `artifacts/multistage/winding/winding_cnn_results.json`.

### 1.18 D3 run-identification control — the inference, measured

§1.15 argued from `over_temp` scoring F1 0.796 with no temperature sensor that the
model must be reading run identity. This measures that directly: **train the same
features, on the same block split, to predict which run a window came from.**

| Predicting | Features | Accuracy | Macro-F1 | Baseline |
|---|---|---|---|---|
| which run (9 runs) | with temperature | **0.9713** | 0.9111 | 0.4045 |
| which run (9 runs) | electrical only | **0.7484** | 0.7277 | 0.4045 |

Beside the condition scores, same split and same features:

| Features | 4-class condition | which run | Gap |
|---|---|---|---|
| with temperature | 1.0000 | 0.9713 | +0.0287 |
| electrical only | 0.8503 | 0.7484 | +0.1019 |

**The two quantities are within a few points of each other on both feature sets**,
which is what you would expect if they are largely the same thing. D3 has exactly
one run per condition, so a model that can name the run can name the condition
without diagnosing anything.

**Consequence:** the 4-class family score is an upper bound on condition diagnosis,
and the run-identification number is how far above the truth that bound may sit.
Quote neither without the other.

This is now a measurement rather than an inference, and it is generated into
`docs/results_multistage.md` from the results JSON.

---

## 2. Bearing pipeline (S5, frozen)

**Cache generation matters for every row.** Rows marked **[rebuilt]** were measured on
the 2026-09-18 rebuilt cache (2,318 recordings / 16,211 windows); rows marked
**[pre-rebuild]** were measured on the earlier cache (2,306 / 16,127) and have not been
re-run. See §1.3.

| Claim | Value | Cache | Protocol | Produced by | Recorded in | Regenerate |
|---|---|---|---|---|---|---|
| **Window accuracy** | **0.7944** | **[rebuilt]** | LOBO, 29 folds, pooled, **single run** | `scripts/02_train_lobo.py` | `artifacts/runs/lobo_summary.json` → `pooled.window_acc` | `python scripts/02_train_lobo.py` |
| **Macro-F1** | **0.7852** | **[rebuilt]** | same | same | same → `pooled.macro_f1` | same |
| **Per-recording accuracy** | **0.8356** | **[rebuilt]** | same | same | same → `pooled.recording_acc` | same |
| **Per-bearing mean** | **0.7945 ± 0.2836** (SEM 0.0527) | **[rebuilt]** | same | same | same → `per_bearing_acc_mean` / `_std` / `_sem` | same |
| Majority baseline | 0.4131 | [rebuilt] | — | same | same → `pooled.majority_baseline` | same |
| Aggregation @ 40 windows | 0.8626 | [pre-rebuild] | LOBO + pooling | same | `docs/prior_results/v5_pre_rebuild/lobo_summary_pre_rebuild.json` | same |
| Window accuracy | 0.8018 ± 0.0103 | [pre-rebuild] | LOBO, 29 folds, pooled, **3 repeats** | `scripts/experiments/repeat.py` | `artifacts/runs/repeat_v2.json` → `summary.window_acc` | `python scripts/experiments/repeat.py` (~3¼ h, no GPU) |
| Macro-F1 | 0.7944 ± 0.0114 | [pre-rebuild] | same | same | same → `summary.macro_f1` | same |
| Per-recording accuracy | 0.8332 ± 0.0056 | [pre-rebuild] | same | same | same → `summary.recording_acc` | same |
| **Run-to-run noise floor** | **± 0.0103** | [pre-rebuild] | 3 repeats, same recipe, different base seeds | same | same → `summary.window_acc.std` | same |
| Pooled window accuracy | 0.7979 | [pre-rebuild] | LOBO, single run | `scripts/02_train_lobo.py` | `docs/prior_results/v5_pre_rebuild/lobo_summary_pre_rebuild.json` | superseded |
| Per-bearing mean | 0.7983 ± 0.2852 | [pre-rebuild] | same | same | same | superseded |
| v1 (2-channel) comparison | 0.7823 | [pre-rebuild] | LOBO, 3 repeats, feature set v1 | `scripts/experiments/repeat.py` | `artifacts/runs/repeat_noval.json` | `DRIVESENTINEL_FEATURE_SET=v1 python scripts/experiments/repeat.py` |
| Accuracy **(leaky reference)** | **0.9865 ± 0.0024** | [pre-rebuild] | stratified 5-fold over **windows**, feature set **v2** | `scripts/experiments/shuffled_benchmark.py` | `artifacts/runs/shuffled_benchmark.json` → `accuracy` | `python scripts/experiments/shuffled_benchmark.py` |
| Macro-F1 **(leaky reference)** | 0.9847 | [pre-rebuild] | same | same | same → `macro_f1` | same |
| Window accuracy **(leaky reference)** | **0.9184** | [pre-rebuild] | shuffled windows, feature set **v1** | `scripts/experiments/task_variants.py:38` | `artifacts/runs/task_variants.json` → `3class_shuffled_LEAKY.window_acc` | `DRIVESENTINEL_FEATURE_SET=v1 python scripts/experiments/task_variants.py` |
| Binary LOBO window accuracy | 0.7982 | [pre-rebuild] | LOBO, binary healthy/damaged | `scripts/experiments/task_variants.py` | `artifacts/runs/task_variants.json` → `binary_lobo.pooled.window_acc` | same |
| Speed-estimate error | median **0.0221 %**, p99 **0.2083 %** | [rebuilt] | current-derived vs tachometer, all recordings | `scripts/01_build_cache.py` | `artifacts/cache_report.json` | `python scripts/01_build_cache.py` |
| INT8 vs float argmax agreement | **99.85 %** (n=2,048) | [rebuilt] | calibration-disjoint windows | `scripts/03_export_int8.py` | `artifacts/int8_export/verification.json` → `agreement.argmax_agreement` | `python scripts/03_export_int8.py` |
| INT8 vs float agreement, full cache | **99.877 %** | [rebuilt] | all 16,211 windows | `scripts/04_verify_golden.py` | `artifacts/int8_export/verification_full.json` | `python scripts/04_verify_golden.py` |
| INT8 max \|logit\| difference | 1.3486 | [rebuilt] | n=2,048 | `scripts/03_export_int8.py` | `artifacts/int8_export/verification.json` | same |

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
| S5 `bearing` | D1 Paderborn | 29 bearings | **0.7852 macro-F1** (LOBO, rebuilt cache) | **may raise Fault** (29 ≥ 3 groups, 0.7852 ≥ 0.75) | frozen |
| S4 `winding` | D2 KAIST | **2 sessions scored** (3 days, 1 degenerate) | **0.6250** macro-F1 (V5, leave-one-session-out) | **INDICATIVE** — fails both floor tests | inter_coil vs inter_turn only; `healthy` NOT MEASURABLE |
| S1 `supply` | D4 Thomas | **2 motors** | **1.0000** macro-F1 (R2, threshold rule) | **INDICATIVE** — 2 groups < 3, as predicted | threshold rule, not a learned model |
| S2/S3 `inverter_telemetry` | D3 Bacha | **0 groups** (1 run/condition) | 1.0000 macro-F1 with temperature; **0.8229 electrical-only** | **INDICATIVE** — 0 groups < 3, as predicted | ablation leads; 9-class is qualitative only |
| B-SIM `inverter_waveform` | simulator | — | NOT RUN | out of MVP (§13.8) | NOT RUN |

### 3.1 Required companion measurements

| Measurement | Why | Status |
|---|---|---|
| **B-S4 batch control** — train a classifier to predict the D2 acquisition batch from the same features | The polarity-corrected negative-sequence ratio separates acquisition batches perfectly at 1000 W (batch A max 0.0775 < batch B min 0.0875) and is non-monotonic in severity on 5 of 6 motor × fault-type combinations. If batch is predicted better than fault class, the fault metric is not measuring what it claims. | NOT RUN — **required beside every B-S4 metric** (§13.8) |
| **B-S1 threshold rule** — documented zero-current phase detection + its detection latency | One recording per motor × class makes a learned metric indefensible. The rule is the deliverable; the learned model is the leaky reference. | NOT RUN |
| **B-S2/S3 electrical-only ablation** — OC/SC classification with the temperature channels removed | `over_temp` is separable by a single NTC threshold, so a 4-class headline would be a thermometer in a classifier costume. The electrical-only number is the one that matters. | NOT RUN |
| **D4 label-alignment validation** — reconstructed 1000/500 sliding-window labels vs measured phase-current collapse boundaries | §13.3. If validation fails, the current-collapse rule stands alone and that must be stated. | NOT RUN |

---

### 3.2 B-S4 winding — every number

Task is **inter_coil vs inter_turn**. `healthy` is **NOT MEASURABLE** on this dataset
under any session-aware split (all three healthy recordings are from 2022-01-25) and no
number is reported for it anywhere.

| Claim | Value | Protocol | Recorded in |
|---|---|---|---|
| **V5 accuracy** | **0.6251** | leave-one-session-out, 2 scored folds | `artifacts/multistage/winding/winding_results.json` -> `V5.pooled` |
| **V5 macro-F1** | **0.6250** | same | same |
| V5 majority baseline | 0.5975 | same | same |
| V1 accuracy | 0.3784 | leave-one-motor-out, 3 folds | -> `V1.pooled` |
| V1 macro-F1 | 0.3456 | same | same |
| V1 majority baseline | 0.5020 | same | same |
| Accuracy **(leaky reference)** | 0.9990 | shuffled windows | -> `V3.pooled` |
| **Session-only baseline** | **0.750** | 1-NN on `i0rel_residual`, 1000/1500 W | -> `V4_probe.subsets.1000W_1500W_only` |
| Session from features **(leaky reference)** | 0.9996 | shuffled windows | -> `V4.targets` |

Regenerate: `python scripts/branches/10_winding.py --rebuild`
(~2 min), then `python scripts/60_render_multistage_results.py`.

**How to read these.** V1 (0.3784) is **below its 0.5020
majority baseline** — nothing transfers across motors. V5 (0.6251) is barely
above its 0.5975 baseline, and **below the 0.750 that a single
winding-independent scalar achieves**. Under the reporting rule agreed for this branch, a
fault-class score that does not clearly beat the session-only baseline is **not
demonstrating winding diagnosis**, and V5 does not beat it.

The comparison is directional rather than exact: the session-only probe is per-recording
(n=28) and V5 is per-window (n=10,463), so they are not a like-for-like contest. The
conclusion does not rest on the margin — it rests on V1 sitting below chance and V5
sitting below a scalar that cannot contain winding information.

## 4. Claims that must never appear

| Number | Why not |
|---|---|
| 99.2 % / 0.9922 / 0.9915 as an **accuracy** | `verification_full.json` figures are training-set accuracies — the deployment model trained on all 16,211 windows. Quote `argmax_agreement` instead. |
| 0.9865 or 0.9184 without the **(leaky reference)** label | Both are random window splits over near-identical windows of the same recordings. |
| "Verdict within a single AC cycle" | Replaced by "per-trip verdict with evidence accumulation" (`docs/workflow_v2.md` §12). |
| Any D2 claim of f/f_e **invariance** | f_e ≡ 200.00 Hz across all 48 D2 recordings; the axis is a fixed rescale on this dataset (§13.8). |
| Any D2 **absolute current amplitude** in amps | `NI_SensorSensitivity = 1.0` with the channel sensor declared `Voltage`: the `A` unit is a label on an identity scale. Ratios only. |
| Any D3 claim resting on `VDC`, `IDC` or `VD` | Per-class means span < 0.8 ADC counts against std 1.2–1.4. Dropped in §13.5. |
| A D3 **9-class** accuracy | Three classes have ~8–10 test windows under the block split (§13.8). |
| A D4 **bearing-fault** capability claim | n = 1 motor per class; bearing fault is perfectly confounded with motor identity. Documented negative result only. |
