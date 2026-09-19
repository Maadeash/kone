# DriveSentinel — context handoff

State document for a session that knows nothing about this project.
Generated 2026-09-19 from the repo and the results JSON, at commit `824e786`.

---

## 1. What this is

**KONE Elevate'26, PS-06: real-time drive health monitoring from various-stage voltage
and current waveforms.** The drive is modelled as five stages — S1 supply input, S2 DC
link, S3 inverter, S4 motor winding, S5 bearing — with one diagnostic branch per stage,
fused by a rule-based layer, and shown in a replay dashboard.

| | |
|---|---|
| Repo | `C:\kone` — remote `https://github.com/Maadeash/kone`, branch `main` |
| Build plan | `docs/workflow_v2.md` — §13 holds post-audit decisions and **supersedes** anything above it |
| Claims register | `docs/claims_audit.md` — every number with its protocol and source |
| Environment | `.venv/`, Python 3.12, **CPU torch 2.14.0+cpu**. No NVIDIA GPU (Intel UHD only) |
| Tests | **347 pass**, ~35 s |
| Datasets | `data/` (D1 Paderborn, 21 GB), `data_ext/` (D2 KAIST, D3 Bacha, D4 Thomas, 16 GB). All gitignored |

No dataset shares a physical machine across stages, so the demo is labelled
**COMPOSITE REPLAY** and fusion is rule-based rather than learned.

### Phase status

| Phase | Deliverable | Status |
|---|---|---|
| P0 | Env, schema, split utilities, bearing rebuild | **done** |
| P1 | D2 winding branch | **done** — the confound analysis is the deliverable |
| P2 | D4 supply branch | **done** — threshold rule |
| P3 | D3 inverter telemetry branch | **done** |
| P4 | Trip gating + fusion | **done** |
| P5 | Scenarios + Streamlit dashboard | **done** |
| P6 | Results, claims audit, README | **done**, rolling |
| P7/P8 | Simulation branch, simulated speed-ramp validation | **out of scope** (§5) |
| — | **RTL / FPGA track** | **NOT STARTED** (§6) |

---

## 2. Branches

Authority is read from the results JSON at runtime by `fusion.load_branch_metrics()` —
never hardcoded. Regenerate this table with that function, don't retype it.

| Branch | Stage | Protocol | Groups | Macro-F1 | Tier | Results JSON |
|---|---|---|---|---|---|---|
| `bearing` | S5 | leave-one-bearing-out | **29** | **0.7852** | **FAULT-CAPABLE** | `artifacts/runs/lobo_summary.json` |
| `winding` | S4 | V5 leave-one-session-out | 2 | 0.6250 | INDICATIVE | `artifacts/multistage/winding/winding_results.json` |
| `supply` | S1 | R2 threshold rule, leave-one-motor-out | 2 | 1.0000 | INDICATIVE | `artifacts/multistage/supply/supply_results.json` |
| `inverter_telemetry` | S2/S3 | V1 contiguous block, within-run | **0** | 1.0000 | INDICATIVE | `artifacts/multistage/inverter_telemetry/inverter_telemetry_results.json` |

**Exactly one stage of five can raise `Fault`.** Three branches sit at or above the
macro-F1 floor and all three are INDICATIVE — every one fails on the *group* test. The
binding constraint on this project is how many independent machines each dataset
contains, not how well a model fits.

Headline numbers, all from the JSON above:

| Branch | Honest | Leaky reference | Baseline / control |
|---|---|---|---|
| bearing | 0.7944 window, 0.8356 per-recording | 0.9865 *(stratified 5-fold over windows)* | 0.4131 majority |
| winding | V5 0.6251; **V1 0.3784, below chance** | V3 0.9990 | **session-only probe 0.750** |
| supply | R2 rule **1.0000**, 10/10 recordings | L3 learned 0.9983 | L1 learned honest 0.6593 |
| inverter | V3 electrical-only 0.8503 / F1 0.8229 | V2 1.0000 | V1 with temperature 1.0000 |

---

## 3. Standing rules

These are not stylistic. Several were violated once already and the corrections are in
`claims_audit.md`.

1. **The fusion floor is pre-registered and must never be adjusted.**
   `config.FUSION_CONFIG["fault_authority"]`: ≥ 3 independent validation groups **and**
   honest macro-F1 ≥ 0.75, fixed **2026-09-18T07:19:23Z** at commit `7344a436`, before
   any multi-stage branch existed. If a branch fails it, the branch is INDICATIVE — the
   floor does not move. `tests/test_fusion.py` asserts an INDICATIVE branch held at
   `p_fault = 0.99` for 50 updates reaches `Warning` and stops.
2. **No invented numbers.** Every figure comes from a results JSON. Results markdown is
   *generated* (`scripts/60_render_multistage_results.py`), never hand-typed. Anything
   not run reads `NOT RUN`.
3. **Honest validation only.** Group splits, routed through
   `common.splits.iter_checked`, which asserts group separation and refuses an
   undeclared leak. No tuning against test folds — the one historical violation is
   declared in `claims_audit.md` §1.2.
4. **Leaky references are labelled `(leaky reference)` in the heading, not just the
   body.** Headings are what get screenshotted.
5. **The v5 bearing pipeline is frozen.** `drivesentinel/{config,dataset,dsp,export,
   features,folds,model,quantize,train}.py` and `scripts/01..05` get no behaviour
   changes. Extend around them.
6. **Commit and push after every phase.** 15 commits, all pushed.
7. **Measure claims before writing them down.** Two docstring claims in this project
   turned out false when measured (§5). Assert the property in a test.

---

## 4. Key findings

| Finding | Where it is documented |
|---|---|
| **D2 three-day acquisition confound.** The KAIST dataset was recorded on 2022-01-25, 2022-03-08 and 2022-08-11. Four independent metadata signals agree on the partition — `Date Created`, DAQ chassis, duration discipline, root-name style. The 3000 W motor's seven inter-coil faults are on a *different chassis and module* (`cDAQ5Mod1`) from everything else, so within that motor fault type is perfectly predictable from the instrument. | `docs/data_notes_d2.md` §7A |
| **The `i0rel_residual` probe.** Residual zero-sequence after the polarity fix is zero by Kirchhoff in a three-wire machine, so it cannot contain winding information. A 1-NN on that single scalar recovers acquisition session at **1.000** on the two clean motors and still predicts **0.750** of the fault class (baseline 0.467) — beating the full feature set under an honest split. | `docs/data_notes_d2.md` §7B |
| **All three D2 healthy recordings are from one day**, so `healthy` is not measurable under any session-aware split. | `docs/data_notes_d2.md` §7D |
| **D4 single-session verification.** Six signal-derived fingerprints (mains frequency to 0.001 Hz, DC offsets, noise floor) fail to separate the two motors; mains frequency spans only 0.081 Hz across all ten files. No D2-style confound. Stated as absence of evidence plus corroboration, **not proof** — the `.mat` headers are export times, not acquisition times. | `docs/data_notes_d4.md` §4 |
| **D3 over_temp run-identification.** The electrical-only ablation scores `over_temp` at **F1 0.796 with no temperature sensor in the feature set**. A thermal fault is not detectable from two 10 Hz phase currents, so that is the model identifying *which run* a window came from. Treat every 4-class number on D3 as contaminated by run identification. | `claims_audit.md` §1.15, `results_multistage.md` |
| **Order-resolution table.** `Δorder = 60/(T·rpm)`. At 900 rpm a 1 s window resolves 0.0667 orders; at 20 rpm it resolves 3.00, wider than the 1.89-order BPFO/BPFI gap, so those lines merge. Holding 900 rpm resolution at 20 rpm needs a **45 s** window. Arithmetic, labelled extrapolation. This is the answer to the low-speed sheave question. | `trip.resolution_report()`, `results_multistage.md`, dashboard panel 3 |
| **KI05 missed detection.** A genuinely faulty inner-race bearing the model scores **0.0089** on. It is shipped as a demo scenario on purpose, and panel 4 raises a MISSED DETECTION banner rather than letting "Normal" pass as a healthy machine. | `docs/accuracy_ceiling.md`, `artifacts/demo/bearing_KI05.npz` |

---

## 5. Closed decisions — do not re-litigate

| Decision | Why |
|---|---|
| **D2 gain calibration is computed but OFF by default** | Zeroes \|I₀\| by construction (25/45 came out exactly 0.0000), hits its ±15 % clamp on 20/45 files, and moves the negative-sequence ratio by up to 0.0504 on a dataset whose whole fault range is 0.01–0.11. `calibrate_gains()` docstring. |
| **D2 `healthy` is NOT MEASURABLE** | All three healthy recordings are from 2022-01-25. No session-aware protocol can train and test the class. **No caveated number for it anywhere.** |
| **D2 task is `inter_coil` vs `inter_turn` only** | Follows from the above. |
| **D4 label vector is not used** | The 1000/500 reconstruction matches 6/14 *file* boundaries but **0/4** measured phase-loss events. Labels come from the current-collapse rule. |
| **No CNN for D2** | 30 fault recordings across 2 usable folds. |
| **Demo NPZs are tracked in git** (3.3 MB) | A fresh clone must run the dashboard without a 7-minute retrain. |
| **P7/P8 simulation out of scope** | Stretch, cut for the MVP. |
| **Trip gating implemented but disabled** | No dataset in the roster has a speed ramp; gating would mark every window `cruise` and change nothing. |
| **B-S1 is a threshold rule, not a learned model** | One recording per (motor × class); a learned metric separates two 20 s captures. |
| **D3 drops `VDC`, `IDC`, `VD`** | Per-class means span < 0.8 ADC counts against std 1.2–1.4. |

**Two claims that were measured and found false** — both corrected, both now asserted in
tests. Do not reintroduce either:

- "D2 channels within one file differ by up to a second" — wrong, an artefact of a
  hand-written TDMS parser. All 48 files have equal-length channels.
- B-S1's "any threshold between 1 % and 30 % gives the same answer" — wrong. The
  *recording-level* verdict holds across 0.03–0.40, but the *window-level* safe band is
  only (0.048, 0.053).

---

## 6. What is NOT done

**The RTL / FPGA track, in full.** Nothing exists — no Verilog, no SystemVerilog, no HLS,
no constraints, no testbench. What *does* exist is everything needed to start:

- `artifacts/int8_export/` — 10 `.mem` files (27,024 int8 weights, 179 int32 biases),
  `scales.json`, `input_norm.json`
- `artifacts/int8_export/golden_reference.py` — self-contained integer reference
- `artifacts/int8_export/verification.json` — 99.85 % INT8-vs-float argmax agreement
  (n=2,048), accumulator headroom 21/22/23/23/21 bits, all fitting int32
- Target is a PYNQ-Z2 (Zynq XC7Z020) per `README.md`

Also outstanding, smaller:

| Item | Note |
|---|---|
| `repeat.py` not re-run after the cache rebuild | The 3-repeat ±0.0103 noise floor is pre-rebuild. ~3¼ h on this CPU. `claims_audit.md` §1.3 |
| `Sensor_raw_data_conversion_formulas.pdf` not transcribed | D3 divider ratio unknown. Moot — those channels are dropped. `pypdf` is installed |
| D2 severity as % of turns | Taken from the Mendeley description, not independently verifiable |
| Slide deck / pitch | `workflow_v2.md` §12 lists the fixes; not started |

---

## 7. How to run things

```bash
.venv/Scripts/python.exe -m pytest -q                        # 347 tests, ~35 s
.venv/Scripts/python.exe -m pytest -q -m "not slow"          # skip full-dataset reads
```

```bash
streamlit run dashboard/app.py                               # replay dashboard
```

Branch scripts — each writes its own results JSON:

```bash
.venv/Scripts/python.exe scripts/branches/10_winding.py --rebuild
.venv/Scripts/python.exe scripts/branches/20_supply.py
.venv/Scripts/python.exe scripts/branches/30_inverter_telemetry.py
```

Bearing pipeline (frozen) and demo scenarios:

```bash
.venv/Scripts/python.exe scripts/01_build_cache.py --jobs 4  # ~6 min, reads 21 GB
.venv/Scripts/python.exe scripts/02_train_lobo.py            # ~66 min on this CPU
.venv/Scripts/python.exe scripts/03_export_int8.py
.venv/Scripts/python.exe scripts/04_verify_golden.py
.venv/Scripts/python.exe scripts/demo/build_scenarios.py     # ~8 min (retrains 3 folds)
.venv/Scripts/python.exe scripts/demo/build_scenarios.py --skip-bearings   # ~10 s
```

Regenerate the results docs (never edit them by hand):

```bash
.venv/Scripts/python.exe scripts/05_render_results.py                 # docs/results.md
.venv/Scripts/python.exe scripts/60_render_multistage_results.py      # docs/results_multistage.md
```

**Timings are for this machine** — 4 threads, i3-1115G4, no GPU. The "minutes on a 3050"
figures in older docs and `folds.py:19` do not apply; re-time rather than quoting them.

---

## 8. Where to read next

`CLAUDE.md` first — it is short and holds the non-negotiables and the dataset traps.
Then `docs/workflow_v2.md` §13 for the decisions, `docs/claims_audit.md` for any number
you intend to quote, and the relevant `docs/data_notes_d*.md` before touching a dataset.
`docs/analysis_report.md` is the original pre-build audit; it is superseded where the
data notes disagree, and each disagreement is listed in the notes.
