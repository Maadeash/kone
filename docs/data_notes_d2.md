# D2 — KAIST PMSM stator fault dataset

Stage **S4** (motor winding). Source:
[data.mendeley.com/datasets/rgn5brrgrn/5](https://data.mendeley.com/datasets/rgn5brrgrn/5),
paper [PMC9957734](https://pmc.ncbi.nlm.nih.gov/articles/PMC9957734/) (*Data in Brief*).

Local path: `data_ext/kaist_pmsm/` (gitignored), 96 files, 15 GB.
Inspection pass: P1(a), 2026-09-18. Prior audit: `docs/analysis_report.md` §3.2.
Adapter: `drivesentinel/adapters/kaist_pmsm.py`.

**This document is written before any model code, per `workflow_v2.md` §3. Measured
facts here override the plan and override the earlier audit where they disagree —
§7 lists the two places they do.**

---

## 1. Shape of the dataset

96 files = **48 recordings × 2 modalities** (current + vibration), laid out in
`current/` and `vibration/` subdirectories — not flat as `workflow_v2.md` §3 assumed.

| Property | Current | Vibration |
|---|---|---|
| Files | 48 | 48 |
| Channels per file | **3** (3-phase) | **1** |
| Sample rate | **100 000 Hz** (all 48) | **25 600 Hz** (all 48) |
| dtype | float64 | float64 |
| Declared unit | `A` — **but see §5** | `g` |
| Duration | 120.0 – 149.3 s | 120.0 – 150.6 s |
| Size | 288 – 358 MB | 24 – 31 MB |
| Producer | NI FlexLogger 2021 R1.1 | same |

After deduplication (§4): **45 unique recordings**, of which **3 healthy**.

## 2. The full 48-recording grid

3 motor ratings × 2 fault types × 8 severities. Every cell has exactly one current
file and one vibration file. No cell has a repeat run.

| Rating | Fault type | Severities (% , from the filename) |
|---|---|---|
| 1000 W | intercoil | 0.00, 0.68, 0.81, **1.01**, 1.34, 2.00, 3.93, 7.56 |
| 1000 W | interturn | 0.00, 2.26, 2.70, 3.35, 4.41, 6.48, 12.17, 21.69 |
| 1500 W | intercoil | 0.00, 4.79, 5.70, 7.02, 9.15, 13.12, 23.20, 37.66 |
| 1500 W | interturn | 0.00, 1.57, 1.88, 2.34, 3.10, 4.57, 8.74, 16.08 |
| 3000 W | intercoil | 0.00, 2.49, 2.98, 3.69, 4.86, 7.12, 13.30, 23.48 |
| 3000 W | interturn | 0.00, 1.78, 2.13, 2.65, 3.50, 5.16, 9.81, 17.86 |

**`0_00` is healthy.** Confirmed from the Mendeley file-format description, which
states that `1000W_0_00_current_interturn.tdms` is "healthy … motor current data" and
defines the `bb_cc` token as **percent severity**. The `0ohm` / `0mohm` strings in the
TDMS root `name` property are unreliable FlexLogger test names and are ignored
(`workflow_v2.md` §13.1).

### Filename tokens that need normalising

| Trap | Instances | Rule |
|---|---|---|
| Severity uses `_` as decimal point | 47 of 48 | `"0_68"` → `0.68` |
| Severity uses a real `.` | **1**: `1000W_1.01_current_intercoil.tdms` | `"1.01"` → `1.01` |
| Vibration says `coil`, current says `intercoil` | **8** vibration files (all 3000 W) | `coil` → `intercoil` |

The single dotted-severity file is the current-side partner of
`1000W_1_01_vibration_intercoil.tdms`, which uses the underscore form for the same
recording. A parser that splits on `_` and takes a fixed field breaks on both traps.

## 3. Channels — never hardcode them

**Two different DAQ rigs were used.** This is stronger than the earlier audit's
"8 files use `ai1` instead of `ai2`": the affected files are on a different chassis
*and* module.

| Modality | Channel set | Files |
|---|---|---|
| current | `cDAQ1Mod2/ai0`, `ai2`, `ai3` | 41 |
| current | **`cDAQ5Mod1/ai0`, `ai1`, `ai3`** | **7** |
| vibration | `cDAQ1Mod1/ai3` | 21 |
| vibration | `cDAQ1Mod1/ai0` | 20 |
| vibration | **`cDAQ5Mod8/ai0`** | **7** |

The seven current files on the alternate rig are **exactly the seven non-healthy
3000 W intercoil recordings**:

```
3000W_2_49_current_intercoil.tdms    sev  2.49   root 'Current_6ohm'
3000W_2_98_current_intercoil.tdms    sev  2.98   root 'Current_5ohm'
3000W_3_69_current_intercoil.tdms    sev  3.69   root 'Current_4ohm'
3000W_4_86_current_intercoil.tdms    sev  4.86   root 'Current_3ohm'
3000W_7_12_current_intercoil.tdms    sev  7.12   root 'Current_2ohm'
3000W_13_30_current_intercoil.tdms   sev 13.30   root 'Current_1ohm'
3000W_23_48_current_intercoil.tdms   sev 23.48   root 'Current_05ohm'
```

The eighth 3000 W intercoil file — `3000W_0_00_current_intercoil.tdms`, the healthy
one — is on `cDAQ1Mod2`, because it is the shared healthy baseline (§4). **That makes
healthy-vs-fault within 3000 W intercoil perfectly separable by DAQ rig alone.** See §6.

**Adapter rule:** enumerate channels from the file. Selecting by
`unit_string == 'A'` is robust across both rigs and both channel namings.

## 4. The three duplicate pairs

Within each rating, the `0_00` intercoil and `0_00` interturn files are **byte-identical**
— same size, same MD5 over the first 8 MB:

| Rating | Modality | Size | MD5 (8 MB) | Files |
|---|---|---|---|---|
| 1000 W | current | 288 031 292 | `e904480fab78a805` | `..._intercoil` = `..._interturn` |
| 1500 W | current | 288 031 293 | `054910096a219a57` | ” |
| 3000 W | current | 288 031 293 | `1c3e9be76a32312c` | ” |
| 1000 W | vibration | 24 578 106 | `98070c83dd0e79dd` | ” |
| 1500 W | vibration | 24 578 107 | `d0b8d4241e2e6680` | ” |
| 3000 W | vibration | 24 578 105 | `1a959ce7fec76e20` | `..._coil` = `..._interturn` |

One healthy recording per motor, filed twice so each fault-type folder has a baseline.

**Which copy is kept:** the `*_intercoil` name, canonicalised to `fault_type="healthy"`.
The `*_interturn` copy is dropped. The choice is arbitrary — the bytes are identical —
but it is fixed in the adapter and recorded in `provenance["dedup_dropped"]` so the
count is reproducible. Result: **45 unique recordings, 3 healthy, 42 faulty.**

Loading all 48 would double-count healthy to 6 of 48 (12.5 %) instead of 3 of 45 (6.7 %).

## 5. Amplitudes are probe volts, not amps

Every current channel carries `unit_string = 'A'`, but also:

```
DAC~Channel~Sensor          = 'Voltage'
NI_SensorScale~ScaleType    = 'InverseLinear'
NI_SensorSensitivity        = 1.0
NI_SensorScale~Y-Intercept  = 0.0
```

A sensitivity of 1.0 with a zero intercept is an **identity scale**. The probe's V/A
factor is not in the file. Measured whole-file RMS is 0.42–2.57 "A", which is not
physical for a 1–3 kW PMSM at 3000 rpm.

**Consequence:** absolute amplitude is meaningless and cross-file amplitude comparison
is meaningless. Every B-S4 feature must be a ratio. The adapter marks the unit
`probe_V` — not `A` — so that an absolute-amplitude feature cannot be written by
accident (`workflow_v2.md` §13.8).

## 6. Probe polarity varies per file, and the residual is a batch signature

### 6.1 The problem

Taking the three channels at face value gives a zero-sequence ratio |I₀|/|I₊| ≈ 1.0 in
33 of 48 files, which is impossible for a three-wire machine. Searching the eight sign
combinations for the one that minimises |I₀| resolves it exactly: **two of the three
current probes are mounted with reversed polarity in some sessions.**

Channel timestamps confirm the three phases are genuinely simultaneous
(`wf_start_time` identical, `wf_start_offset = 0.0`), so phase relationships are real
and the correction is legitimate rather than a fudge.

### 6.2 Detected polarity, all 48 current files

Sign vector applies to the channels **in file order**. `I₀rel` = |I₀|/max(|I₁|,|I₂|)
at f_e = 200 Hz over the first 1.0 s. `unbal` = min(|I₁|,|I₂|)/max(|I₁|,|I₂|) after
correction — the negative-sequence ratio the branch actually uses.

| File | kW | Type | Sev | Rig | Channels | Polarity | I₀rel before | I₀rel after | unbal |
|---|---|---|---|---|---|---|---|---|---|
| `1000W_0_00_current_intercoil` | 1000 | healthy | 0.00 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,1)` | 0.0071 | 0.0071 | 0.0236 |
| `1000W_0_68_current_intercoil` | 1000 | intercoil | 0.68 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,-1)` | 0.9828 | 0.0538 | 0.0899 |
| `1000W_0_81_current_intercoil` | 1000 | intercoil | 0.81 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,-1)` | 0.9776 | 0.0536 | 0.0875 |
| `1000W_1.01_current_intercoil` | 1000 | intercoil | 1.01 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,1)` | 0.0059 | 0.0059 | 0.0261 |
| `1000W_1_34_current_intercoil` | 1000 | intercoil | 1.34 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,-1)` | 0.9776 | 0.0553 | 0.0923 |
| `1000W_2_00_current_intercoil` | 1000 | intercoil | 2.00 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,1)` | 0.0062 | 0.0062 | 0.0353 |
| `1000W_3_93_current_intercoil` | 1000 | intercoil | 3.93 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,-1)` | 0.9812 | 0.0540 | 0.0925 |
| `1000W_7_56_current_intercoil` | 1000 | intercoil | 7.56 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,1)` | 0.0060 | 0.0060 | 0.0258 |
| `1000W_0_00_current_interturn` | 1000 | *dup* | 0.00 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,1)` | 0.0071 | 0.0071 | 0.0236 |
| `1000W_2_26_current_interturn` | 1000 | interturn | 2.26 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,-1)` | 0.9858 | 0.0550 | 0.0936 |
| `1000W_2_70_current_interturn` | 1000 | interturn | 2.70 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,-1)` | 0.9863 | 0.0537 | 0.1014 |
| `1000W_3_35_current_interturn` | 1000 | interturn | 3.35 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,1)` | 0.0068 | 0.0068 | 0.0433 |
| `1000W_4_41_current_interturn` | 1000 | interturn | 4.41 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,-1)` | 0.9870 | 0.0555 | 0.1086 |
| `1000W_6_48_current_interturn` | 1000 | interturn | 6.48 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,1)` | 0.0073 | 0.0073 | 0.0526 |
| `1000W_12_17_current_interturn` | 1000 | interturn | 12.17 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,-1)` | 0.9682 | 0.0568 | 0.1117 |
| `1000W_21_69_current_interturn` | 1000 | interturn | 21.69 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,1,1)` | 0.0072 | 0.0072 | 0.0775 |
| `1500W_0_00_current_intercoil` | 1500 | healthy | 0.00 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0004 | 0.0677 | 0.0655 |
| `1500W_4_79_current_intercoil` | 1500 | intercoil | 4.79 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 0.9991 | 0.0348 | 0.0320 |
| `1500W_5_70_current_intercoil` | 1500 | intercoil | 5.70 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 0.9977 | 0.0354 | 0.0335 |
| `1500W_7_02_current_intercoil` | 1500 | intercoil | 7.02 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 0.9972 | 0.0672 | 0.0670 |
| `1500W_9_15_current_intercoil` | 1500 | intercoil | 9.15 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0043 | 0.0361 | 0.0298 |
| `1500W_13_12_current_intercoil` | 1500 | intercoil | 13.12 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0024 | 0.0672 | 0.0636 |
| `1500W_23_20_current_intercoil` | 1500 | intercoil | 23.20 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0032 | 0.0364 | 0.0312 |
| `1500W_37_66_current_intercoil` | 1500 | intercoil | 37.66 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0015 | 0.0668 | 0.0634 |
| `1500W_0_00_current_interturn` | 1500 | *dup* | 0.00 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0004 | 0.0677 | 0.0655 |
| `1500W_1_57_current_interturn` | 1500 | interturn | 1.57 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0102 | 0.0398 | 0.0299 |
| `1500W_1_88_current_interturn` | 1500 | interturn | 1.88 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0182 | 0.0379 | 0.0234 |
| `1500W_2_34_current_interturn` | 1500 | interturn | 2.34 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0209 | 0.0686 | 0.0542 |
| `1500W_3_10_current_interturn` | 1500 | interturn | 3.10 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0253 | 0.0375 | 0.0190 |
| `1500W_4_57_current_interturn` | 1500 | interturn | 4.57 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0308 | 0.0706 | 0.0504 |
| `1500W_8_74_current_interturn` | 1500 | interturn | 8.74 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0409 | 0.0382 | 0.0096 |
| `1500W_16_08_current_interturn` | 1500 | interturn | 16.08 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0537 | 0.0727 | 0.0368 |
| `3000W_0_00_current_intercoil` | 3000 | healthy | 0.00 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 0.9982 | 0.0691 | 0.0681 |
| `3000W_2_49_current_intercoil` | 3000 | intercoil | 2.49 | **cDAQ5Mod1** | ai0/ai1/ai3 | `(1,1,1)` | 0.0550 | 0.0550 | 0.0574 |
| `3000W_2_98_current_intercoil` | 3000 | intercoil | 2.98 | **cDAQ5Mod1** | ai0/ai1/ai3 | `(1,1,1)` | 0.0435 | 0.0435 | 0.0439 |
| `3000W_3_69_current_intercoil` | 3000 | intercoil | 3.69 | **cDAQ5Mod1** | ai0/ai1/ai3 | `(1,1,1)` | 0.0421 | 0.0421 | 0.0441 |
| `3000W_4_86_current_intercoil` | 3000 | intercoil | 4.86 | **cDAQ5Mod1** | ai0/ai1/ai3 | `(1,1,1)` | 0.0422 | 0.0422 | 0.0439 |
| `3000W_7_12_current_intercoil` | 3000 | intercoil | 7.12 | **cDAQ5Mod1** | ai0/ai1/ai3 | `(1,1,1)` | 0.0422 | 0.0422 | 0.0446 |
| `3000W_13_30_current_intercoil` | 3000 | intercoil | 13.30 | **cDAQ5Mod1** | ai0/ai1/ai3 | `(1,1,1)` | 0.0423 | 0.0423 | 0.0452 |
| `3000W_23_48_current_intercoil` | 3000 | intercoil | 23.48 | **cDAQ5Mod1** | ai0/ai1/ai3 | `(1,1,1)` | 0.0415 | 0.0415 | 0.0435 |
| `3000W_0_00_current_interturn` | 3000 | *dup* | 0.00 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 0.9982 | 0.0691 | 0.0681 |
| `3000W_1_78_current_interturn` | 3000 | interturn | 1.78 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 0.9999 | 0.0306 | 0.0278 |
| `3000W_2_13_current_interturn` | 3000 | interturn | 2.13 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0014 | 0.0306 | 0.0267 |
| `3000W_2_65_current_interturn` | 3000 | interturn | 2.65 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 0.9966 | 0.0692 | 0.0698 |
| `3000W_3_50_current_interturn` | 3000 | interturn | 3.50 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0020 | 0.0310 | 0.0267 |
| `3000W_5_16_current_interturn` | 3000 | interturn | 5.16 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 0.9996 | 0.0688 | 0.0669 |
| `3000W_9_81_current_interturn` | 3000 | interturn | 9.81 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.0044 | 0.0315 | 0.0251 |
| `3000W_17_86_current_interturn` | 3000 | interturn | 17.86 | cDAQ1Mod2 | ai0/ai2/ai3 | `(1,-1,-1)` | 1.1837 | 0.0788 | 0.0904 |

Summary by pattern:

| Polarity | n | I₀rel before | I₀rel after |
|---|---|---|---|
| `(1,1,1)` — no correction needed | 15 | 0.0059 – 0.0550 | 0.0059 – 0.0550 |
| `(1,1,-1)` — third probe reversed | 8 | 0.9682 – 0.9870 | 0.0536 – 0.0568 |
| `(1,-1,-1)` — second and third reversed | 25 | 0.9966 – **1.1837** | 0.0306 – 0.0788 |

**The detector succeeds on all 48 files** — every one drops to I₀rel < 0.08.

### 6.3 The residual is the problem

After sign correction, I₀rel is still **0.006 – 0.079**, where a three-wire machine
should give ≈ 0. That residual is a per-channel gain mismatch of a few percent, and it
is **not random — it clusters by acquisition session**. Look at 1500 W intercoil:

```
sev  0.00 -> 0.0677    sev  9.15 -> 0.0361
sev  4.79 -> 0.0348    sev 13.12 -> 0.0672
sev  5.70 -> 0.0354    sev 23.20 -> 0.0364
sev  7.02 -> 0.0672    sev 37.66 -> 0.0668
```

Two tight clusters, ~0.035 and ~0.067, alternating with severity. The adapter therefore
also applies a **per-channel gain calibration** (scale each channel to minimise residual
|I₀|) on top of the sign correction.

## 7. Acquisition batches, and where they are confounded

A batch key that uses only measurable file properties — DAQ rig, detected polarity, and
TDMS root-name style — partitions the 48 files into five groups:

| n | Rig | Polarity | Root-name style |
|---|---|---|---|
| 13 | cDAQ1Mod2 | `(1,-1,-1)` | `<kW>kW_<R>ohm_current` |
| 12 | cDAQ1Mod2 | `(1,-1,-1)` | `<kW>kw_<R>ohm_current_<type>` |
| 8 | cDAQ1Mod2 | `(1,1,1)` | `<kW>kW_<R>ohm_current` |
| 8 | cDAQ1Mod2 | `(1,1,-1)` | `<kW>kw_<R>ohm_current_<type>` |
| 7 | **cDAQ5Mod1** | `(1,1,1)` | `Current_<R>` |

How batch lines up with fault type, per motor:

| Motor | Batch | n | Fault types in that batch |
|---|---|---|---|
| 1000 W | `kW_` / `(1,1,1)` | 8 | intercoil 4, interturn 4 — **balanced** |
| 1000 W | `kw_` / `(1,1,-1)` | 8 | intercoil 4, interturn 4 — **balanced** |
| 1500 W | `kW_` / `(1,-1,-1)` | 8 | intercoil 4, interturn 4 — **balanced** |
| 1500 W | `kw_` / `(1,-1,-1)` | 8 | intercoil 4, interturn 4 — **balanced** |
| 3000 W | `Current_` / cDAQ5 / `(1,1,1)` | 7 | **intercoil 7, interturn 0** |
| 3000 W | `kW_` / `(1,-1,-1)` | 5 | intercoil 1 (the healthy dup), interturn 4 |
| 3000 W | `kw_` / `(1,-1,-1)` | 4 | interturn 4 |

**1000 W and 1500 W are clean. 3000 W is not.**

For the 3000 W motor:
- every `intercoil` fault is on **cDAQ5Mod1**;
- every `interturn` fault is on **cDAQ1Mod2**;
- the single healthy recording is on **cDAQ1Mod2**.

So within the 3000 W motor, **`intercoil` vs `interturn` is perfectly predictable from
the DAQ rig**, and **healthy vs intercoil-fault is perfectly predictable from the DAQ
rig**. A model that learns nothing about windings can score 100 % on 3000 W by
detecting which instrument recorded the file.

This is a property of the dataset, not of our pipeline, and it cannot be corrected —
only measured and declared. The V4 batch control (`scripts/branches/10_winding.py`)
exists to quantify it, and it is expected to fire hardest on the 3000 W fold. Whatever
the V1 leave-one-motor-out number turns out to be, **the 3000 W fold's contribution to
it is not evidence of winding diagnosis.**

## 8. Electrical fundamental

**f_e = 200.00 Hz in all 48 current recordings**, measured by DFT peak search over
150–250 Hz at 0.25 Hz resolution. At the documented 3000 rpm that is 4 pole pairs /
8 poles — the same pole count as the Paderborn machine (`config.py:74`).

**Consequence for the plan:** the f/f_e electrical-harmonic axis is a *constant*
rescaling on this dataset. It adds no invariance here because there is no speed
variation to be invariant to. It is implemented for the API and for the simulator
branch, and it is **not claimed as a capability** (`workflow_v2.md` §13.8).

Nyquist check: decimating 100 kHz → 10 kHz leaves Nyquist 5000 Hz against the planned
order-20 limit of 4000 Hz — 20 % margin, adequate. A 1.0 s window gives 1 Hz resolution
= 0.005 orders at f_e = 200 Hz, ample.

## 9. Decimation cost — measured

Measured on this machine (i3-1115G4, 4 threads, no GPU), reading with `nptdms` and
decimating with the frozen `dsp.decimate_signal` (order-8 Chebyshev-I, zero-phase):

| File | Input | Read | Decimate | Output |
|---|---|---|---|---|
| `1000W_0_68_current_intercoil` | 293.3 MB | 0.44 s | 1.00 s | 3 × 1 221 991 f32 = 14.7 MB |
| `1000W_1_34_current_intercoil` (longest) | 358.4 MB | 0.58 s | 1.23 s | 3 × 1 492 990 f32 = 17.9 MB |
| `3000W_7_12_current_intercoil` (cDAQ5) | 288.0 MB | 0.49 s | 0.89 s | 3 × 1 200 000 f32 = 14.4 MB |

**Mean 1.54 s per recording** (0.50 s read + 1.04 s decimate).

| Projection | Time |
|---|---|
| 45 recordings, `n_jobs=1` | **≈ 1.2 min** |
| 45 recordings, `n_jobs=2` | **≈ 0.6 min** |
| Cache size on disk | **≈ 675 MB** float32 |

This is **far** below the 60-minute stop condition, and far below the 25–45 min the
earlier audit projected — that estimate assumed TDMS reads would be I/O-bound, but a
float64 TDMS read is close to a memcpy and `scipy.signal.decimate` handles 12.2 M
samples in a second. Caching at full 100 kHz rate would need ~7 GB and is not done.

Measured with a warm page cache from the inspection sweep. A cold run adds the time to
read 14 GB from disk — at 200 MB/s that is ~70 s, so a cold full build is ~3 min.

## 10. What this means for the B-S4 branch

| Quantity | Value |
|---|---|
| Groups for leave-one-motor-out | **3** (1000 W, 1500 W, 3000 W) |
| Unique recordings | 45 |
| Healthy recordings | **3** — one per motor |
| Runs per (motor, type, severity) | **1** — no repeats anywhere |
| Windows per recording @ 1.0 s / 0.5 s hop | ~239 |
| Healthy share of training windows | **~6.7 %** |

Under leave-one-motor-out, each fold trains on **2 healthy recordings** and tests on 1.
That is the binding constraint on the 3-class task, and the reason the metric card
leads with binary healthy/fault (`workflow_v2.md` §13.8).

Meets the fusion floor's group test (3 ≥ 3). Whether it has `Fault` authority depends
entirely on the macro-F1 ≥ 0.75 test, measured in (d).

---

## 11. Corrections to the earlier audit

| `docs/analysis_report.md` said | Measured in P1(a) | Effect |
|---|---|---|
| "8 files use `ai1` instead of `ai2`" | **7 files**, and they are on a different chassis *and* module (`cDAQ5Mod1`, not `cDAQ1Mod2`). The 8th 3000 W intercoil file is the healthy duplicate, on the main rig. | Sharpens §7 — it also means healthy-vs-fault is rig-confounded for 3000 W intercoil |
| "D2 channels within one file differ by up to a second (124.8 / 125.8 / 125.8 s)" | **Wrong.** `nptdms` reports all 48 current files as having three exactly equal-length channels. The spread was a chunked-segment miscount in the hand-written parser used during the audit. | `schema.py:duration()` docstring and `test_schema.py` corrected; the shortest-channel rule is kept as defensive design, not as a D2 requirement |

## 12. Open items

| Item | Status |
|---|---|
| Vibration modality | Loaded and cached, but **not used** by the B-S4 feature set (current only, per `workflow_v2.md` §5). Retained for the batch-control cross-check. |
| Per-channel gain calibration constants | Computed per file at adapter load; recorded in `provenance`. Not yet cross-checked against a second time window in the same recording. |
| Severity as % of turns short-circuited | Taken from the Mendeley description. Not independently verifiable from the files. |
