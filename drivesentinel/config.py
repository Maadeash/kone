"""
DriveSentinel v5 -- config.py
=============================
Single source of truth for paths, physical constants and the signal-processing
contract. Everything downstream imports from here; nothing else hard-codes a
sample rate, a bearing label or a fault-frequency coefficient.

WHY THE NUMBERS ARE WHAT THEY ARE
---------------------------------
Sample rates and channel names come from the Paderborn measuring logs, verified
against the .mat contents (all 32 bearings x 4 conditions loaded and checked).

POLE_PAIRS = 4 is the one value that lets the whole order-normalisation scheme
work.  The drive motor is a Hanning SD4CDu8S PMSM, 8 poles, fed by a KEB
Combivert inverter.  Because it is a *permanent-magnet* machine there is no
slip, so shaft speed follows the electrical fundamental exactly:

    f_shaft = f_elec / POLE_PAIRS

Measured directly on K001/N15_M07_F10_K001_1.mat: fundamental 100.00 Hz at
1499.7 rpm (= 25.0 Hz shaft), and 60 Hz at 900 rpm.  That is what makes it
possible to recover shaft speed from the current spectrum alone at deployment,
where no tachometer exists.

  >>> DEPLOYMENT CAVEAT <<<
  On an *induction* machine this identity does not hold -- rotor speed lags the
  field by the slip.  Porting this pipeline to an induction motor requires a
  slip estimator (e.g. from the rotor-slot or eccentricity harmonics) before the
  order axis is trustworthy.  Flagged here because it is invisible on this
  dataset and would silently corrupt the order axis on the next one.
"""

from __future__ import annotations

import os
from typing import Dict, List

# ===========================================================================
# 1. PATHS
# ===========================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# Cache paths carry the feature-set name so the v1 (2-channel) and v2
# (5-channel) caches coexist and cannot be confused for one another.  The
# contract fingerprint inside each file is the real guard; this just keeps the
# two from overwriting each other on disk.
def _cache_paths(feature_set: str):
    return (os.path.join(ARTIFACT_DIR, f"order_spectra_{feature_set}.npz"),
            os.path.join(ARTIFACT_DIR, f"order_spectra_{feature_set}_meta.parquet"))
FOLDS_PATH = os.path.join(ARTIFACT_DIR, "folds_lobo.json")
RUN_DIR = os.path.join(ARTIFACT_DIR, "runs")
EXPORT_DIR = os.path.join(ARTIFACT_DIR, "int8_export")

# ===========================================================================
# 2. ACQUISITION -- what the rig actually recorded
# ===========================================================================

FS_FAST = 64_000.0          # phase_current_1/2 and vibration_1
FS_MECH = 4_000.0           # force, speed, torque  (training-time reference only)
RECORD_SECONDS = 4.0

CH_CURRENT = ("phase_current_1", "phase_current_2")
CH_VIBRATION = "vibration_1"
CH_SPEED = "speed"          # ground-truth rpm; used ONLY to validate the
                            # current-derived speed estimate, never as a feature

# ===========================================================================
# 3. MOTOR / BEARING PHYSICS
# ===========================================================================

POLE_PAIRS = 4              # Hanning SD4CDu8S, 8-pole PMSM -- see module docstring

# FAG 6203 deep-groove ball bearing.  Ball count and raceway diameters are
# stated in the per-bearing damage-profile PDFs; ball diameter and pitch
# diameter are literature-typical for a 6203 and are NOT given by the KAt
# documentation.  The order coefficients below are therefore accurate to about
# +/- 1 %, which is why the model is fed a spectrum and left to find the lines
# rather than being handed a +/- 8 Hz integration band around each one.
BEARING_GEOMETRY = {
    "n_balls": 8,
    "ball_diameter_mm": 6.75,
    "pitch_diameter_mm": 28.55,
    "contact_angle_deg": 0.0,
}

# Characteristic frequencies as multiples of shaft rotation frequency (orders).
# Computed once in dsp.fault_orders(); repeated here as documentation and as the
# expected values the self-test asserts against.
FAULT_ORDERS_NOMINAL = {
    "ftf": 0.3818,          # cage
    "bsf": 1.9966,          # ball spin
    "bpfo": 3.0544,         # outer race
    "bpfi": 4.9456,         # inner race
}

# ===========================================================================
# 4. SIGNAL-PROCESSING CONTRACT
# ===========================================================================
# These five numbers define the accelerator's input and must match between the
# Python model, the golden reference and any later RTL.

WINDOW_SECONDS = 1.0        # 1 Hz native resolution; 4x better than the v4
                            # 0.25 s window, which merged BSF h1 with shaft h2
HOP_SECONDS = 0.5           # 7 windows per 4 s recording
N_ORDER_BINS = 512
ORDER_MAX = 16.0            # covers BPFI h3 = 14.8 orders
# Input channel set.  "v1" is the 2-channel contract v5.0 shipped with; "v2"
# adds three channels that carry information envelope analysis throws away.
#
#   v1  0 current MCSA sideband fold          (order axis)
#       1 vibration envelope, 1-10 kHz band   (order axis)
#
#   v2  0 current MCSA sideband fold          (order axis)
#       1 current raw spectrum                (order axis) -- shaft harmonics
#         appearing directly in current, e.g. eccentricity, which the sideband
#         fold about f_elec cannot see
#       2 vibration envelope, LOW resonance   (order axis)
#       3 vibration envelope, HIGH resonance  (order axis) -- splitting the band
#         replaces the "a kurtogram would be better but is not fixed-function"
#         compromise with a fixed two-band choice the model can weigh itself
#       4 vibration raw log-frequency spectrum -- WHERE the resonances sit and
#         how hard they are driven.  Deliberately on an absolute-frequency axis,
#         not an order axis: structural resonances do not move with shaft speed,
#         so frequency is the physically correct axis for them.
# MEASURED, over 3 repeats each, with the inner validation split removed so the
# run-to-run standard deviation is ~0.01 rather than ~0.11:
#
#     v2 (5 ch) + strong augmentation : 0.8018 window / 0.8332 per-recording
#     v1 (2 ch) + strong augmentation : 0.7823 window / 0.8122 per-recording
#     v2 (5 ch) + weak augmentation   : 0.7611 window / 0.7784 per-recording
#
# v2 ships.  Note it only wins WITH strong augmentation -- more channels means
# more to overfit, so it needs the extra regularisation.  An earlier comparison
# concluded v2 was 10 points WORSE; that was two single draws against a +/-0.11
# noise floor, using weak augmentation.  Both errors at once.  Hence
# scripts/experiments/repeat.py, and hence measuring the spread before comparing.
# Read from the environment so that joblib WORKER PROCESSES inherit it.
# Workers re-import this module fresh; setting C.FEATURE_SET in the parent has
# no effect on them, which silently produced a 2-channel cache written to the
# 5-channel path.  DRIVESENTINEL_FEATURE_SET is the only safe way to switch.
FEATURE_SET = os.environ.get("DRIVESENTINEL_FEATURE_SET", "v2")

VIBRATION_BAND_LOW_HZ = (500.0, 2_500.0)
VIBRATION_BAND_HIGH_HZ = (2_500.0, 10_000.0)
RAW_SPECTRUM_HZ = (200.0, 16_000.0)      # log-spaced, ends below the 16 kHz
                                         # inverter switching tone

_CHANNELS = {"v1": 2, "v2": 5}
N_INPUT_CHANNELS = _CHANNELS[FEATURE_SET]

CACHE_PATH, CACHE_META_PATH = _cache_paths(FEATURE_SET)

ORDER_RESOLUTION = ORDER_MAX / N_ORDER_BINS      # 0.03125 orders/bin

# MCSA sideband fold (current branch).  Because f_elec = POLE_PAIRS * f_shaft
# exactly, supply harmonics fall on a fixed order grid -- the fundamental
# reappears at order 2*POLE_PAIRS, the 2nd harmonic at orders POLE_PAIRS and
# 3*POLE_PAIRS, and so on.  They are notched out of the magnitude spectrum
# before folding; see dsp.current_sideband_order_spectrum for the derivation.
SUPPLY_HARMONICS_NOTCHED = 8
SUPPLY_NOTCH_HALFWIDTH_HZ = 2.5     # ~2-3 bins at a 1 s window: enough to remove
                                    # a Hann main lobe, narrow enough that BPFI
                                    # at 4.95 orders stays 30 bins clear

VIBRATION_BAND_HZ = (1_000.0, 10_000.0)  # broad resonance band.  A per-recording
                                    # kurtogram would be better but is not a
                                    # fixed-function operation, so a fixed band
                                    # is used to keep the FPGA path static.

# Decimation applied before demodulation.  64 kHz -> 16 kHz keeps the whole
# vibration resonance band (Nyquist 8 kHz... see DECIMATE note below).
DECIMATE_CURRENT = 16       # 64 kHz -> 4 kHz  (f_elec <= 100 Hz, band <= 600 Hz)
DECIMATE_VIBRATION = 2      # 64 kHz -> 32 kHz (keeps the 10 kHz band edge, and
                            # stays above the 16 kHz inverter switching tone so
                            # it is filtered rather than aliased)

# Envelope-spectrum conditioning, applied identically at train and inference:
#   1. magnitude of the analytic envelope's rFFT
#   2. divide by the spectrum's own median  -> relative line prominence
#   3. log1p                                -> compress the dynamic range
# Step 2 makes the input invariant to absolute sensor gain and to bearing-to-
# bearing amplitude offsets, which is where identity leakage lives.  It does NOT
# repeat the v4 norm/rank mistake: that normalised feature values *across
# windows of a recording*, which discarded ~89 % of the discriminative spread.
# This normalises *within one spectrum*, preserving its shape.
SPECTRUM_EPS = 1e-12

# ===========================================================================
# 5. LABELS
# ===========================================================================

LABELS = ["healthy", "inner_race", "outer_race"]      # deployable 3-class task
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}

# KB23/KB24/KB27 carry combined inner+outer damage.  Three physical bearings
# cannot support a class under leave-one-bearing-out, and v4 measured RF's
# combined-class F1 at exactly 0.000.  They are excluded by default and the
# exclusion is reported, not hidden.
EXCLUDED_BEARINGS = ["KB23", "KB24", "KB27"]

BEARING_LABELS: Dict[str, str] = {
    "K001": "healthy", "K002": "healthy", "K003": "healthy",
    "K004": "healthy", "K005": "healthy", "K006": "healthy",
    "KA01": "outer_race", "KA03": "outer_race", "KA04": "outer_race",
    "KA05": "outer_race", "KA06": "outer_race", "KA07": "outer_race",
    "KA08": "outer_race", "KA09": "outer_race", "KA15": "outer_race",
    "KA16": "outer_race", "KA22": "outer_race", "KA30": "outer_race",
    "KI01": "inner_race", "KI03": "inner_race", "KI04": "inner_race",
    "KI05": "inner_race", "KI07": "inner_race", "KI08": "inner_race",
    "KI14": "inner_race", "KI16": "inner_race", "KI17": "inner_race",
    "KI18": "inner_race", "KI21": "inner_race",
    "KB23": "combined", "KB24": "combined", "KB27": "combined",
}

# Damage origin.  Never pool these into one accuracy figure: v4 measured
# artificial -> real transfer at 37.9 % on current features, because engraver
# and EDM damage is geometrically clean in a way fatigue pitting is not.
ARTIFICIAL_BEARINGS = [
    "KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09",
    "KI01", "KI03", "KI05", "KI07", "KI08",
]


def damage_origin(bearing: str) -> str:
    if BEARING_LABELS.get(bearing) == "healthy":
        return "none"
    return "artificial" if bearing in ARTIFICIAL_BEARINGS else "real"


# ===========================================================================
# 6. KNOWN DATA DEFECTS  -- see docs/data_notes.md
# ===========================================================================

# Byte-identical duplicate of N09_M07_F10_KA04_18.mat.  The KA04 measuring log
# documents the substitution: run 17 was corrupt and was replaced with a copy of
# run 18.  Verified here by array comparison on every channel.  Dropped so the
# same window does not appear twice inside one bearing.
DUPLICATE_RECORDINGS = {"N09_M07_F10_KA04_17.mat"}

# Structurally corrupt: correct file size (8,714,872 B) and a valid MATLAB 5.0
# header, but the variable stream does not parse ("Expecting matrix here").
# Independently corroborated -- the v4 feature cache also holds only 79 KA08
# recordings, so the previous pipeline hit the same file and silently skipped it.
CORRUPT_RECORDINGS = {"N15_M01_F10_KA08_2.mat"}

UNUSABLE_RECORDINGS = DUPLICATE_RECORDINGS | CORRUPT_RECORDINGS

# This copy of the dataset holds 2,548 .mat files.  KI14 is short 12 recordings
# (68 instead of 80) relative to the copy the v4 cache was built from, which had
# 2,559.  The v4 cache is therefore NOT reproducible on this machine; the v5
# cache is built from what is actually present and records its own count.
EXPECTED_RECORDING_COUNT = 2_548 - len(UNUSABLE_RECORDINGS)

# ===========================================================================
# 7. MODEL / TRAINING
# ===========================================================================

SEED = 42

MODEL_CONFIG = {
    "in_channels": N_INPUT_CHANNELS,
    "n_classes": len(LABELS),
    # (out_channels, kernel_size, stride)
    "conv_stages": ((16, 9, 2), (32, 7, 2), (64, 5, 2), (64, 3, 2)),
}

TRAIN_CONFIG = {
    "epochs": 20,                  # a completed OneCycle at 20 epochs matches a
                                   # truncated 60-epoch one and costs a third of
                                   # the time; see the note on n_seeds below for
                                   # why the difference is not measurable anyway
    "batch_size": 1024,
    "lr": 3e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    # 0 = NO inner validation split.
    #
    # MEASURED: with 1 validation bearing per class, three repeats of an
    # otherwise identical config scored 0.6041 / 0.7790 / 0.8018 -- std 0.108.
    # Each already averaged 3 models internally, so that spread was not weight
    # initialisation: the base seed also picks WHICH bearings are held out, and
    # with 6 healthy specimens, removing a different one changes what the model
    # can learn about the class.  Dropping the split removes that variance
    # source and returns 3 bearings to the fit set.
    #
    # Affordable only because the schedule is a completed OneCycle with early
    # stopping off -- there is nothing left for a validation set to decide.
    "val_bearings_per_class": 0,
    "augment": True,

    # EARLY STOPPING IS OFF, DELIBERATELY.
    # OneCycleLR anneals the learning rate across the whole `epochs` budget, so
    # a run cut short at epoch 8 of 60 never reaches the low-LR refinement phase
    # -- it stops near peak LR with the weights still moving. The v5.0 sweep
    # showed exactly that: best_epoch was 5-14 for most folds and accuracy sat
    # at 0.777. Letting the schedule complete and keeping the best-validation
    # snapshot gets the refinement without the truncation.
    "early_stop": False,
    "patience": 20,                # only consulted when early_stop is True

    # Inference-time aggregation. Both are free at training time and cost the
    # FPGA nothing: the accelerator still runs one window at a time, and the
    # averaging happens in the PS over a rolling buffer.
    # MEASURED, and the only changes that survived re-running.
    #
    # Per-bearing accuracy has std ~0.36 over 29 bearings, so the SEM on any
    # pooled figure is ~0.067, and cuDNN's non-deterministic convolution
    # backward moves an identical re-run by several points -- the v50 recipe
    # scored 0.7773 once and 0.6972 later with the same seed and data.  So
    # single-run recipe differences below ~8 points are not evidence, and
    # tuning against them is fitting noise.
    #
    # Seed ensembling and TTA are different in kind: they REDUCE variance
    # rather than gambling on a lucky configuration, and they held up on
    # re-running (0.7417 vs 0.6972 for the same schedule without them).
    # MEASURED AGAIN once the validation-split variance was removed: ensembling
    # and TTA now buy essentially nothing (0.7837 with 3 seeds + TTA vs 0.7822
    # with neither -- about 1 standard deviation apart).  Their earlier "+4.5
    # points" was variance reduction, and removing the inner validation split
    # provides that far more cheaply.
    #
    # So a SINGLE model ships, which also means the exported INT8 weights match
    # the reported accuracy exactly instead of being a weaker artefact than the
    # number quoted for them.
    "n_seeds": 1,
    "tta_shifts": (0,),
}

# Aggregating a recording's windows by averaging SOFTMAX rather than taking a
# hard majority vote. A window the model is unsure about should not carry the
# same weight as one it is certain of, and hard voting throws that away.
SOFT_VOTE = True

AUGMENT_CONFIG = {
    "order_jitter_bins": 4,     # +/- 4 bins ~ +/- 0.125 orders; covers the ~1 %
                                # uncertainty in the assumed 6203 geometry
    "amplitude_scale": 0.25,    # +/- 25 % per-channel gain, mimics sensor and
                                # bearing-to-bearing amplitude variation
    "channel_dropout": 0.15,    # zero one channel sometimes, so the model has
                                # to use both current and vibration instead of
                                # leaning on whichever is easier that fold
    "noise_std": 0.05,
    "prob": 0.8,
}

QUANT_CONFIG = {
    "weight_bits": 8,
    "act_bits": 8,
    "per_channel_weights": True,   # conv layers lose materially more to
                                   # per-tensor scaling than dense layers do
    "calibration_windows": 4096,
    "activation_percentile": 99.9,  # clip outliers rather than using the max,
                                    # which a single spike would otherwise set
}


def ensure_dirs() -> None:
    for d in (ARTIFACT_DIR, RUN_DIR, EXPORT_DIR):
        os.makedirs(d, exist_ok=True)


def summary() -> str:
    return (
        f"window {WINDOW_SECONDS}s hop {HOP_SECONDS}s | "
        f"{N_INPUT_CHANNELS} x {N_ORDER_BINS} bins over 0-{ORDER_MAX} orders "
        f"({ORDER_RESOLUTION:.5f} orders/bin) | classes {LABELS}"
    )
