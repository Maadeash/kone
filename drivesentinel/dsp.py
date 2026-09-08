"""
DriveSentinel v5 -- dsp.py
==========================
The signal-processing front end, in the exact order the FPGA will implement it:

    raw 64 kHz
      -> decimate
      -> band-pass around the carrier
      -> Hilbert analytic envelope
      -> rFFT  (envelope spectrum, in Hz)
      -> resample onto a SHAFT-ORDER axis
      -> median-normalise + log1p
    = one (2, 512) tensor per window

WHY AN ORDER AXIS AND NOT HERTZ
-------------------------------
Bearing fault lines sit at fixed multiples of shaft rotation frequency, so in
hertz they move when the motor speeds up: BPFO is 76.4 Hz at 1500 rpm and
45.8 Hz at 900 rpm.  A network fed a hertz axis has to learn both, and the
cheapest way for it to do that is to first identify the operating condition
from the supply fundamental -- which is exactly the confound the v3 methodology
flags as untested in section III-G.

Resampling to orders removes the speed dependence entirely: BPFO sits at
3.05 orders at every speed, so 900 rpm and 1500 rpm produce the same picture and
there is nothing for the network to key on but the fault.

WHERE SHAFT SPEED COMES FROM
----------------------------
From the current spectrum, never from the tachometer.  The rig has a Magtrol
TM305 measuring shaft; a lift does not.  estimate_shaft_speed() finds the
electrical fundamental and divides by the pole-pair count, which is exact for
this PMSM (no slip).  The measured `speed` channel is used in exactly one place
-- validate_speed_estimate() -- to confirm the estimator is right, and never as
a model input.

TESTABILITY
-----------
Every stage is a pure function of arrays.  tests/test_dsp.py plants a synthetic
BPFO modulation at a known depth and asserts it is recovered at the right ORDER
bin at two different shaft speeds; that is the test that would catch a wrong
pole-pair count, an off-by-one in the order axis, or a broken envelope.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt, decimate, hilbert

from . import config as C


# ===========================================================================
# 1. BEARING KINEMATICS
# ===========================================================================

def fault_orders(geometry: Dict = None) -> Dict[str, float]:
    """
    Characteristic bearing frequencies as multiples of shaft rotation frequency.

    Returning ORDERS rather than hertz is deliberate -- these are constants of
    the bearing, independent of running speed, which is the whole premise of the
    order axis.
    """
    g = geometry or C.BEARING_GEOMETRY
    z = g["n_balls"]
    ratio = (g["ball_diameter_mm"] / g["pitch_diameter_mm"]) * np.cos(
        np.deg2rad(g["contact_angle_deg"])
    )
    return {
        "ftf": 0.5 * (1.0 - ratio),
        "bsf": (1.0 / (2.0 * g["ball_diameter_mm"] / g["pitch_diameter_mm"]))
               * (1.0 - ratio ** 2),
        "bpfo": 0.5 * z * (1.0 - ratio),
        "bpfi": 0.5 * z * (1.0 + ratio),
    }


def order_to_bin(order: float) -> int:
    """Index of the order-axis bin a given order falls in."""
    return int(round(order / C.ORDER_RESOLUTION))


# ===========================================================================
# 2. SPEED ESTIMATION  -- from current alone
# ===========================================================================

def estimate_electrical_fundamental(current: np.ndarray, fs: float,
                                    lo: float = 20.0, hi: float = 200.0) -> float:
    """
    Dominant spectral peak of the stator current inside a plausible drive band.

    Parabolic interpolation over the three bins around the peak recovers
    sub-bin accuracy, which matters: a 1 Hz error at 60 Hz propagates to a 1.7 %
    error on the order axis, enough to smear BPFI h3 across two bins.
    """
    x = np.asarray(current, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    mag = np.abs(rfft(x * np.hanning(n)))
    freqs = rfftfreq(n, 1.0 / fs)

    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        raise ValueError(f"no spectral bins in {lo}-{hi} Hz at fs={fs}")
    idx = np.flatnonzero(band)[np.argmax(mag[band])]

    if 0 < idx < mag.size - 1:
        a, b, c = mag[idx - 1], mag[idx], mag[idx + 1]
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-20 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    return float((idx + delta) * (freqs[1] - freqs[0]))


def estimate_shaft_speed(current: np.ndarray, fs: float) -> Tuple[float, float]:
    """
    Returns (shaft_frequency_hz, electrical_fundamental_hz).

    Exact for the Paderborn PMSM -- see config.POLE_PAIRS for why, and for the
    induction-motor caveat that applies when this moves off this dataset.
    """
    f_elec = estimate_electrical_fundamental(current, fs)
    return f_elec / C.POLE_PAIRS, f_elec


def validate_speed_estimate(estimated_fr_hz: float, speed_channel: np.ndarray) -> float:
    """
    Relative error of the current-derived shaft speed against the tachometer.

    Diagnostic only.  Called during cache building so a systematic error shows
    up as a report line rather than as silently smeared spectra.
    """
    measured_fr = float(np.mean(speed_channel)) / 60.0
    if measured_fr <= 0:
        return float("nan")
    return abs(estimated_fr_hz - measured_fr) / measured_fr


# ===========================================================================
# 3. FILTERS AND ENVELOPE
# ===========================================================================

def _bandpass_sos(lo: float, hi: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999)
    if lo_n >= hi_n:
        raise ValueError(f"bad band {lo}-{hi} Hz at fs={fs}")
    return butter(order, [lo_n, hi_n], btype="bandpass", output="sos")


def analytic_envelope(signal: np.ndarray, fs: float,
                      band: Tuple[float, float]) -> np.ndarray:
    """
    Band-pass then take |analytic signal|, minus its DC.

    The DC removal matters: the envelope of a band-passed carrier is dominated
    by a large constant term, and leaving it in would put all the spectrum's
    energy in bin 0 and flatten everything the median normalisation then does.
    """
    sos = _bandpass_sos(band[0], band[1], fs)
    filtered = sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))
    env = np.abs(hilbert(filtered))
    return env - env.mean()


def decimate_signal(signal: np.ndarray, factor: int) -> np.ndarray:
    """
    Anti-aliased decimation.  scipy applies an order-8 Chebyshev-I low-pass
    before downsampling, which is what keeps the 16 kHz inverter switching tone
    from folding back into the analysis band.
    """
    if factor <= 1:
        return np.asarray(signal, dtype=np.float64)
    return decimate(np.asarray(signal, dtype=np.float64), factor,
                    ftype="iir", zero_phase=True)


# ===========================================================================
# 4. ORDER SPECTRUM
# ===========================================================================

def envelope_order_spectrum(envelope: np.ndarray, fs: float, fr_hz: float,
                            n_bins: int = None, order_max: float = None) -> np.ndarray:
    """
    Envelope spectrum resampled onto a shaft-order axis.

    The resampling is a linear interpolation of the magnitude spectrum onto
    `n_bins` evenly spaced orders in [0, order_max).  It creates no information
    -- it aligns the axis so that the same fault sits in the same bin regardless
    of running speed, which is what lets one set of convolution kernels work at
    both 900 and 1500 rpm.
    """
    n_bins = n_bins or C.N_ORDER_BINS
    order_max = order_max or C.ORDER_MAX

    x = np.asarray(envelope, dtype=np.float64)
    n = x.size
    mag = np.abs(rfft(x * np.hanning(n))) * (2.0 / n)
    freqs = rfftfreq(n, 1.0 / fs)

    if fr_hz <= 0:
        return np.zeros(n_bins, dtype=np.float64)

    target_orders = np.arange(n_bins, dtype=np.float64) * (order_max / n_bins)
    target_hz = target_orders * fr_hz
    # Orders beyond the envelope's Nyquist cannot be observed; np.interp would
    # clamp to the last bin and fabricate a plateau, so zero them explicitly.
    out = np.interp(target_hz, freqs, mag, left=0.0, right=0.0)
    out[target_hz > freqs[-1]] = 0.0
    return out


def current_sideband_order_spectrum(current: np.ndarray, fs: float, fr_hz: float,
                                    f_elec: float, n_bins: int = None,
                                    order_max: float = None) -> np.ndarray:
    """
    MCSA sideband energy as a function of shaft order.

    WHY NOT HILBERT DEMODULATION HERE
    ---------------------------------
    Envelope demodulation assumes a narrowband signal whose carrier sits far
    above the modulation.  That holds for vibration (1-10 kHz resonance carrying
    a <200 Hz modulation) and fails badly for current: the electrical
    fundamental is 60-100 Hz while BPFI reaches 124 Hz, so the fault sidebands
    are the same order of magnitude as the carrier and even fold through DC.
    Taking |analytic| in that regime returns the carrier's own rectification
    products, which is exactly what the planted-fault test caught -- the BPFO
    peak landed on the fundamental instead of on BPFO.

    WHAT THIS DOES INSTEAD
    ----------------------
    A load impulse at f_fault modulates machine torque and therefore stator
    current, appearing as sidebands at f_elec +/- f_fault.  So for each order o
    we read the spectrum at both sideband positions directly:

        S(o) = |X(f_elec + o*fr)| + |X(|f_elec - o*fr|)|

    The absolute value on the lower sideband is physics, not a guard: a real
    signal's spectrum is symmetric, so a lower sideband pushed below DC reflects
    back up.  Without it the fold would go discontinuous at o = POLE_PAIRS,
    since f_elec / fr = POLE_PAIRS exactly for this PMSM.

    SUPPLY HARMONICS MUST BE NOTCHED FIRST
    --------------------------------------
    Because f_elec = POLE_PAIRS * fr identically, the supply harmonics land on a
    FIXED order grid no matter the running speed:

        upper sideband hits k*f_elec at order  (k-1) * POLE_PAIRS
        lower sideband hits k*f_elec at order  (k+1) * POLE_PAIRS

    so with POLE_PAIRS = 4 the fundamental reappears at order 8, the 2nd
    harmonic at orders 4 and 12, and so on.  These are enormous relative to any
    sideband and would dominate the input -- the planted-fault test caught
    exactly this, with the BPFO peak landing at order 8.0.  Zeroing a few bins
    around each k*f_elec before folding removes them; the notch is narrow enough
    (a few bins, sized to the Hann main lobe) that no genuine fault line is lost,
    and BPFI at 4.95 orders stays 30 bins clear of the order-4 notch.

    The result is divided by the fundamental, making it the sideband-to-
    fundamental ratio -- the standard MCSA severity indicator, and dimensionless,
    so it does not care about current-transducer gain.

    On an FPGA this is one rFFT plus an indexed gather with linear interpolation:
    cheaper than the Hilbert path it replaces.
    """
    n_bins = n_bins or C.N_ORDER_BINS
    order_max = order_max or C.ORDER_MAX

    x = np.asarray(current, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    mag = np.abs(rfft(x * np.hanning(n))) * (2.0 / n)
    freqs = rfftfreq(n, 1.0 / fs)

    if fr_hz <= 0 or f_elec <= 0:
        return np.zeros(n_bins, dtype=np.float64)

    # Fundamental amplitude is read BEFORE notching -- it is the denominator.
    fundamental = float(np.interp(f_elec, freqs, mag))

    notched = mag.copy()
    for k in range(1, C.SUPPLY_HARMONICS_NOTCHED + 1):
        centre = k * f_elec
        if centre > freqs[-1]:
            break
        notched[np.abs(freqs - centre) <= C.SUPPLY_NOTCH_HALFWIDTH_HZ] = 0.0

    orders = np.arange(n_bins, dtype=np.float64) * (order_max / n_bins)
    offsets = orders * fr_hz

    upper = np.interp(f_elec + offsets, freqs, notched, left=0.0, right=0.0)
    lower = np.interp(np.abs(f_elec - offsets), freqs, notched, left=0.0, right=0.0)
    upper[f_elec + offsets > freqs[-1]] = 0.0

    return (upper + lower) / (fundamental + C.SPECTRUM_EPS)


def raw_order_spectrum(signal: np.ndarray, fs: float, fr_hz: float,
                      f_elec: float = None, n_bins: int = None,
                      order_max: float = None) -> np.ndarray:
    """
    Magnitude spectrum read directly on the shaft-order axis, no demodulation.

    Complements the sideband fold: that measures energy at f_elec +/- k*f_r,
    which is what load-torque modulation produces.  This measures energy at
    k*f_r itself -- shaft-rate harmonics from eccentricity, misalignment or a
    rubbing rotor, which appear in current without going through the carrier.
    A fold about f_elec is blind to them.

    Supply harmonics are notched when f_elec is given, for the same reason as
    in the fold: they are enormous and sit on a fixed order grid.
    """
    n_bins = n_bins or C.N_ORDER_BINS
    order_max = order_max or C.ORDER_MAX

    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    mag = np.abs(rfft(x * np.hanning(n))) * (2.0 / n)
    freqs = rfftfreq(n, 1.0 / fs)
    if fr_hz <= 0:
        return np.zeros(n_bins, dtype=np.float64)

    if f_elec and f_elec > 0:
        mag = mag.copy()
        for k in range(1, C.SUPPLY_HARMONICS_NOTCHED + 1):
            centre = k * f_elec
            if centre > freqs[-1]:
                break
            mag[np.abs(freqs - centre) <= C.SUPPLY_NOTCH_HALFWIDTH_HZ] = 0.0

    target = np.arange(n_bins, dtype=np.float64) * (order_max / n_bins) * fr_hz
    out = np.interp(target, freqs, mag, left=0.0, right=0.0)
    out[target > freqs[-1]] = 0.0
    return out


def raw_log_spectrum(signal: np.ndarray, fs: float,
                     band: Tuple[float, float] = None,
                     n_bins: int = None) -> np.ndarray:
    """
    Magnitude spectrum on a LOG-FREQUENCY axis over a fixed absolute band.

    Absolute frequency, not orders, and that is deliberate.  Structural
    resonances of the bearing housing do not move with shaft speed -- only how
    hard they are driven changes.  Putting them on an order axis would smear a
    fixed resonance across different bins at 900 and 1500 rpm, which is exactly
    backwards.

    Log spacing because resonance structure spans two decades (200 Hz to 16 kHz)
    and a linear axis would spend most of its bins on the top octave.

    Energy is integrated between adjacent log-spaced edges rather than sampled,
    so nothing falls between bins at the sparse low end.
    """
    band = band or C.RAW_SPECTRUM_HZ
    n_bins = n_bins or C.N_ORDER_BINS

    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    power = np.abs(rfft(x * np.hanning(n))) ** 2
    freqs = rfftfreq(n, 1.0 / fs)

    hi = min(band[1], 0.98 * fs / 2)
    if hi <= band[0]:
        return np.zeros(n_bins, dtype=np.float64)
    edges = np.geomspace(band[0], hi, n_bins + 1)
    idx = np.searchsorted(freqs, edges)
    cumulative = np.concatenate([[0.0], np.cumsum(power)])
    out = cumulative[np.clip(idx[1:], 0, len(power))] -           cumulative[np.clip(idx[:-1], 0, len(power))]
    return np.sqrt(np.maximum(out, 0.0))


def condition_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """
    Median-normalise then log1p.  See config for why this is safe and why it is
    NOT the v4 norm/rank mistake.

    The median is a robust noise-floor estimate: fault lines are sparse, so they
    barely move it, and dividing by it turns absolute amplitude into 'how far
    above this recording's own noise floor'.  That is invariant to sensor gain
    and to bearing-to-bearing amplitude offsets.
    """
    s = np.asarray(spectrum, dtype=np.float64)
    med = np.median(s)
    if not np.isfinite(med) or med <= C.SPECTRUM_EPS:
        med = float(np.mean(np.abs(s))) or 1.0
    return np.log1p(s / (med + C.SPECTRUM_EPS))


# ===========================================================================
# 5. FULL PER-WINDOW PIPELINE
# ===========================================================================

def window_bounds(n_samples: int, fs: float,
                  window_s: float = None, hop_s: float = None):
    """Start/stop sample indices of each analysis window."""
    window_s = window_s or C.WINDOW_SECONDS
    hop_s = hop_s or C.HOP_SECONDS
    win = int(round(window_s * fs))
    hop = int(round(hop_s * fs))
    if n_samples < win:
        return []
    return [(s, s + win) for s in range(0, n_samples - win + 1, hop)]


def process_window(cur_1_win: np.ndarray, cur_2_win: np.ndarray,
                   vib_envs: Dict[str, np.ndarray], vib_raw_win: np.ndarray,
                   fs_cur: float, fs_vib: float) -> Tuple[np.ndarray, Dict]:
    """
    One analysis window -> (N_INPUT_CHANNELS, N_ORDER_BINS) float32 + diagnostics.

    Current channels average the two phases.  Averaging is not laziness: both
    carry the same amplitude modulation 120 degrees apart, so averaging raises
    the fault lines relative to per-phase noise.  At deployment the third phase
    comes free from i3 = -(i1 + i2) and folds into the same mean.

    Speed is estimated per window, from current only.  It is cheap (one 4096-
    point rFFT) and it is what a real installation has to do anyway.

    See config.FEATURE_SET for what each channel is and why.
    """
    fr_hz, f_elec = estimate_shaft_speed(cur_1_win, fs_cur)
    cur_pair = (cur_1_win, cur_2_win)

    sideband = np.mean([current_sideband_order_spectrum(s, fs_cur, fr_hz, f_elec)
                        for s in cur_pair], axis=0)

    if C.FEATURE_SET == "v1":
        channels = [sideband, envelope_order_spectrum(vib_envs["full"], fs_vib, fr_hz)]
    else:
        channels = [
            sideband,
            np.mean([raw_order_spectrum(s, fs_cur, fr_hz, f_elec)
                     for s in cur_pair], axis=0),
            envelope_order_spectrum(vib_envs["low"], fs_vib, fr_hz),
            envelope_order_spectrum(vib_envs["high"], fs_vib, fr_hz),
            raw_log_spectrum(vib_raw_win, fs_vib),
        ]

    stacked = np.stack([condition_spectrum(c) for c in channels]).astype(np.float32)
    return stacked, {"fr_hz": fr_hz, "f_elec_hz": f_elec, "rpm": fr_hz * 60.0}


def process_recording(current_1: np.ndarray, current_2: np.ndarray,
                      vibration: np.ndarray, fs: float = None):
    """
    A whole 4 s recording -> (n_windows, 2, N_ORDER_BINS) float32 + per-window
    diagnostics.

    Decimation and vibration demodulation run ONCE over the full record, then
    the result is windowed.  That ordering is chosen deliberately: it is both
    faster and closer to deployment, where the FPGA filters a continuous stream
    and windows fall out of it.  Filtering each window independently would
    introduce edge transients that the real system will never see.
    """
    fs = fs or C.FS_FAST
    fs_cur = fs / C.DECIMATE_CURRENT
    fs_vib = fs / C.DECIMATE_VIBRATION

    cur_1 = decimate_signal(current_1, C.DECIMATE_CURRENT)
    cur_2 = decimate_signal(current_2, C.DECIMATE_CURRENT)
    vib = decimate_signal(vibration, C.DECIMATE_VIBRATION)

    # Demodulate once over the whole record, per band.
    if C.FEATURE_SET == "v1":
        vib_envs = {"full": analytic_envelope(vib, fs_vib, C.VIBRATION_BAND_HZ)}
    else:
        vib_envs = {
            "low": analytic_envelope(vib, fs_vib, C.VIBRATION_BAND_LOW_HZ),
            "high": analytic_envelope(vib, fs_vib, C.VIBRATION_BAND_HIGH_HZ),
        }

    n_cur = int(round(C.WINDOW_SECONDS * fs_cur))
    hop_cur = int(round(C.HOP_SECONDS * fs_cur))
    n_vib = int(round(C.WINDOW_SECONDS * fs_vib))
    hop_vib = int(round(C.HOP_SECONDS * fs_vib))

    vib_len = min(v.size for v in vib_envs.values())
    if min(cur_1.size, cur_2.size) < n_cur or vib_len < n_vib:
        return np.zeros((0, C.N_INPUT_CHANNELS, C.N_ORDER_BINS), np.float32), []

    n_windows = min(
        1 + (min(cur_1.size, cur_2.size) - n_cur) // hop_cur,
        1 + (vib_len - n_vib) // hop_vib,
    )

    specs, diags = [], []
    for w in range(n_windows):
        cs, vs = w * hop_cur, w * hop_vib
        X, d = process_window(
            cur_1[cs:cs + n_cur], cur_2[cs:cs + n_cur],
            {k: v[vs:vs + n_vib] for k, v in vib_envs.items()},
            vib[vs:vs + n_vib], fs_cur, fs_vib,
        )
        specs.append(X)
        diags.append(d)
    return np.stack(specs).astype(np.float32), diags
