"""
Trip gating: instantaneous electrical frequency, and accel/cruise/decel segmentation.

WHAT THIS IS FOR
----------------
Order-normalised branches (S4 winding, S5 bearing) assume the shaft speed is
steady across a window.  During an accel or decel ramp f_e moves, the order axis
smears, and a fault line spreads across bins.  Gating those branches to cruise
windows is the fix.

>>> AND ON THE DATA WE HAVE, IT GATES NOTHING. <<<

Measured f_e across every real dataset in the roster:

    D1 Paderborn   900 / 1500 rpm, constant within each 4 s recording
    D2 KAIST       200.00 Hz in all 48 recordings, to 0.25 Hz
    D3 Bacha       10 rad/s constant; 10 Hz telemetry, Nyquist 5 Hz
    D4 Thomas      mains-fed, 49.955-50.036 Hz across all ten files

Not one ramps.  Run the segmenter on any of them and every window comes back
`cruise`.  So this module is implemented, unit-tested against synthetic ramps with
known ground truth, and **disabled by default** (`config.TRIP_CONFIG["enabled"]`).

That is a deliberate choice rather than an oversight.  Switching it on would add a
code path that no real data exercises and a claim -- "our branches run only on
cruise windows" -- that nothing in the roster tests.  It becomes live when there is
a speed profile to gate, which means the simulator branch (P7/P8, outside the MVP).

The one real-data thing it CAN do is segment the D4 motor start-up transients:
those files contain a genuine off -> inrush -> running sequence.  But D4 is
mains-fed, so f_e is pinned at 50 Hz throughout and the transient is in the
CURRENT ENVELOPE, not in frequency.  `segment_envelope()` exists for that, and the
docs report segmentation only -- never frequency tracking -- on D4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import hilbert, stft

from . import config as C

PHASES = ("accel", "cruise", "decel")


@dataclass(frozen=True)
class Segment:
    phase: str          # "accel" | "cruise" | "decel"
    t0: float
    t1: float
    f_mean: float
    df_dt: float

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


# ===========================================================================
# instantaneous frequency
# ===========================================================================

def _median_filter(x: np.ndarray, taps: int) -> np.ndarray:
    if taps <= 1 or x.size < taps:
        return x
    k = taps // 2
    pad = np.pad(x, (k, k), mode="edge")
    return np.array([np.median(pad[i:i + taps]) for i in range(x.size)])


def stft_ridge(x: np.ndarray, fs: float, cfg: Dict = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track f_e(t) as the per-frame spectral peak inside the search band.

    Returns (times, f_e).  A plain ridge and not something cleverer, because the
    signal of interest here is a strong single fundamental: the peak IS the
    fundamental, and a fancier tracker would add failure modes without adding
    accuracy.  A short median filter removes single-frame bin flicker.

    Resolution is the STFT bin width (fs/nperseg), which at the default 0.25 s
    window is 4 Hz.  That is coarse against a 50 Hz fundamental, which is exactly
    why `hilbert_refine` exists.
    """
    cfg = cfg or C.TRIP_CONFIG
    n = max(16, int(round(cfg["stft_window_s"] * fs)))
    hop = max(1, int(round(cfg["stft_hop_s"] * fs)))
    f, t, Z = stft(np.asarray(x, dtype=np.float64), fs=fs, nperseg=n,
                   noverlap=n - hop, padded=False, boundary=None)
    lo, hi = cfg["f_search"]
    band = (f >= lo) & (f <= hi)
    if not band.any():
        raise ValueError(f"search band {cfg['f_search']} empty at fs={fs}")
    mag = np.abs(Z[band])
    fb = f[band]
    idx = mag.argmax(axis=0)

    # Parabolic interpolation on the log-magnitude peak. The ridge is otherwise
    # quantised to the bin grid, which turns a smooth ramp into a staircase and
    # makes df/dt a train of spikes.
    fe = np.empty(idx.size)
    for j, i in enumerate(idx):
        if 0 < i < mag.shape[0] - 1:
            a, b, c = (math.log(max(mag[i + d, j], 1e-30)) for d in (-1, 0, 1))
            denom = a - 2 * b + c
            delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
            fe[j] = fb[i] + delta * (fb[1] - fb[0])
        else:
            fe[j] = fb[i]
    return t, _median_filter(fe, cfg["ridge_smooth"])


def hilbert_frequency(x: np.ndarray, fs: float, f_center: float,
                      bandwidth: float = 20.0) -> float:
    """
    Mean instantaneous frequency from the analytic-signal phase slope.

    Band-passes about `f_center`, takes the unwrapped Hilbert phase and fits a
    straight line.  Over a whole window this is far more precise than an STFT bin
    (it uses every sample's phase rather than a spectral peak), which is what
    makes it useful for REFINING a coarse ridge estimate.

    It cannot replace the ridge: it needs to be told roughly where to look, and
    it assumes a single dominant component in the band.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    X[(fr < f_center - bandwidth) | (fr > f_center + bandwidth)] = 0.0
    phase = np.unwrap(np.angle(hilbert(np.fft.irfft(X, n))))
    t = np.arange(n) / fs
    slope = np.polyfit(t, phase, 1)[0]
    return float(slope / (2 * math.pi))


def estimate_f_elec(x: np.ndarray, fs: float, cfg: Dict = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    f_e(t): STFT ridge, optionally refined per frame with the Hilbert phase.

    The refinement is applied only where the ridge is already locally flat -- a
    phase-slope fit across a ramp measures the average, not the instantaneous,
    frequency, so refining a ramping frame would make it worse, not better.
    """
    cfg = cfg or C.TRIP_CONFIG
    t, fe = stft_ridge(x, fs, cfg)
    if not cfg.get("hilbert_refine", True) or fe.size < 3:
        return t, fe

    win = max(16, int(round(cfg["stft_window_s"] * fs)))
    slope = np.abs(np.gradient(fe, t)) if t.size > 1 else np.zeros_like(fe)
    out = fe.copy()
    for j in range(fe.size):
        if slope[j] > cfg["accel_threshold_hz_s"]:
            continue                                   # ramping: leave the ridge
        i0 = int(round(t[j] * fs)) - win // 2
        i0 = max(0, min(i0, x.size - win))
        if x.size < win:
            continue
        try:
            out[j] = hilbert_frequency(x[i0:i0 + win], fs, fe[j])
        except (ValueError, np.linalg.LinAlgError):
            pass
    return t, out


# ===========================================================================
# segmentation
# ===========================================================================

def _runs(flags: Sequence[str]) -> List[Tuple[str, int, int]]:
    out, prev, start = [], None, 0
    for i, f in enumerate(flags):
        if f != prev:
            if prev is not None:
                out.append((prev, start, i))
            prev, start = f, i
    if prev is not None:
        out.append((prev, start, len(flags)))
    return out


def segment(t: np.ndarray, fe: np.ndarray, cfg: Dict = None) -> List[Segment]:
    """
    accel / cruise / decel from the sign and magnitude of df_e/dt.

    Runs shorter than `min_segment_s` are absorbed into the neighbour they
    resemble most, so a single noisy frame cannot split a cruise in two.
    """
    cfg = cfg or C.TRIP_CONFIG
    t = np.asarray(t, dtype=np.float64)
    fe = np.asarray(fe, dtype=np.float64)
    if t.size < 2:
        return []

    d = np.gradient(fe, t)
    thr = cfg["accel_threshold_hz_s"]
    flags = ["accel" if v > thr else "decel" if v < -thr else "cruise" for v in d]

    # Absorb runs that are too short to be a real phase.
    min_s = cfg["min_segment_s"]
    changed = True
    while changed and len(flags) > 1:
        changed = False
        for ph, a, b in _runs(flags):
            if t[b - 1] - t[a] >= min_s or (b - a) == len(flags):
                continue
            left = flags[a - 1] if a > 0 else None
            right = flags[b] if b < len(flags) else None
            fill = left or right
            if fill is None:
                continue
            for i in range(a, b):
                flags[i] = fill
            changed = True
            break

    return [Segment(phase=ph, t0=float(t[a]), t1=float(t[b - 1]),
                    f_mean=float(np.mean(fe[a:b])),
                    df_dt=float(np.mean(d[a:b])))
            for ph, a, b in _runs(flags)]


def segment_envelope(x: np.ndarray, fs: float, smooth_s: float = 0.05,
                     on_frac: float = 0.25) -> List[Segment]:
    """
    Segment on the CURRENT ENVELOPE rather than on frequency.

    For mains-fed machines (D4) f_e is pinned by the grid, so a start-up
    transient is invisible to a frequency tracker but obvious in the envelope.
    Returns `accel` while the envelope is rising through the threshold band,
    `cruise` once settled, and `decel` while falling.

    `f_mean` is left at NaN on purpose: this function knows nothing about
    frequency and must not be read as if it did.
    """
    x = np.asarray(x, dtype=np.float64)
    n = max(1, int(round(smooth_s * fs)))
    env = np.abs(hilbert(x - x.mean()))
    k = np.ones(n) / n
    env = np.convolve(env, k, mode="same")
    t = np.arange(env.size) / fs

    hi = float(np.percentile(env, 95))
    on = on_frac * hi
    running = env > on
    d = np.gradient(env, t)
    dthr = 0.5 * hi / max(smooth_s * 4, 1e-6)

    flags = []
    for i in range(env.size):
        if not running[i]:
            flags.append("decel" if i and running[i - 1] else "accel")
        elif d[i] > dthr:
            flags.append("accel")
        elif d[i] < -dthr:
            flags.append("decel")
        else:
            flags.append("cruise")

    segs = []
    for ph, a, b in _runs(flags):
        if t[b - 1] - t[a] < 0.20:
            continue
        segs.append(Segment(phase=ph, t0=float(t[a]), t1=float(t[b - 1]),
                            f_mean=float("nan"),
                            df_dt=float(np.mean(d[a:b]))))
    return segs


def cruise_mask(segments: Sequence[Segment], t: np.ndarray) -> np.ndarray:
    """Boolean mask over `t`, True inside a cruise segment."""
    t = np.asarray(t, dtype=np.float64)
    m = np.zeros(t.shape, dtype=bool)
    for s in segments:
        if s.phase == "cruise":
            m |= (t >= s.t0) & (t <= s.t1)
    return m


def gate(segments: Sequence[Segment], t: np.ndarray,
         enabled: Optional[bool] = None) -> np.ndarray:
    """
    Which windows an order-normalised branch may use.

    With gating disabled -- the default, see the module docstring -- this returns
    all-True rather than raising, so a caller does not need to branch on the
    config.  Disabled gating means "use everything", not "use nothing".
    """
    enabled = C.TRIP_CONFIG["enabled"] if enabled is None else enabled
    if not enabled:
        return np.ones(np.asarray(t).shape, dtype=bool)
    return cruise_mask(segments, t)


# ===========================================================================
# order resolution vs shaft speed -- the analytic artefact
# ===========================================================================

def order_resolution(shaft_rpm: float, window_s: float = None,
                     order_max: float = None, n_bins: int = None) -> Dict:
    """
    What one order actually costs in frequency at a given shaft speed.

    A window of length T resolves 1/T Hz.  One shaft order is f_shaft =
    rpm/60 Hz wide.  So the order resolution is

        delta_order = (1/T) / f_shaft = 60 / (T * rpm)

    Slower shaft, coarser order axis, for a fixed window.  This is the honest
    answer to "will this work on a gearless elevator sheave", and it is
    arithmetic rather than a claim about performance.
    """
    window_s = window_s or C.WINDOW_SECONDS
    order_max = order_max or C.ORDER_MAX
    n_bins = n_bins or C.N_ORDER_BINS

    f_shaft = shaft_rpm / 60.0
    df = 1.0 / window_s
    d_order = df / f_shaft if f_shaft > 0 else float("inf")
    bin_width = order_max / n_bins
    return {
        "shaft_rpm": shaft_rpm,
        "f_shaft_hz": f_shaft,
        "freq_resolution_hz": df,
        "order_resolution": d_order,
        "bin_width_orders": bin_width,
        # How many analysis bins one genuinely resolved order spans. Below 1 the
        # axis is finer than the physics supports and neighbouring bins are
        # correlated rather than independent.
        "resolved_bins_per_order": d_order / bin_width,
        "window_s": window_s,
        "shaft_revs_in_window": f_shaft * window_s,
    }


def resolution_table(rpms: Sequence[float] = (900.0, 1500.0, 300.0, 100.0,
                                              50.0, 20.0, 10.0),
                     window_s: float = None) -> List[Dict]:
    return [order_resolution(r, window_s) for r in rpms]


def resolution_report(window_s: float = None) -> str:
    """
    Rendered table, for the docs and the one slide that answers this question.

    EXTRAPOLATION, CLEARLY LABELLED. Every row below 900 rpm is arithmetic about
    a machine we have no data from. It says what the analysis window can resolve,
    which is a statement about signal processing, not about whether a fault would
    be detectable on a real sheave.
    """
    window_s = window_s or C.WINDOW_SECONDS
    rows = resolution_table(window_s=window_s)
    L = [f"Order resolution at a {window_s:g} s window "
         f"({C.N_ORDER_BINS} bins over 0-{C.ORDER_MAX:g} orders, "
         f"bin width {C.ORDER_MAX / C.N_ORDER_BINS:.4f} orders)",
         "",
         f"{'shaft rpm':>10} {'f_shaft Hz':>11} {'revs/window':>12} "
         f"{'order res':>10} {'bins/order':>11}  note",
         "-" * 74]
    for r in rows:
        if r["shaft_rpm"] == 900.0:
            note = "Paderborn (measured)"
        elif r["shaft_rpm"] == 1500.0:
            note = "Paderborn (measured)"
        elif r["shaft_rpm"] >= 100.0:
            note = "EXTRAPOLATION"
        else:
            note = "EXTRAPOLATION -- gearless sheave range"
        L.append(f"{r['shaft_rpm']:10.0f} {r['f_shaft_hz']:11.2f} "
                 f"{r['shaft_revs_in_window']:12.2f} {r['order_resolution']:10.4f} "
                 f"{r['resolved_bins_per_order']:11.2f}  {note}")
    L += ["",
          "Reading it: at 900 rpm a 1 s window sees 15 shaft revolutions and",
          "resolves 0.067 orders, comfortably finer than the 0.031-order bin",
          "grid. At 20 rpm it sees 0.33 of a revolution and resolves 3 orders --",
          "coarser than the entire spacing between BPFO (3.05) and BPFI (4.95),",
          "so those two lines are no longer separable at that window length.",
          "",
          "The fix is a longer window, not a cleverer algorithm: resolution is",
          "1/T. Holding 0.067 orders at 20 rpm needs a 45 s window, which is",
          "longer than many elevator trips. That trade -- window length against",
          "shaft speed -- is the real constraint on porting this to a sheave,",
          "and it is arithmetic, not a performance claim."]
    return "\n".join(L)
