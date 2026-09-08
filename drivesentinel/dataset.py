"""
DriveSentinel v5 -- dataset.py
==============================
Indexes the Paderborn recordings on disk and loads their channels.

This module is the only place that touches .mat files, and the only place that
knows about the dataset's defects.  It applies them by construction rather than
by convention, so a defect cannot be forgotten by a caller:

  * N09_M07_F10_KA04_17.mat is dropped -- it is a byte-identical copy of run 18
    (the KA04 measuring log documents the substitution; verified by array
    comparison on all seven channels).  Left in, it would place the same window
    twice inside one bearing.
  * N15_M01_F10_KA08_2.mat is dropped -- structurally corrupt.  Correct size and
    header, but scipy cannot parse the variable stream.  The v4 cache holds only
    79 KA08 recordings, so the previous pipeline hit this too and skipped it
    without saying so.
  * KB23/KB24/KB27 are excluded by default -- three physical bearings cannot
    support a class under leave-one-bearing-out.
  * Record lengths vary (256,000-289,013 samples against a nominal 256,001), so
    nothing here assumes a fixed length.

The index is built from what is ACTUALLY on disk and reports its own count.
This copy holds 2,548 files; the copy the v4 cache came from held 2,559, so v4
numbers are not reproducible here.  Recording that discrepancy loudly is the
point -- silently producing a different cache is how two machines drift apart.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.io import loadmat

from . import config as C

_FNAME_RE = re.compile(
    r"^(?P<cond>N\d{2}_M\d{2}_F\d{2})_(?P<bearing>[A-Z]+\d+)_(?P<run>\d+)\.mat$"
)


@dataclass(frozen=True)
class Recording:
    path: str
    filename: str
    bearing: str
    label: str
    condition: str
    run: int
    origin: str          # none | artificial | real


def build_index(data_root: str = None,
                include_excluded: bool = False) -> List[Recording]:
    """
    Every usable recording on disk, sorted deterministically.

    include_excluded=True keeps the KB bearings, for the separate combined-class
    reporting the methodology promises.  The duplicate is dropped either way.
    """
    root = data_root or C.DATA_ROOT
    out: List[Recording] = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*.mat"))):
        fname = os.path.basename(path)
        m = _FNAME_RE.match(fname)
        if m is None:
            continue
        if fname in C.UNUSABLE_RECORDINGS:
            continue
        bearing = m.group("bearing")
        label = C.BEARING_LABELS.get(bearing)
        if label is None:
            continue
        if not include_excluded and bearing in C.EXCLUDED_BEARINGS:
            continue
        out.append(Recording(
            path=path, filename=fname, bearing=bearing, label=label,
            condition=m.group("cond"), run=int(m.group("run")),
            origin=C.damage_origin(bearing),
        ))
    return out


def load_channels(path: str) -> Dict[str, np.ndarray]:
    """
    All seven channels of one recording as float64 1-D arrays.

    struct_as_record=False + squeeze_me=True gives the nested mat_struct layout
    the KAt files use: one top-level variable named after the file, holding a
    Y array of per-channel structs.
    """
    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    keys = [k for k in mat if not k.startswith("__")]
    if not keys:
        raise ValueError(f"{path}: no data variable")
    struct = mat[keys[0]]
    return {
        y.Name: np.asarray(y.Data, dtype=np.float64).ravel()
        for y in struct.Y
    }


def load_analysis_channels(path: str):
    """The three channels the model is allowed to see, plus the tachometer.

    The `speed` channel is returned for validate_speed_estimate() only.  It is
    never passed to the feature builder -- a lift has no torque shaft, and the
    v5 ablation showed the rig-only mechanical channels are worth 0.53 accuracy
    on their own, which is signal the product will not have.
    """
    ch = load_channels(path)
    missing = [c for c in (*C.CH_CURRENT, C.CH_VIBRATION) if c not in ch]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: missing channels {missing}")
    return (
        ch[C.CH_CURRENT[0]],
        ch[C.CH_CURRENT[1]],
        ch[C.CH_VIBRATION],
        ch.get(C.CH_SPEED),
    )


def index_report(records: List[Recording]) -> str:
    from collections import Counter
    by_label = Counter(r.label for r in records)
    by_origin = Counter(r.origin for r in records)
    bearings = sorted({r.bearing for r in records})
    lines = [
        f"recordings   : {len(records):,}  (expected {C.EXPECTED_RECORDING_COUNT:,})",
        f"bearings     : {len(bearings)}",
        f"by label     : {dict(by_label)}",
        f"by origin    : {dict(by_origin)}",
    ]
    per_bearing = Counter(r.bearing for r in records)
    short = {b: n for b, n in sorted(per_bearing.items()) if n != 80}
    if short:
        lines.append(f"NOT 80 runs  : {short}  (KA04 79 = duplicate dropped; "
                     f"KI14 68 = absent from this copy)")

    # Compare like with like: the config constant counts every usable recording,
    # so add back whatever the caller excluded before checking.
    excluded_here = sum(
        1 for b in C.EXCLUDED_BEARINGS if b not in {r.bearing for r in records}
    )
    if excluded_here:
        lines.append(f"excluded     : {C.EXCLUDED_BEARINGS} (combined class, "
                     f"3 bearings -- reported separately, never pooled)")
    return "\n".join("  " + ln for ln in lines)


def check_expected_count(data_root: str = None) -> tuple:
    """(found, expected, ok) over ALL usable recordings, exclusions included.

    Separate from index_report so the number being checked is unambiguous.
    """
    everything = build_index(data_root, include_excluded=True)
    found = len(everything)
    return found, C.EXPECTED_RECORDING_COUNT, found == C.EXPECTED_RECORDING_COUNT


def verify_duplicate(data_root: str = None) -> Optional[bool]:
    """
    Re-check that the excluded KA04 run really is a duplicate of run 18.

    Cheap insurance against blindly trusting a hard-coded exclusion: if a future
    copy of the dataset has a genuine run 17, this returns False and the
    self-test fails rather than silently discarding real data.
    """
    root = data_root or C.DATA_ROOT
    a = os.path.join(root, "KA04", "N09_M07_F10_KA04_17.mat")
    b = os.path.join(root, "KA04", "N09_M07_F10_KA04_18.mat")
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    ca, cb = load_channels(a), load_channels(b)
    return all(
        k in cb and ca[k].shape == cb[k].shape and np.array_equal(ca[k], cb[k])
        for k in ca
    )
