"""
DriveSentinel v5 -- features.py
==============================
Builds the order-spectrum cache: 21 GB of .mat files in, one ~150 MB .npz out.

This is the only expensive step in the project and it runs once.  Everything
after it (32-fold training, quantisation, export) reads the cache and never
touches data/ again.

WHAT LANDS IN THE CACHE
-----------------------
    X     float32 (N, 2, 512)   conditioned order spectra -- the model input
    y     int8    (N,)          class index into config.LABELS
    meta  parquet (N rows)      bearing, condition, run, origin, window index,
                                estimated rpm and the speed-estimate error

The speed-estimate error is stored per window rather than asserted away: it is
the diagnostic that would reveal a wrong pole-pair count or a bad recording, and
build_cache() reports its distribution at the end.  Nothing downstream uses it
as a feature -- it depends on the tachometer, which deployment does not have.

DETERMINISM
-----------
Recordings are processed in sorted order and written in that order, so two runs
on the same copy of the data produce byte-identical caches.  The cache header
stores the recording count and a config fingerprint; loaders check it, so a
cache built under a different DSP contract fails loudly instead of silently
training on stale features.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from . import dataset as ds
from . import dsp


def contract_fingerprint() -> str:
    """
    Identifies the DSP contract that produced a cache.

    Any change to these values changes the meaning of X, so a cache built before
    the change must not be reused after it.
    """
    # Only the settings the ACTIVE feature set actually consumes go in.  Adding
    # a parameter that a code path does not use must not invalidate that path's
    # cache -- that is how a 2-minute rebuild turns into a confusing failure on
    # a change that could not have affected the data.
    payload = {
        "window_s": C.WINDOW_SECONDS, "hop_s": C.HOP_SECONDS,
        "n_bins": C.N_ORDER_BINS, "order_max": C.ORDER_MAX,
        "channels": C.N_INPUT_CHANNELS, "pole_pairs": C.POLE_PAIRS,
        "dec_cur": C.DECIMATE_CURRENT, "dec_vib": C.DECIMATE_VIBRATION,
        "feature_set": C.FEATURE_SET,
        "notch_k": C.SUPPLY_HARMONICS_NOTCHED,
        "notch_hw": C.SUPPLY_NOTCH_HALFWIDTH_HZ,
        "geometry": C.BEARING_GEOMETRY,
        "labels": C.LABELS,
    }
    if C.FEATURE_SET == "v1":
        payload["vib_band"] = list(C.VIBRATION_BAND_HZ)
    else:
        payload["vib_band_low"] = list(C.VIBRATION_BAND_LOW_HZ)
        payload["vib_band_high"] = list(C.VIBRATION_BAND_HIGH_HZ)
        payload["raw_band"] = list(C.RAW_SPECTRUM_HZ)
    import hashlib
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _process_one(rec: ds.Recording) -> Tuple:
    """Worker: one recording -> (spectra, rows) or (None, error)."""
    try:
        i1, i2, vib, speed = ds.load_analysis_channels(rec.path)
        specs, diags = dsp.process_recording(i1, i2, vib)
        if specs.shape[0] == 0:
            return None, f"{rec.filename}: too short for a {C.WINDOW_SECONDS}s window"
        if not np.isfinite(specs).all():
            return None, f"{rec.filename}: non-finite spectrum"

        rows = []
        for w, d in enumerate(diags):
            err = (dsp.validate_speed_estimate(d["fr_hz"], speed)
                   if speed is not None else float("nan"))
            rows.append({
                "bearing": rec.bearing, "label": rec.label,
                "condition": rec.condition, "run": rec.run,
                "origin": rec.origin, "filename": rec.filename,
                "window": w, "rpm_est": d["rpm"], "f_elec_hz": d["f_elec_hz"],
                "speed_err": err,
            })
        return specs, rows
    except Exception as exc:                      # noqa: BLE001 -- reported, not raised
        return None, f"{rec.filename}: {type(exc).__name__}: {exc}"


def build_cache(data_root: str = None, out_path: str = None,
                meta_path: str = None, n_jobs: int = -1,
                include_excluded: bool = False, limit: int = None) -> dict:
    C.ensure_dirs()
    out_path = out_path or C.CACHE_PATH
    meta_path = meta_path or C.CACHE_META_PATH

    records = ds.build_index(data_root, include_excluded=include_excluded)
    if limit:
        records = records[:limit]
    print(ds.index_report(records), flush=True)
    print(f"  contract     : {contract_fingerprint()}  ({C.summary()})", flush=True)
    print(f"  processing {len(records):,} recordings on {n_jobs} workers...", flush=True)

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5, batch_size=8)(
        delayed(_process_one)(r) for r in records
    )

    X_parts, meta_rows, failures = [], [], []
    for specs, payload in results:
        if specs is None:
            failures.append(payload)
            continue
        X_parts.append(specs)
        meta_rows.extend(payload)

    if not X_parts:
        raise RuntimeError("cache build produced nothing; see failures above")

    X = np.concatenate(X_parts).astype(np.float32)
    meta = pd.DataFrame(meta_rows)
    y = meta["label"].map(C.LABEL_TO_INDEX).to_numpy(dtype=np.int8)
    assert len(meta) == X.shape[0], "meta/X length mismatch"

    np.savez_compressed(
        out_path, X=X, y=y,
        contract=np.array(contract_fingerprint()),
        n_recordings=np.array(len(records) - len(failures)),
    )
    meta.to_parquet(meta_path, index=False)

    elapsed = time.time() - t0
    err = meta["speed_err"].to_numpy(dtype=float)
    err = err[np.isfinite(err)]
    report = {
        "windows": int(X.shape[0]),
        "recordings_ok": int(len(records) - len(failures)),
        "failures": failures,
        "shape": list(X.shape),
        "mb": round(X.nbytes / 1e6, 1),
        "seconds": round(elapsed, 1),
        "speed_err_median_pct": float(np.median(err) * 100) if err.size else None,
        "speed_err_p99_pct": float(np.percentile(err, 99) * 100) if err.size else None,
        "class_counts": meta["label"].value_counts().to_dict(),
    }

    print(f"\n  windows      : {report['windows']:,}  shape {tuple(X.shape)}"
          f"  ({report['mb']} MB float32)", flush=True)
    print(f"  classes      : {report['class_counts']}", flush=True)
    print(f"  speed error  : median {report['speed_err_median_pct']:.3f} %  "
          f"p99 {report['speed_err_p99_pct']:.3f} %   "
          f"(current-derived vs tachometer)", flush=True)
    print(f"  elapsed      : {elapsed / 60:.1f} min", flush=True)
    if failures:
        print(f"  FAILURES ({len(failures)}):", flush=True)
        for f in failures[:20]:
            print(f"    {f}", flush=True)
    with open(os.path.join(C.ARTIFACT_DIR, "cache_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def load_cache(path: str = None, meta_path: str = None,
               check_contract: bool = True):
    """(X, y, meta).  Refuses a cache built under a different DSP contract."""
    path = path or C.CACHE_PATH
    meta_path = meta_path or C.CACHE_META_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run scripts/01_build_cache.py first"
        )
    blob = np.load(path, allow_pickle=False)
    stored = str(blob["contract"])
    if check_contract and stored != contract_fingerprint():
        raise RuntimeError(
            f"cache contract {stored} != current {contract_fingerprint()}. "
            f"The DSP settings changed since this cache was built; rebuild it."
        )
    return blob["X"], blob["y"], pd.read_parquet(meta_path)
