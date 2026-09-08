"""
DriveSentinel v5 -- folds.py
============================
Leave-one-bearing-out cross-validation, and the grouped inner split that early
stopping is allowed to see.

WHY 29 FOLDS AND NOT 5
----------------------
The v4 study used k=5.  With five folds the standard error on a fold-mean
accuracy is 0.03-0.09, which is larger than almost every difference it was used
to argue about: paired t-tests on the v4 checkpoints put RF vs MLP at p=0.979,
envelope vs raw at p=0.502, and raw+envelope vs raw at p=0.088.  None of the
model or feature comparisons in that report were distinguishable from fold
noise.

One fold per bearing gives 29 estimates instead of 5, roughly 2.4x tighter error
bars, and every bearing gets its own held-out score -- which also makes it
obvious WHICH bearings the model fails on, information a 5-fold split averages
away.  It costs ~6x the compute, which on a 3050 is minutes.

THE INNER SPLIT IS THE ONE PEOPLE GET WRONG
-------------------------------------------
v4 called sklearn's MLPClassifier with early_stopping=True, which carves off a
RANDOM 10 % of the training rows for its internal validation set.  Seven windows
of one 4 s recording are near-identical, so that validation set was full of
near-duplicates of rows being fitted, its score was inflated, and the stopping
criterion was miscalibrated.  The same project had already grouped the inner CV
for LR and SVM; the MLP's internal split was simply missed.

inner_split() below holds out whole BEARINGS, never windows, so the early-
stopping signal is an honest estimate of generalisation to an unseen bearing.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import config as C


def leave_one_bearing_out(bearings: Sequence[str]) -> List[Dict]:
    """One fold per bearing, in sorted order. Deterministic, no RNG involved."""
    return [
        {"fold": i, "test_bearings": [b],
         "test_label": C.BEARING_LABELS.get(b, "?")}
        for i, b in enumerate(sorted(set(bearings)))
    ]


def inner_split(train_bearings: Sequence[str], labels: Dict[str, str],
                seed: int = C.SEED,
                per_class: int = None) -> Tuple[List[str], List[str]]:
    """
    Split training bearings into (fit, validation), holding out whole bearings.

    One bearing per class by default.  Fewer would make the early-stopping
    signal too noisy to act on; more would starve the fit set, which already has
    only ~9 bearings per class.
    """
    per_class = (C.TRAIN_CONFIG["val_bearings_per_class"]
                 if per_class is None else per_class)

    # per_class = 0 means NO inner validation split at all.
    #
    # MEASURED: with one validation bearing per class, three repeats of an
    # identical config scored 0.6041 / 0.7790 / 0.8018 -- a standard deviation
    # of 0.108.  Each of those runs already averaged three models internally, so
    # weight initialisation was not the cause; the base seed also chooses WHICH
    # bearings are held out for validation, and with only 6 healthy specimens,
    # removing a different one changes what the model can learn about the class.
    #
    # Dropping the split removes that variance source AND returns 3 bearings to
    # the fit set.  It is only affordable because the schedule is a completed
    # OneCycle with no early stopping -- there is nothing left for a validation
    # set to decide.
    if per_class <= 0:
        return sorted(train_bearings), []

    rng = np.random.default_rng(seed)

    by_class: Dict[str, List[str]] = {}
    for b in sorted(train_bearings):
        by_class.setdefault(labels[b], []).append(b)

    val: List[str] = []
    for cls in sorted(by_class):
        members = by_class[cls]
        if len(members) <= per_class:
            # Never strip a class down to nothing -- a validation set missing a
            # class produces a meaningless early-stopping score.
            continue
        val.extend(rng.choice(members, size=per_class, replace=False).tolist())

    fit = [b for b in sorted(train_bearings) if b not in set(val)]
    return fit, sorted(val)


def make_folds(meta, path: str = None, overwrite: bool = False) -> List[Dict]:
    """
    Build (or reload) the fold definition and persist it.

    Persisting matters for the same reason it did in v4: every later script --
    training, quantisation calibration, the golden-reference check -- must see
    identical splits, or their numbers are not comparable.
    """
    path = path or C.FOLDS_PATH
    bearings = sorted(meta["bearing"].unique())

    if os.path.exists(path) and not overwrite:
        with open(path) as fh:
            blob = json.load(fh)
        stored = sorted(b for f in blob["folds"] for b in f["test_bearings"])
        if stored != bearings:
            raise RuntimeError(
                f"{path} covers {len(stored)} bearings but the cache has "
                f"{len(bearings)}. Delete the file to regenerate."
            )
        return blob["folds"]

    folds = leave_one_bearing_out(bearings)
    C.ensure_dirs()
    with open(path, "w") as fh:
        json.dump({"scheme": "leave-one-bearing-out", "n_folds": len(folds),
                   "bearings": bearings, "folds": folds}, fh, indent=2)
    return folds


def verify_folds(folds: List[Dict], bearings: Sequence[str]) -> None:
    """Hard assertions. A silent leak here invalidates every number downstream."""
    seen = set()
    for f in folds:
        test = set(f["test_bearings"])
        assert test, f"fold {f['fold']}: empty test set"
        assert not (seen & test), f"fold {f['fold']}: bearing in two folds"
        seen |= test
    missing = set(bearings) - seen
    assert not missing, f"bearings never tested: {sorted(missing)}"
    assert seen == set(bearings), f"unknown bearings in folds: {seen - set(bearings)}"


def split_masks(meta, test_bearings: Sequence[str]):
    """Boolean masks over cache rows. The invariant this project never breaks:
    a physical bearing is entirely in train or entirely in test."""
    bearing = meta["bearing"].to_numpy()
    is_test = np.isin(bearing, list(test_bearings))
    is_train = ~is_test
    overlap = set(bearing[is_train]) & set(bearing[is_test])
    assert not overlap, f"BEARING LEAK: {sorted(overlap)}"
    return is_train, is_test


def majority_baseline(y: np.ndarray) -> float:
    """Accuracy of always predicting the most common class in y.

    Printed beside every result. Two of the ten configurations in the v5
    ablation failed to beat it, which is exactly the sort of thing that goes
    unnoticed when the baseline is not shown.
    """
    if y.size == 0:
        return float("nan")
    return float(np.bincount(y, minlength=len(C.LABELS)).max() / y.size)
