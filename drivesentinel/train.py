"""
DriveSentinel v5 -- train.py
============================
Leave-one-bearing-out training for the order-spectrum CNN.

THREE THINGS THIS FILE IS CAREFUL ABOUT
---------------------------------------
1. The whole cache lives on the GPU as one tensor.  At 16k windows x 2 x 512
   float32 that is 66 MB -- a rounding error against 4 GB -- and it removes the
   DataLoader entirely.  For a 27k-parameter model the dataloader IS the
   bottleneck; indexing a resident tensor turns each epoch into pure kernel
   launches.

2. Early stopping watches a GROUPED validation split (whole bearings held out of
   the fit set), never a random slice of windows.  See folds.inner_split for why
   the random version silently inflates the stopping signal.

3. The learning-rate schedule is allowed to FINISH.  OneCycleLR anneals across
   the whole epoch budget, so a run cut short at epoch 8 of 60 stops near peak
   LR with the weights still moving and never reaches the refinement phase.
   The v5.0 sweep did exactly that -- best_epoch was 5-14 on most folds -- which
   is why early stopping is off by default now; the best-validation snapshot is
   still kept, so nothing is lost by letting it run.

NORMALISATION STATISTICS COME FROM THE FIT SET ONLY
---------------------------------------------------
Per-channel mean and standard deviation (two numbers per channel) are computed
on the fit bearings of each fold and applied to validation and test.  They
become baked-in constants in the deployed accelerator, so they must never see a
test bearing.

REPORTED METRICS
----------------
Window accuracy, macro-F1, per-recording accuracy, and the majority-class
baseline.  All four, always.  Accuracy alone flatters a model on a 2:2:1
problem, and without the baseline it is impossible to tell "learned something"
from "learned to say outer_race".
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from . import folds as F_
from .model import build_model, OrderSpectrumCNN


def channel_stats(x: torch.Tensor):
    """
    PER-CHANNEL mean and std -- two numbers per channel, not 512.

    Per-BIN standardisation would whiten the average spectral shape, and more
    importantly it cannot be implemented cheaply on the FPGA: it would need
    2 x 512 float constants in front of the first convolution.  Per-channel is
    four constants total, and the input is already median-normalised and
    log1p-compressed by the DSP front end, so it is well conditioned before it
    gets here.  export.py asserts on this shape rather than trusting it.
    """
    mean = x.mean(dim=(0, 2), keepdim=True)
    std = x.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    return mean, std


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# augmentation -- mimics bearing-to-bearing variation, on-GPU, in-batch
# ===========================================================================

def augment(x: torch.Tensor, cfg: Dict = None) -> torch.Tensor:
    """
    Perturbations, each defensible as something a different bearing or a
    different installation would actually do to the spectrum:

      order jitter    the 6203 internal geometry is assumed, not documented, so
                      true fault orders sit within about +/- 1 % of nominal.
      amplitude scale sensor gain and mounting differences scale a channel.
      channel dropout forces the model to use BOTH current and vibration rather
                      than leaning entirely on whichever is easier that fold.
      additive noise  rig and electrical noise floor.

    Capacity is not the lever on 29 bearings -- this is.
    """
    cfg = cfg or C.AUGMENT_CONFIG
    b, ch, n = x.shape
    device = x.device

    apply = torch.rand(b, device=device) < cfg["prob"]
    if not apply.any():
        return x
    x = x.clone()

    jit = int(cfg.get("order_jitter_bins", 0))
    if jit > 0:
        shifts = torch.randint(-jit, jit + 1, (b,), device=device) * apply
        idx = (torch.arange(n, device=device)[None, :] - shifts[:, None]) % n
        x = torch.gather(x, 2, idx[:, None, :].expand(-1, ch, -1))

    amp = float(cfg.get("amplitude_scale", 0.0))
    if amp > 0:
        gain = 1.0 + (torch.rand(b, ch, 1, device=device) * 2 - 1) * amp
        x = x * torch.where(apply[:, None, None], gain, torch.ones_like(gain))

    drop_p = float(cfg.get("channel_dropout", 0.0))
    if drop_p > 0 and ch > 1:
        # zero at most one channel per sample, never all of them
        which = torch.randint(0, ch, (b,), device=device)
        hit = ((torch.rand(b, device=device) < drop_p) & apply).float()
        mask = torch.ones(b, ch, 1, device=device)
        mask[torch.arange(b, device=device), which, 0] = 1.0 - hit
        x = x * mask

    noise = float(cfg.get("noise_std", 0.0))
    if noise > 0:
        x = x + torch.randn_like(x) * noise * apply[:, None, None]

    return x


# ===========================================================================
# metrics
# ===========================================================================

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def recording_vote(filenames: np.ndarray, y_true: np.ndarray,
                   y_pred: np.ndarray, n_classes: int,
                   probs: np.ndarray = None):
    """
    Aggregate a recording's windows into one verdict -> per-recording accuracy.

    Given `probs` (per-window softmax) this averages PROBABILITIES -- soft
    voting.  A window the model is unsure about should not weigh as much as one
    it is confident about, and a hard majority throws that distinction away.
    Falls back to a hard majority when probabilities are not supplied.

    This is also what the deployed system does: the accelerator emits one
    softmax per window and the PS averages over a rolling buffer.
    """
    order = np.argsort(filenames, kind="stable")
    fn, yt, yp = filenames[order], y_true[order], y_pred[order]
    pr = probs[order] if probs is not None else None
    _, starts = np.unique(fn, return_index=True)
    groups = np.split(np.arange(len(fn)), starts[1:])

    correct = 0
    for g in groups:
        if pr is not None:
            verdict = int(pr[g].mean(axis=0).argmax())
        else:
            verdict = int(np.bincount(yp[g], minlength=n_classes).argmax())
        correct += int(verdict == yt[g][0])
    return correct / len(groups), len(groups)


def aggregation_curve(filenames: np.ndarray, bearings: np.ndarray,
                      y_true: np.ndarray, probs: np.ndarray,
                      n_observations=(1, 2, 5, 10, 20, 40),
                      n_draws: int = 200, seed: int = C.SEED) -> Dict:
    """
    Accuracy as a function of how many 4-second observations are pooled.

    A window-level verdict is the wrong unit for the product.  A lift makes
    dozens of trips a day, so a health call can accumulate evidence instead of
    betting on one 4-second look at the machine.  This measures what that buys,
    WITHOUT loosening the protocol: every probability here is still an
    out-of-sample prediction from the fold in which its bearing was held out.

    For each N, recordings of one bearing are drawn at random N at a time, their
    per-window softmax averaged, and the argmax scored.  Repeated `n_draws`
    times per bearing so the estimate is not one lucky grouping.
    """
    rng = np.random.default_rng(seed)
    # collapse windows -> recordings first
    order = np.argsort(filenames, kind="stable")
    fn, bt, yt, pr = filenames[order], bearings[order], y_true[order], probs[order]
    _, starts = np.unique(fn, return_index=True)
    groups = np.split(np.arange(len(fn)), starts[1:])

    rec_prob = np.stack([pr[g].mean(axis=0) for g in groups])
    rec_true = np.array([yt[g][0] for g in groups])
    rec_bear = np.array([bt[g][0] for g in groups])

    out = {}
    for n in n_observations:
        hits = total = 0
        for b in np.unique(rec_bear):
            sel = np.flatnonzero(rec_bear == b)
            if sel.size == 0:
                continue
            k = min(n, sel.size)
            for _ in range(n_draws):
                pick = rng.choice(sel, size=k, replace=False)
                verdict = int(rec_prob[pick].mean(axis=0).argmax())
                hits += int(verdict == rec_true[pick[0]])
                total += 1
        out[str(n)] = float(hits / total) if total else float("nan")
    return out


@dataclass
class FoldResult:
    fold: int
    test_bearing: str
    test_label: str
    origin: str
    n_test_windows: int
    n_test_recordings: int
    window_acc: float
    macro_f1: float
    recording_acc: float
    baseline: float
    best_epoch: int
    val_acc: float
    seconds: float


# ===========================================================================
# inference
# ===========================================================================

@torch.no_grad()
def predict_probs(model: OrderSpectrumCNN, X: torch.Tensor,
                  tta_shifts=(0,), batch: int = 8192) -> torch.Tensor:
    """
    Softmax probabilities, averaged over test-time order-axis shifts.

    TTA here is matched to the one real uncertainty in the input rather than
    being a generic trick: the 6203 internal geometry is assumed, not
    documented, so true fault orders sit within about +/- 1 % of nominal and the
    lines can land a bin or two off.  Averaging over small shifts marginalises
    that out instead of hoping the network learned the invariance.

    Costs the FPGA nothing -- the accelerator still evaluates one window at a
    time, and a shift is a re-indexed read of the same input buffer.
    """
    model.eval()
    out = torch.zeros(X.shape[0], len(C.LABELS), device=X.device)
    for shift in tta_shifts:
        xs = torch.roll(X, shifts=int(shift), dims=2) if shift else X
        for s in range(0, xs.shape[0], batch):
            chunk = xs[s:s + batch]
            out[s:s + chunk.shape[0]] += torch.softmax(model(chunk), dim=1)
    return out / len(tta_shifts)


# ===========================================================================
# one model
# ===========================================================================

def _fit_one(Xf, yf, Xv, yv, weights, cfg, device, seed):
    """Train a single model. Returns (model, best_val_score, best_epoch)."""
    torch.manual_seed(seed)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    steps_per_epoch = max(1, (Xf.shape[0] + cfg["batch_size"] - 1) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch)

    best, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(Xf.shape[0], device=device)
        for s in range(0, perm.numel(), cfg["batch_size"]):
            b = perm[s:s + cfg["batch_size"]]
            xb = augment(Xf[b]) if cfg["augment"] else Xf[b]
            loss = F.cross_entropy(model(xb), yf[b], weight=weights,
                                   label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        if Xv.shape[0] == 0:
            # No validation split (val_bearings_per_class = 0). There is nothing
            # to select on, so the final epoch of the completed schedule IS the
            # model. See folds.inner_split for why this is preferred.
            best, best_epoch = float("nan"), epoch
            continue

        model.eval()
        with torch.no_grad():
            vp = model(Xv).argmax(1)
        # Balanced over classes: the validation set is a handful of bearings and
        # can be badly skewed, so plain accuracy would select the wrong snapshot.
        score = float(np.mean([
            float((vp[yv == c] == c).float().mean())
            for c in range(len(C.LABELS)) if (yv == c).any()
        ]))

        if score > best:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if cfg.get("early_stop", False) and stale >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), best, best_epoch


# ===========================================================================
# one fold
# ===========================================================================

def train_one_fold(X: torch.Tensor, y: torch.Tensor, meta, fold: Dict,
                   device: torch.device, cfg: Dict = None,
                   seed: int = C.SEED, verbose: bool = True):
    cfg = cfg or C.TRAIN_CONFIG
    np.random.seed(seed)

    bearings = meta["bearing"].to_numpy()
    filenames = meta["filename"].to_numpy()
    labels_by_bearing = dict(zip(meta["bearing"], meta["label"]))

    test_b = fold["test_bearings"]
    is_train = ~np.isin(bearings, test_b)
    is_test = ~is_train
    assert not (set(bearings[is_train]) & set(bearings[is_test])), "BEARING LEAK"

    fit_b, val_b = F_.inner_split(sorted(set(bearings[is_train])),
                                  labels_by_bearing, seed=seed)
    is_val = np.isin(bearings, val_b)
    is_fit = is_train & ~is_val

    i_fit = torch.from_numpy(np.flatnonzero(is_fit)).to(device)
    i_val = torch.from_numpy(np.flatnonzero(is_val)).to(device)
    i_test = torch.from_numpy(np.flatnonzero(is_test)).to(device)

    # -- normalisation: fit-set statistics only ----------------------------
    mean, std = channel_stats(X[i_fit])
    Xf, yf = (X[i_fit] - mean) / std, y[i_fit]
    Xt, yt = (X[i_test] - mean) / std, y[i_test]
    if i_val.numel():
        Xv, yv = (X[i_val] - mean) / std, y[i_val]
    else:
        Xv, yv = X[:0], y[:0]

    counts = torch.bincount(yf, minlength=len(C.LABELS)).float()
    weights = (counts.sum() / (len(C.LABELS) * counts.clamp_min(1))).to(device)

    t0 = time.time()
    n_seeds = int(cfg.get("n_seeds", 1))
    tta = tuple(cfg.get("tta_shifts", (0,)))

    probs = torch.zeros(Xt.shape[0], len(C.LABELS), device=device)
    val_scores, best_epochs, models = [], [], []
    for k in range(n_seeds):
        model, v, ep = _fit_one(Xf, yf, Xv, yv, weights, cfg, device,
                                seed + 1000 * k)
        probs += predict_probs(model, Xt, tta)
        val_scores.append(v)
        best_epochs.append(ep)
        models.append(model)
    probs /= n_seeds

    pred = probs.argmax(1).cpu().numpy()
    truth = yt.cpu().numpy()
    prob_np = probs.cpu().numpy()

    rec_acc, n_rec = recording_vote(
        filenames[is_test], truth, pred, len(C.LABELS),
        probs=prob_np if C.SOFT_VOTE else None)

    result = FoldResult(
        fold=fold["fold"], test_bearing=test_b[0],
        test_label=C.BEARING_LABELS[test_b[0]],
        origin=C.damage_origin(test_b[0]),
        n_test_windows=int(is_test.sum()), n_test_recordings=n_rec,
        window_acc=float((pred == truth).mean()),
        macro_f1=macro_f1(truth, pred, len(C.LABELS)),
        recording_acc=float(rec_acc),
        baseline=F_.majority_baseline(truth),
        best_epoch=int(np.mean(best_epochs)), val_acc=float(np.mean(val_scores)),
        seconds=round(time.time() - t0, 1),
    )
    if verbose:
        print(f"  fold {result.fold:2d} {result.test_bearing:5} "
              f"({result.test_label:10} {result.origin:10}) "
              f"acc={result.window_acc:.4f} F1={result.macro_f1:.4f} "
              f"rec={result.recording_acc:.4f} val={result.val_acc:.3f} "
              f"ep={result.best_epoch:3d} {result.seconds:5.1f}s", flush=True)

    stats = {"mean": mean.cpu().numpy(), "std": std.cpu().numpy()}
    return result, models[0], stats, (truth, pred, prob_np)


# ===========================================================================
# full sweep
# ===========================================================================

def run_lobo(X_np: np.ndarray, y_np: np.ndarray, meta,
             fold_defs: List[Dict], device: Optional[torch.device] = None,
             cfg: Dict = None, seed: int = C.SEED) -> Dict:
    device = device or get_device()
    cfg = cfg or C.TRAIN_CONFIG

    X = torch.from_numpy(X_np).float().to(device)
    y = torch.from_numpy(y_np).long().to(device)
    print(f"  device: {device}  cache resident: "
          f"{X.element_size() * X.nelement() / 1e6:.1f} MB", flush=True)
    print(f"  epochs={cfg['epochs']} early_stop={cfg.get('early_stop', False)} "
          f"seeds={cfg.get('n_seeds', 1)} tta={tuple(cfg.get('tta_shifts', (0,)))}",
          flush=True)

    filenames_all = meta["filename"].to_numpy()
    bearings_all = meta["bearing"].to_numpy()

    results = []
    pool_truth, pool_pred, pool_files, pool_bearing, pool_probs = [], [], [], [], []
    t0 = time.time()
    for fd in fold_defs:
        res, _, _, (truth, pred, prob) = train_one_fold(
            X, y, meta, fd, device, cfg, seed)
        results.append(res)
        sel = np.isin(bearings_all, fd["test_bearings"])
        pool_truth.append(truth)
        pool_pred.append(pred)
        pool_probs.append(prob)
        pool_files.append(filenames_all[sel])
        pool_bearing.append(bearings_all[sel])

    truth = np.concatenate(pool_truth)
    pred = np.concatenate(pool_pred)
    prob = np.concatenate(pool_probs)
    files = np.concatenate(pool_files)
    bear = np.concatenate(pool_bearing)

    # -- POOLED metrics are the headline -----------------------------------
    # Each bearing is tested exactly once, so concatenating every fold's
    # predictions gives one out-of-sample prediction per window across the whole
    # dataset.  Per-FOLD macro-F1 would be meaningless: a single-bearing test
    # set contains one class, so its F1 caps at 1/n_classes and its majority
    # baseline is always 1.0.
    confusion = np.zeros((len(C.LABELS), len(C.LABELS)), dtype=np.int64)
    for t, p in zip(truth, pred):
        confusion[t, p] += 1
    rec_acc, n_rec = recording_vote(files, truth, pred, len(C.LABELS),
                                    probs=prob if C.SOFT_VOTE else None)

    per_bearing_acc = np.array([r.window_acc for r in results])

    summary = {
        "aggregation_curve": aggregation_curve(files, bear, truth, prob),
        "scheme": "leave-one-bearing-out, pooled",
        "n_folds": len(results),
        "pooled": {
            "window_acc": float((pred == truth).mean()),
            "macro_f1": macro_f1(truth, pred, len(C.LABELS)),
            "recording_acc": float(rec_acc),
            "n_windows": int(truth.size),
            "n_recordings": int(n_rec),
            "majority_baseline": float(F_.majority_baseline(truth)),
        },
        "per_bearing_acc_mean": float(per_bearing_acc.mean()),
        "per_bearing_acc_std": float(per_bearing_acc.std(ddof=1)),
        "per_bearing_acc_sem": float(per_bearing_acc.std(ddof=1) / np.sqrt(len(results))),
        "confusion": confusion.tolist(),
        "labels": C.LABELS,
        "seconds": round(time.time() - t0, 1),
        "folds": [asdict(r) for r in results],
    }

    # -- by damage origin: pooled, never mixed ------------------------------
    # Healthy bearings join both groups because they are the reference class for
    # each; artificial and real damage are never pooled together, since v4
    # measured artificial->real transfer at 37.9 %.
    origin_of = {r.test_bearing: r.origin for r in results}
    origins = np.array([origin_of[b] for b in bear])
    by_origin = {}
    for name in ("real", "artificial"):
        sel = (origins == name) | (origins == "none")
        if sel.any():
            by_origin[name] = {
                "n_bearings": int(len({b for b, o in origin_of.items()
                                       if o in (name, "none")})),
                "window_acc": float((pred[sel] == truth[sel]).mean()),
                "macro_f1": macro_f1(truth[sel], pred[sel], len(C.LABELS)),
                "majority_baseline": float(F_.majority_baseline(truth[sel])),
            }
    summary["by_damage_origin"] = by_origin

    # Per-window out-of-sample probabilities, so the aggregation curve and any
    # later threshold analysis can be recomputed without retraining.
    np.savez_compressed(
        os.path.join(C.RUN_DIR, "pooled_predictions.npz"),
        truth=truth, pred=pred, probs=prob.astype(np.float32),
        filenames=files.astype(str), bearings=bear.astype(str),
    )
    return summary


# ===========================================================================
# the model that ships
# ===========================================================================

def train_deployment_model(X_np: np.ndarray, y_np: np.ndarray, meta,
                           device: Optional[torch.device] = None,
                           cfg: Dict = None, seed: int = C.SEED):
    """
    Trained on every bearing, no holdout.

    Correct rather than sloppy: deployment targets bearings none of which are in
    this dataset, so holding one out buys nothing at ship time.  The honest
    accuracy estimate is the leave-one-bearing-out sweep; this is the artefact
    that gets quantised.

    Always a SINGLE model regardless of cfg["n_seeds"] -- the FPGA runs one
    network, so shipping an ensemble would make the exported weights stop
    matching the accuracy that was reported for them.
    """
    device = device or get_device()
    cfg = dict(cfg or C.TRAIN_CONFIG)
    np.random.seed(seed)

    X = torch.from_numpy(X_np).float().to(device)
    y = torch.from_numpy(y_np).long().to(device)

    bearings = meta["bearing"].to_numpy()
    labels_by_bearing = dict(zip(meta["bearing"], meta["label"]))
    fit_b, val_b = F_.inner_split(sorted(set(bearings)), labels_by_bearing, seed=seed)
    is_val = np.isin(bearings, val_b)
    i_fit = torch.from_numpy(np.flatnonzero(~is_val)).to(device)
    i_val = torch.from_numpy(np.flatnonzero(is_val)).to(device)

    mean, std = channel_stats(X[i_fit])
    Xf, yf = (X[i_fit] - mean) / std, y[i_fit]
    Xv, yv = ((X[i_val] - mean) / std, y[i_val]) if i_val.numel() else (X[:0], y[:0])

    counts = torch.bincount(yf, minlength=len(C.LABELS)).float()
    weights = (counts.sum() / (len(C.LABELS) * counts.clamp_min(1))).to(device)

    model, best, best_epoch = _fit_one(Xf, yf, Xv, yv, weights, cfg, device, seed)

    return model, {
        "mean": mean.cpu().numpy(), "std": std.cpu().numpy(),
        "val_bearings": val_b, "val_acc": best, "best_epoch": best_epoch,
    }
