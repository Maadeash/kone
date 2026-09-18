"""
Rule-based fusion across drive stages.  No learning, by design.

WHY RULES AND NOT A LEARNED COMBINER
------------------------------------
A learned fusion model needs examples of the same physical machine exhibiting
faults at several stages.  No dataset in the roster shares a machine across
branches -- Paderborn bearings, KAIST windings, Bacha inverters and Thomas supply
faults are four different rigs.  Training a combiner on them would be learning
which dataset a window came from, which is the failure this project spends most of
its effort measuring elsewhere.  So: explicit rules, stated thresholds, and a demo
labelled COMPOSITE REPLAY.

THE AUTHORITY RULE IS THE POINT
-------------------------------
Evidence accumulation is ordinary engineering.  The part that matters is that a
branch cannot raise `Fault` unless its own validated metric earns it:

    >= min_validation_groups independent validation groups, AND
    >= min_macro_f1 honest (non-leaky) macro-F1

Both numbers were pre-registered on 2026-09-18 at commit 7344a436, before any
multi-stage branch existed (`config.FUSION_CONFIG`, `docs/claims_audit.md` §1.1).
A branch that misses either test is INDICATIVE: it may raise `Warning` and
contribute evidence, but never `Fault`.

Authority is read from each branch's results JSON at load time and never
hardcoded, so it cannot drift away from the measured number.  A branch whose JSON
is missing is `NOT MEASURED` and gets no authority at all -- absence of a metric
is not the same as a bad metric, but it earns the same lack of trust.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

from . import config as C

# Worst-first. `system_status` returns the worst status across branches.
STATUS_ORDER = ("Normal", "Warning", "Fault")

STAGE_OF_BRANCH = {
    "bearing": "S5", "winding": "S4", "supply": "S1", "inverter_telemetry": "S3",
}


# ===========================================================================
# branch metrics and authority
# ===========================================================================

@dataclass
class BranchMetric:
    """What a branch measured about itself, and whether it earns Fault authority."""
    name: str
    stage: str
    metric_name: str = "macro_f1"
    metric_value: Optional[float] = None
    protocol: Optional[str] = None
    n_validation_groups: Optional[int] = None
    measured: bool = False
    source: Optional[str] = None
    note: Optional[str] = None

    @property
    def fault_authority(self) -> bool:
        if not self.measured:
            return False
        return C.fault_authority(self.n_validation_groups, self.metric_value)

    @property
    def tier(self) -> str:
        if not self.measured:
            return "NOT MEASURED"
        return "Fault-capable" if self.fault_authority else "INDICATIVE"

    def describe(self) -> str:
        if not self.measured:
            return f"{self.name}: NOT MEASURED" + (f" ({self.note})" if self.note else "")
        return (f"{self.name}: {self.metric_name} {self.metric_value:.4f} "
                f"({self.protocol}, {self.n_validation_groups} groups) -> {self.tier}")


def _bearing_metric(run_dir: str = None) -> BranchMetric:
    """S5 reads lobo_summary.json: 29 bearings, pooled macro-F1."""
    path = os.path.join(run_dir or C.RUN_DIR, "lobo_summary.json")
    if not os.path.exists(path):
        return BranchMetric("bearing", "S5", measured=False,
                            note=f"{os.path.basename(path)} absent")
    with open(path) as fh:
        d = json.load(fh)
    return BranchMetric(
        name="bearing", stage="S5",
        metric_value=float(d["pooled"]["macro_f1"]),
        protocol="leave-one-bearing-out",
        n_validation_groups=int(d["n_folds"]),
        measured=True, source=path)


def _multistage_metric(slug: str, stage: str, art_dir: str = None) -> BranchMetric:
    path = os.path.join(art_dir or C.ARTIFACT_DIR, "multistage", slug,
                        f"{slug}_results.json")
    if not os.path.exists(path):
        return BranchMetric(slug, stage, measured=False,
                            note=f"{slug}_results.json absent -- branch NOT RUN")
    with open(path) as fh:
        d = json.load(fh)
    fu = d.get("fusion", {})
    return BranchMetric(
        name=slug, stage=stage,
        metric_value=fu.get("honest_macro_f1"),
        protocol=fu.get("protocol_used"),
        n_validation_groups=fu.get("n_validation_groups"),
        measured=fu.get("honest_macro_f1") is not None,
        source=path, note=d.get("healthy_status"))


def load_branch_metrics(run_dir: str = None, art_dir: str = None
                        ) -> Dict[str, BranchMetric]:
    """
    Every branch's authority, read from disk.

    Never hardcoded. As measured on 2026-09-18: S5 bearing holds Fault authority
    (29 groups, macro-F1 0.7852); S4 winding is INDICATIVE (2 scored sessions,
    0.6250 -- fails both tests); S1 supply and S2/S3 inverter are NOT RUN.
    """
    out = {"bearing": _bearing_metric(run_dir)}
    for slug, stage in (("winding", "S4"), ("supply", "S1"),
                        ("inverter_telemetry", "S3")):
        out[slug] = _multistage_metric(slug, stage, art_dir)
    return out


# ===========================================================================
# per-branch evidence accumulation
# ===========================================================================

@dataclass
class BranchState:
    """
    Rolling evidence for one branch, and its status.

    `update()` takes a per-window probability vector and returns the branch's
    current view. Status changes need K consecutive updates in the same direction,
    and clearing needs p_fault below tau_W - h, so a single noisy window can
    neither raise nor clear an alarm.
    """
    metric: BranchMetric
    labels: Sequence[str]
    fault_labels: Sequence[str] = ()
    cfg: Dict = field(default_factory=lambda: dict(C.FUSION_CONFIG))

    def __post_init__(self):
        self.window: Deque[np.ndarray] = deque(maxlen=int(self.cfg["rolling_window"]))
        self.status = "Normal"
        self._raise_count = 0
        self._clear_count = 0
        self._warn_count = 0
        self.n_obs_total = 0
        if not self.fault_labels:
            self.fault_labels = tuple(l for l in self.labels
                                      if l not in ("healthy", "normal"))

    # -- accumulation ---------------------------------------------------

    def update(self, probs: Sequence[float]) -> Dict:
        p = np.asarray(probs, dtype=np.float64)
        if p.shape != (len(self.labels),):
            raise ValueError(f"{self.metric.name}: expected {len(self.labels)} "
                             f"probabilities, got {p.shape}")
        self.window.append(p)
        self.n_obs_total += 1
        return self._evaluate()

    def _evaluate(self) -> Dict:
        mean = np.mean(np.stack(self.window), axis=0)
        fault_idx = [i for i, l in enumerate(self.labels) if l in self.fault_labels]
        p_fault = float(mean[fault_idx].sum()) if fault_idx else 0.0
        top = int(mean.argmax())

        tau_F, tau_W = self.cfg["tau_fault"], self.cfg["tau_warning"]
        h, K = self.cfg["hysteresis"], int(self.cfg["k_consecutive"])

        # Counters first, so a run broken by one contrary sample restarts.
        self._raise_count = self._raise_count + 1 if p_fault >= tau_F else 0
        self._warn_count = self._warn_count + 1 if p_fault >= tau_W else 0
        self._clear_count = self._clear_count + 1 if p_fault < tau_W - h else 0

        may_fault = self.metric.fault_authority
        if self._raise_count >= K and may_fault:
            self.status = "Fault"
        elif self._warn_count >= K:
            # An indicative branch caps here however confident it is. That cap is
            # the whole reason the floor exists.
            if self.status != "Fault":
                self.status = "Warning"
        elif self._clear_count >= K:
            self.status = "Normal"

        return {
            "branch": self.metric.name,
            "stage": self.metric.stage,
            "status": self.status,
            "p_fault": p_fault,
            "top_class": self.labels[top],
            "confidence": float(mean[top]),
            "n_obs": len(self.window),
            "n_obs_total": self.n_obs_total,
            "tier": self.metric.tier,
            "fault_authority": may_fault,
        }

    def reset(self) -> None:
        self.window.clear()
        self.status = "Normal"
        self._raise_count = self._warn_count = self._clear_count = 0

    # -- reporting -------------------------------------------------------

    def evidence(self, extra: Optional[str] = None) -> str:
        """
        The one-line evidence string the dashboard shows.

        Names the stage, the status, the confidence, how many observations it is
        pooled over, and the branch's own honest metric with its protocol -- so a
        reader can see what the claim rests on without leaving the panel.
        """
        v = self._evaluate() if self.window else {
            "status": self.status, "confidence": 0.0, "n_obs": 0,
            "top_class": "-", "p_fault": 0.0}
        m = self.metric
        if m.measured:
            metric_txt = f"branch {m.protocol} {m.metric_name} {m.metric_value:.2f}"
        else:
            metric_txt = "branch NOT MEASURED"
        parts = [
            v["status"].upper(),
            f"{m.stage} {m.name}",
            f"{v['confidence'] * 100:.0f}%",
            f"{v['n_obs']} pooled obs",
        ]
        if extra:
            parts.append(extra)
        parts.append(metric_txt)
        if not m.fault_authority:
            parts.append("LOW CONFIDENCE BRANCH")
        return " · ".join(parts)


# ===========================================================================
# system level
# ===========================================================================

def worst(statuses: Sequence[str]) -> str:
    """Worst status wins. An unknown status is treated as the worst thing it could be."""
    if not statuses:
        return "Normal"
    return max(statuses, key=lambda s: STATUS_ORDER.index(s)
               if s in STATUS_ORDER else len(STATUS_ORDER))


def system_status(branch_views: Sequence[Dict]) -> Dict:
    """Worst stage, plus which branch set it."""
    if not branch_views:
        return {"status": "Normal", "driver": None, "branches": []}
    st = worst([v["status"] for v in branch_views])
    driver = next((v for v in branch_views if v["status"] == st), None)
    return {
        "status": st,
        "driver": driver["branch"] if driver else None,
        "driver_stage": driver["stage"] if driver else None,
        "branches": list(branch_views),
    }


def authority_table(metrics: Dict[str, BranchMetric]) -> List[Dict]:
    """Rows for the docs and the dashboard. Generated, never typed."""
    floor = C.FUSION_CONFIG["fault_authority"]
    rows = []
    for name in ("bearing", "winding", "supply", "inverter_telemetry"):
        m = metrics.get(name)
        if m is None:
            continue
        rows.append({
            "branch": name,
            "stage": m.stage,
            "measured": m.measured,
            "protocol": m.protocol,
            "n_validation_groups": m.n_validation_groups,
            "macro_f1": m.metric_value,
            "groups_test": (None if not m.measured else
                            m.n_validation_groups >= floor["min_validation_groups"]),
            "metric_test": (None if not m.measured else
                            m.metric_value >= floor["min_macro_f1"]),
            "tier": m.tier,
            "source": m.source,
            "note": m.note,
        })
    return rows
