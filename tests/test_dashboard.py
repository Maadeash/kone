"""
Dashboard panels.

`panels.py` is deliberately free of Streamlit so the rules it enforces can be
tested without a browser. The rule that matters is `assert_out_of_sample`: a
bearing scenario that cannot prove the model never saw it must not render.
"""

import json
import os

import numpy as np
import pytest

from drivesentinel import config as C
from drivesentinel import fusion as FU
from dashboard import panels as P


# ---------------------------------------------------------------------------
# the out-of-sample rule
# ---------------------------------------------------------------------------

def test_bearing_scenario_without_the_flag_is_refused():
    sc = {"branch": "bearing", "scenario": "bearing_KA04", "bearing": "KA04",
          "out_of_sample": False, "train_bearings": np.array(["K001"])}
    with pytest.raises(ValueError, match="not marked out-of-sample"):
        P.assert_out_of_sample(sc)


def test_bearing_scenario_training_on_itself_is_refused():
    """The violation the rule exists to catch."""
    sc = {"branch": "bearing", "scenario": "bearing_KA04", "bearing": "KA04",
          "out_of_sample": True, "train_bearings": np.array(["K001", "KA04"])}
    with pytest.raises(ValueError, match="OUT-OF-SAMPLE VIOLATION"):
        P.assert_out_of_sample(sc)


def test_a_clean_bearing_scenario_passes():
    sc = {"branch": "bearing", "scenario": "bearing_KA04", "bearing": "KA04",
          "out_of_sample": True, "train_bearings": np.array(["K001", "K002"])}
    P.assert_out_of_sample(sc)


def test_non_bearing_scenarios_are_not_subject_to_the_rule():
    """The winding ramp is not a held-out-unit replay and does not claim to be."""
    P.assert_out_of_sample({"branch": "winding", "out_of_sample": False})


# ---------------------------------------------------------------------------
# stage rows and tier badges
# ---------------------------------------------------------------------------

def test_every_stage_gets_a_row():
    rows = P.stage_rows(FU.load_branch_metrics())
    assert [r["stage"] for r in rows] == ["S1", "S2", "S3", "S4", "S5"]


def test_each_row_carries_its_own_badge_not_a_legend():
    for r in P.stage_rows(FU.load_branch_metrics()):
        assert r["badge"] in ("FAULT-CAPABLE", "INDICATIVE", "NOT MEASURED")
        assert r["badge_colour"].startswith("#")


def test_bearing_stage_is_fault_capable_as_measured():
    rows = {r["branch"]: r for r in P.stage_rows(FU.load_branch_metrics())}
    if not rows["bearing"]["measured"]:
        pytest.skip("lobo_summary.json absent")
    assert rows["bearing"]["can_fault"] is True
    assert rows["bearing"]["badge"] == "FAULT-CAPABLE"


def test_winding_stage_is_indicative_as_measured():
    rows = {r["branch"]: r for r in P.stage_rows(FU.load_branch_metrics())}
    if not rows["winding"]["measured"]:
        pytest.skip("winding_results.json absent")
    assert rows["winding"]["can_fault"] is False
    assert rows["winding"]["badge"] == "INDICATIVE"


def test_unmeasured_stages_are_shown_not_hidden():
    """Never hide a stage: an empty one is more honest than a missing one."""
    rows = {r["stage"]: r for r in P.stage_rows(FU.load_branch_metrics())}
    assert "S1" in rows and "S2" in rows
    assert rows["S1"]["badge"] == "NOT MEASURED"
    assert rows["S1"]["metric_line"]          # still says something


def test_stage_row_metric_line_names_the_protocol():
    rows = {r["branch"]: r for r in P.stage_rows(FU.load_branch_metrics())}
    if rows["bearing"]["measured"]:
        assert "leave-one-bearing-out" in rows["bearing"]["metric_line"]


def test_status_defaults_to_not_measured_for_absent_branches():
    rows = {r["branch"]: r for r in P.stage_rows(FU.load_branch_metrics())}
    assert rows["supply"]["status"] == "NOT MEASURED"


# ---------------------------------------------------------------------------
# metric cards
# ---------------------------------------------------------------------------

def test_cards_exist_for_every_branch():
    names = {c["branch"] for c in P.metric_cards(FU.load_branch_metrics())}
    assert names == {"supply", "inverter_telemetry", "winding", "bearing"}


def test_cards_carry_the_floor_for_comparison():
    fa = C.FUSION_CONFIG["fault_authority"]
    for c in P.metric_cards(FU.load_branch_metrics()):
        assert c["groups_floor"] == fa["min_validation_groups"]
        assert c["metric_floor"] == fa["min_macro_f1"]


def test_winding_card_carries_the_session_only_baseline():
    cards = {c["branch"]: c for c in P.metric_cards(FU.load_branch_metrics())}
    if not cards["winding"]["measured"]:
        pytest.skip("winding_results.json absent")
    labels = [e["label"] for e in cards["winding"]["extra"]]
    assert "Session-only baseline" in labels
    assert any("leaky reference" in l for l in labels)


def test_winding_card_says_healthy_is_not_measurable():
    cards = {c["branch"]: c for c in P.metric_cards(FU.load_branch_metrics())}
    if not cards["winding"]["measured"]:
        pytest.skip("winding_results.json absent")
    entry = [e for e in cards["winding"]["extra"] if e["label"] == "healthy"]
    assert entry and entry[0]["value"] == "NOT MEASURABLE"


def test_unmeasured_card_has_no_metric_and_says_why():
    cards = {c["branch"]: c for c in P.metric_cards(FU.load_branch_metrics())}
    assert cards["supply"]["metric"] is None
    assert cards["supply"]["note"]


# ---------------------------------------------------------------------------
# trip panel
# ---------------------------------------------------------------------------

def test_trip_panel_reports_gating_disabled():
    assert P.trip_panel()["enabled"] is False


def test_trip_panel_carries_the_resolution_argument():
    tp = P.trip_panel()
    assert tp["gap"] == pytest.approx(1.89, abs=0.02)
    assert tp["at_900"] < tp["gap"] < tp["at_20"]
    assert tp["needed_window_at_20"] > 40


def test_trip_panel_caption_states_no_ramp_exists():
    assert "no dataset" in P.trip_panel()["caption"].lower()


def test_resolution_table_marks_extrapolation():
    rows = P.trip_panel()["resolution"]
    assert any(r["shaft_rpm"] == 900.0 for r in rows)
    assert any(r["shaft_rpm"] <= 20.0 for r in rows)


# ---------------------------------------------------------------------------
# replay through fusion
# ---------------------------------------------------------------------------

def _fake_scenario(p_fault=0.95, n=30, branch="bearing"):
    probs = np.tile([[1 - p_fault, p_fault, 0.0]], (n, 1)).astype(np.float32)
    return {"branch": branch, "stage": "S5", "bearing": "KA04",
            "scenario": "bearing_KA04", "out_of_sample": True,
            "train_bearings": np.array(["K001"]),
            "labels": np.array(C.LABELS), "probs": probs}


def test_replay_produces_a_step_per_window():
    v = P.replay(_fake_scenario(n=25), FU.load_branch_metrics())
    assert len(v["trace"]) == 25
    assert len(v["p_fault"]) == 25


def test_replay_of_a_capable_branch_can_reach_fault():
    m = FU.load_branch_metrics()
    if not m["bearing"].fault_authority:
        pytest.skip("bearing metric absent")
    v = P.replay(_fake_scenario(0.99, 30), m)
    assert v["final"]["status"] == "Fault"


def test_replay_of_an_indicative_branch_caps_at_warning():
    """End-to-end through the dashboard path, not just the fusion unit test."""
    m = FU.load_branch_metrics()
    if not m["winding"].measured:
        pytest.skip("winding_results.json absent")
    sc = _fake_scenario(0.99, 30, branch="winding")
    sc["stage"] = "S4"
    v = P.replay(sc, m)
    assert v["final"]["status"] == "Warning"
    assert "LOW CONFIDENCE BRANCH" in v["evidence"]


def test_replay_refuses_an_in_sample_scenario():
    sc = _fake_scenario()
    sc["train_bearings"] = np.array(["KA04"])
    with pytest.raises(ValueError, match="OUT-OF-SAMPLE VIOLATION"):
        P.replay(sc, FU.load_branch_metrics())


def test_evidence_string_names_the_stage_and_the_protocol():
    v = P.replay(_fake_scenario(0.9, 15), FU.load_branch_metrics())
    assert "S5" in v["evidence"] and "pooled obs" in v["evidence"]


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def test_manifest_loads_or_returns_an_empty_shell(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "DEMO_DIR", str(tmp_path))
    assert P.load_manifest() == {"scenarios": [], "built": None}


def test_built_manifest_marks_skipped_stages():
    man = P.load_manifest()
    if not man["scenarios"]:
        pytest.skip("scenarios not built")
    unbuilt = [s for s in man["scenarios"] if not s.get("file")]
    assert all(s.get("reason") for s in unbuilt), \
        "an unbuilt scenario must say why"


def test_built_bearing_scenarios_declare_out_of_sample():
    man = P.load_manifest()
    for s in man["scenarios"]:
        if s.get("branch") == "bearing" and s.get("file"):
            assert s["out_of_sample"] is True


# ---------------------------------------------------------------------------
# scenario files must be safe to load
# ---------------------------------------------------------------------------

def test_every_scenario_loads_without_pickling():
    """
    Regression. `filenames` came from a pandas column, so it was dtype=object,
    and numpy refuses to read an object array with allow_pickle=False. The
    dashboard disables pickling on purpose: a scenario file is something a judge
    might be handed, and loading it must not be able to execute code.
    """
    man = P.load_manifest()
    if not man["scenarios"]:
        pytest.skip("scenarios not built")
    for s in man["scenarios"]:
        if not s.get("file"):
            continue
        sc = P.load_scenario(s["file"])          # allow_pickle=False inside
        assert sc


def test_no_scenario_array_is_an_object_array():
    import numpy as _np
    man = P.load_manifest()
    if not man["scenarios"]:
        pytest.skip("scenarios not built")
    for s in man["scenarios"]:
        if not s.get("file"):
            continue
        with _np.load(os.path.join(P.DEMO_DIR, s["file"]), allow_pickle=True) as z:
            for k in z.files:
                assert z[k].dtype != object, f"{s['file']}:{k} is an object array"


def test_stored_window_count_is_capped():
    man = P.load_manifest()
    for s in man["scenarios"]:
        if s.get("branch") == "bearing" and s.get("file"):
            assert s["n_windows_stored"] <= 120
            assert s["n_windows_stored"] < s["n_windows_in_fold"]
