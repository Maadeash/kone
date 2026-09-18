"""
Render artifacts/multistage/*/ JSON into docs/results_multistage.md.

Per workflow_v2.md §9 rule 2, results markdown is GENERATED, never hand-typed.
Anything not run reads NOT RUN.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivesentinel import config as C

BRANCHES = [
    ("winding", "B-S4", "S4", "D2 KAIST PMSM"),
    ("supply", "B-S1", "S1", "D4 Thomas"),
    ("inverter_telemetry", "B-S2/S3", "S2/S3", "D3 Bacha"),
]
OUT = os.path.join(C.PROJECT_ROOT, "docs", "results_multistage.md")


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return "NaN" if v != v else f"{v:.{nd}f}"
    return str(v)


def render_winding(r):
    L = []
    L.append(f"**Task: {r['task']}.** {r['n_windows']:,} windows, "
             f"{r['window_s']} s / {r['hop_s']} s hop, "
             f"{len(r['feature_names'])} ratio features.\n")

    L.append("### `healthy`: NOT MEASURABLE\n")
    L.append(f"> {r['healthy_reason']}\n")
    L.append("No healthy number is reported here, on any metric card, or as a fusion "
             "input. The confound is total rather than partial, so a caveated number "
             "would be worse than none.\n")

    # -- V4 first: it is the headline -----------------------------------
    L.append("### V4 — batch control (headline)\n")
    L.append("Same features, predicting the acquisition variable instead of the fault "
             "class.\n")
    L.append("| Target | Split | Accuracy | Macro-F1 | Baseline |")
    L.append("|---|---|---|---|---|")
    for k, v in r["V4"]["targets"].items():
        if "accuracy" not in v:
            continue
        leaky = k.endswith("_shuffled_LEAKY")
        split = "shuffled **(leaky reference)**" if leaky else "leave-one-motor-out"
        name = k.replace("_shuffled_LEAKY", "")
        L.append(f"| {name} | {split} | {fmt(v['accuracy'])} | {fmt(v['macro_f1'])} "
                 f"| {fmt(v['majority_baseline'])} |")
    L.append("")

    p = r["V4_probe"]
    L.append("#### Model-free probe — 1-NN on `i0rel_residual` alone\n")
    L.append(f"> {p['note']}\n")
    L.append("| Subset | n | → session | → batch | → chassis | → **fault class** | fault baseline |")
    L.append("|---|---|---|---|---|---|---|")
    for sub, d in p["subsets"].items():
        L.append(f"| {sub} | {d['n']} | {fmt(d['session']['loo_1nn_accuracy'],3)} "
                 f"| {fmt(d['batch']['loo_1nn_accuracy'],3)} "
                 f"| {fmt(d['chassis']['loo_1nn_accuracy'],3)} "
                 f"| **{fmt(d['fault_class']['loo_1nn_accuracy'],3)}** "
                 f"| {fmt(d['fault_class']['majority_baseline'],3)} |")
    L.append("")
    sob = p["subsets"].get("1000W_1500W_only", {}).get("fault_class", {})
    L.append(f"**Session-only baseline = {fmt(sob.get('loo_1nn_accuracy'),3)}** "
             f"(fault class, 1000/1500 W, from a scalar that cannot contain winding "
             f"information). Any fault-class score that does not clearly beat this is "
             f"**not demonstrating winding diagnosis**.\n")

    # -- V1 vs V5 --------------------------------------------------------
    L.append("### V1 vs V5 — the gap is the result\n")
    L.append("| Protocol | Scheme | Groups scored | Accuracy | Macro-F1 | Baseline |")
    L.append("|---|---|---|---|---|---|")
    for key, label in (("V1", "leave-one-motor-out"), ("V5", "leave-one-session-out")):
        d = r[key]["pooled"]
        L.append(f"| **{key}** | {label} | {r[key]['n_groups_scored']} "
                 f"| {fmt(d['accuracy'])} | {fmt(d['macro_f1'])} "
                 f"| {fmt(d['majority_baseline'])} |")
    v3 = r["V3"]["pooled"]
    L.append(f"| V3 **(leaky reference)** | shuffled windows | — "
             f"| {fmt(v3['accuracy'])} | {fmt(v3['macro_f1'])} "
             f"| {fmt(v3['majority_baseline'])} |")
    L.append(f"| — | **session-only baseline** | — | {fmt(sob.get('loo_1nn_accuracy'),3)} "
             f"| — | {fmt(sob.get('majority_baseline'),3)} |")
    L.append("")

    for key, label in (("V1", "V1 — leave-one-motor-out"),
                       ("V5", "V5 — leave-one-session-out")):
        L.append(f"#### {label}\n")
        if r[key].get("excluded_reason"):
            L.append(f"> {r[key]['excluded_reason']}\n")
        L.append("| Fold | Held out | Accuracy | Macro-F1 | Baseline | Train windows | Test windows |")
        L.append("|---|---|---|---|---|---|---|")
        for f in r[key]["folds"]:
            if f.get("excluded"):
                L.append(f"| {f['fold']} | {', '.join(f['held_out'])} | — | — | — | — | "
                         f"**EXCLUDED**: {f['reason']} |")
            else:
                L.append(f"| {f['fold']} | {', '.join(f['held_out'])} "
                         f"| {fmt(f['accuracy'])} | {fmt(f['macro_f1'])} "
                         f"| {fmt(f['majority_baseline'])} "
                         f"| {f['train_windows']} | {f['test_windows']} |")
        L.append("")

    # -- fusion ----------------------------------------------------------
    fu = r["fusion"]
    L.append("### Fusion authority\n")
    L.append(f"| Field | Value |")
    L.append(f"|---|---|")
    L.append(f"| Protocol used | {fu['protocol_used']} |")
    L.append(f"| Validation groups scored | {fu['n_validation_groups']} "
             f"(floor: {fu['floor']['min_validation_groups']}) |")
    L.append(f"| Honest macro-F1 | {fmt(fu['honest_macro_f1'])} "
             f"(floor: {fu['floor']['min_macro_f1']}) |")
    L.append(f"| Floor fixed at | {fu['floor']['fixed_at']}, commit "
             f"`{fu['floor']['fixed_at_commit'][:8]}` |")
    L.append(f"| **Status** | **{fu['status']}** |")
    L.append("")
    L.append(f"{fu['reason']} The floor was pre-registered before any multi-stage "
             f"branch existed and has not been adjusted.\n")
    return L


def main():
    lines = ["# DriveSentinel v2 — multi-stage results",
             "",
             "Generated by `scripts/60_render_multistage_results.py` from the JSON in",
             "`artifacts/multistage/`. Do not edit by hand.",
             "",
             "Every number carries its protocol. Leaky numbers are labelled",
             "**(leaky reference)** in the heading as well as the body.",
             "", "---", ""]

    for slug, code, stage, dataset in BRANCHES:
        path = os.path.join(C.ARTIFACT_DIR, "multistage", slug, f"{slug}_results.json")
        lines.append(f"## {code} `{slug}` — stage {stage}, {dataset}")
        lines.append("")
        if not os.path.exists(path):
            lines += ["**NOT RUN**", ""]
            continue
        with open(path) as fh:
            r = json.load(fh)
        if slug == "winding":
            lines += render_winding(r)
        else:
            lines += ["**NOT RUN**", ""]
        lines.append(f"_Source: `{os.path.relpath(path, C.PROJECT_ROOT)}`, "
                     f"{r.get('seconds', '?')} s._")
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
