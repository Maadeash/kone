"""
DriveSentinel v5 -- export.py
=============================
Writes the deployment artefacts: INT8 weight memories, a scale manifest, and a
standalone NumPy golden reference for RTL verification.

WHAT GETS WRITTEN
-----------------
    layer_<i>_W.mem     int8  weights, hex, one value per line, row-major
    layer_<i>_b.mem     int32 biases, hex, one value per line
    scales.json         per-layer shapes, strides, scales, accumulator analysis
    input_norm.json     the 2 channel means and 2 channel std devs
    golden_reference.py pure-NumPy inference, imports nothing but numpy
    verification.json   INT8-vs-float agreement on held-out windows

THE GOLDEN REFERENCE READS THE SAME .mem FILES THE RTL WILL
-----------------------------------------------------------
It does not inline the weights.  That is deliberate: if both the Verilog and the
Python model load the identical memory images, then any mismatch between them is
a bug in the logic and cannot be a transcription error in the weights.  Inlining
would create a second copy that can silently drift from the first.

MEM FILE FORMAT
---------------
`$readmemh` compatible: one unsigned hex value per line, no prefix, row-major.
INT8 values are written as two-digit two's complement (-1 -> ff), INT32 as eight
digits.  Row i of a weight file is output channel i, so a hardware line buffer
walks the file in the order it needs.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch

from . import config as C
from .model import build_model, fold_batchnorm, OrderSpectrumCNN
from .quantize import QuantLayer, quantize_model, int8_forward, float_forward, agreement


# ===========================================================================
# .mem writers
# ===========================================================================

def _hex_int8(v: int) -> str:
    return f"{np.uint8(np.int8(v)):02x}"


def _hex_int32(v: int) -> str:
    return f"{np.uint32(np.int32(v)):08x}"


def write_mem_int8(path: str, arr: np.ndarray) -> int:
    flat = arr.reshape(-1)
    with open(path, "w") as fh:
        fh.write("\n".join(_hex_int8(int(v)) for v in flat) + "\n")
    return flat.size


def write_mem_int32(path: str, arr: np.ndarray) -> int:
    flat = arr.reshape(-1)
    with open(path, "w") as fh:
        fh.write("\n".join(_hex_int32(int(v)) for v in flat) + "\n")
    return flat.size


def read_mem_int8(path: str, shape) -> np.ndarray:
    with open(path) as fh:
        vals = [int(ln.strip(), 16) for ln in fh if ln.strip()]
    return np.array([v - 256 if v > 127 else v for v in vals], dtype=np.int8).reshape(shape)


def read_mem_int32(path: str, shape) -> np.ndarray:
    with open(path) as fh:
        vals = [int(ln.strip(), 16) for ln in fh if ln.strip()]
    return np.array([v - 2 ** 32 if v >= 2 ** 31 else v for v in vals],
                    dtype=np.int64).astype(np.int32).reshape(shape)


# ===========================================================================
# the generated golden reference
# ===========================================================================

_GOLDEN_TEMPLATE = '''"""
DriveSentinel INT8 Golden Reference  (auto-generated -- do not edit)
====================================================================
Standalone NumPy model of the quantised order-spectrum CNN.  Depends on numpy
and nothing else: no torch, no drivesentinel package.

PURPOSE
-------
Bit-level specification for the RTL.  The Verilog implementation must reproduce
the argmax of run_int8() for every input window.  Both this file and the RTL
load the SAME .mem files sitting next to it, so a disagreement is a bug in the
logic and cannot be a weight-transcription error.

INPUT CONTRACT
--------------
    x : float array, shape ({n_ch}, {n_bins})
        Conditioned order spectra, channel 0 = current sideband fold,
        channel 1 = vibration envelope, produced by the DSP front end:
        {contract_summary}
    The input is standardised inside run_int8() using INPUT_MEAN / INPUT_STD,
    so callers pass the DSP output directly.

ARITHMETIC
----------
    x_int8    = clamp(round(x / s_x), -128, 127)
    acc_int32 = sum(W_int8 * x_int8) + b_int32          # exact integer
    y_float   = s_W[out_channel] * s_x * acc_int32
    relu, then the next layer  (omitted on the last layer)

Accumulators are int32.  The widest layer sums {max_terms} products, worst case
{max_peak:,}, which needs {max_bits} bits -- so no saturation logic is required.

CLASSES: {class_map}

Generated from: {source_ckpt}
Contract fingerprint: {fingerprint}
"""

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = {class_map}
N_CHANNELS = {n_ch}
N_BINS = {n_bins}

with open(os.path.join(_HERE, "scales.json")) as _fh:
    SCALES = json.load(_fh)
with open(os.path.join(_HERE, "input_norm.json")) as _fh:
    _NORM = json.load(_fh)

INPUT_MEAN = np.array(_NORM["mean"], dtype=np.float64).reshape(N_CHANNELS, 1)
INPUT_STD = np.array(_NORM["std"], dtype=np.float64).reshape(N_CHANNELS, 1)


def _read_mem(path, shape, signed_bits):
    with open(path) as fh:
        vals = [int(ln.strip(), 16) for ln in fh if ln.strip()]
    half = 1 << (signed_bits - 1)
    full = 1 << signed_bits
    out = np.array([v - full if v >= half else v for v in vals], dtype=np.int64)
    return out.reshape(shape)


def _load_layers():
    layers = []
    for spec in SCALES["layers"]:
        W = _read_mem(os.path.join(_HERE, spec["W_file"]), spec["W_shape"], 8)
        b = _read_mem(os.path.join(_HERE, spec["b_file"]), spec["b_shape"], 32)
        layers.append({{
            "name": spec["name"], "kind": spec["kind"],
            "W": W.astype(np.int32), "b": b.astype(np.int32),
            "s_W": np.array(spec["s_W"], dtype=np.float64),
            "s_x": float(spec["s_x"]),
            "stride": spec["stride"], "padding": spec["padding"],
        }})
    return layers


LAYERS = _load_layers()


def _quant(x, s):
    return np.clip(np.rint(x / s), -128, 127).astype(np.int32)


def run_int8(x):
    """
    x : (N, {n_ch}, {n_bins}) or ({n_ch}, {n_bins}) float, raw DSP output.
    returns (N, {n_classes}) float logits.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, ...]
    a = (a - INPUT_MEAN[None]) / INPUT_STD[None]

    for L in LAYERS:
        if L["kind"] == "conv":
            xq = _quant(a, L["s_x"])
            pad = L["padding"]
            if pad:
                xq = np.pad(xq, ((0, 0), (0, 0), (pad, pad)))
            k = L["W"].shape[2]
            stride = L["stride"]
            out_len = (xq.shape[2] - k) // stride + 1
            cols = np.stack(
                [xq[:, :, i * stride:i * stride + k] for i in range(out_len)], axis=1
            )
            acc = np.einsum("blik,oik->bol", cols, L["W"], optimize=True)
            acc = acc + L["b"][None, :, None]
            a = acc.astype(np.float64) * (L["s_W"][None, :, None] * L["s_x"])
            a = np.maximum(a, 0.0)
        else:
            a = a.mean(axis=2)
            xq = _quant(a, L["s_x"])
            acc = xq @ L["W"].T + L["b"][None, :]
            a = acc.astype(np.float64) * (L["s_W"][None, :] * L["s_x"])
    return a


def predict(x):
    """Class indices. Map through CLASS_NAMES for labels."""
    return run_int8(x).argmax(axis=1)


if __name__ == "__main__":
    print("DriveSentinel INT8 golden reference")
    print(f"  input      : ({{N_CHANNELS}}, {{N_BINS}})")
    print(f"  classes    : {{CLASS_NAMES}}")
    for L in LAYERS:
        print(f"  {{L['name']:6}} {{L['kind']:6}} W{{tuple(L['W'].shape)}} "
              f"s_x={{L['s_x']:.6g}}")
    probe = np.zeros((1, N_CHANNELS, N_BINS))
    print(f"  zero input -> class {{int(predict(probe)[0])}} "
          f"({{CLASS_NAMES[int(predict(probe)[0])]}})")
'''


def write_golden_reference(path: str, layers: List[QuantLayer],
                           source_ckpt: str, fingerprint: str) -> None:
    acc = max((L.n_in * L.kernel, L) for L in layers)
    max_terms = acc[0]
    max_peak = max_terms * 128 * 127
    text = _GOLDEN_TEMPLATE.format(
        n_ch=C.N_INPUT_CHANNELS, n_bins=C.N_ORDER_BINS,
        n_classes=len(C.LABELS),
        class_map={i: n for i, n in enumerate(C.LABELS)},
        contract_summary=C.summary(),
        source_ckpt=os.path.basename(source_ckpt),
        fingerprint=fingerprint,
        max_terms=max_terms, max_peak=max_peak,
        max_bits=int(np.ceil(np.log2(max_peak + 1))) + 1,
    )
    with open(path, "w") as fh:
        fh.write(text)


# ===========================================================================
# top level
# ===========================================================================

def export(ckpt_path: str = None, out_dir: str = None,
           calib: np.ndarray = None, verify: np.ndarray = None,
           verify_labels: np.ndarray = None) -> Dict:
    ckpt_path = ckpt_path or os.path.join(C.RUN_DIR, "deployment_model.pt")
    out_dir = out_dir or C.EXPORT_DIR
    os.makedirs(out_dir, exist_ok=True)

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(blob["state_dict"])
    model.eval()

    norm_mean = np.asarray(blob["norm_mean"], dtype=np.float64).reshape(-1)
    norm_std = np.asarray(blob["norm_std"], dtype=np.float64).reshape(-1)
    assert norm_mean.size == C.N_INPUT_CHANNELS, (
        f"expected {C.N_INPUT_CHANNELS} per-channel normalisation constants, "
        f"got {norm_mean.size} -- the exporter cannot emit per-bin statistics"
    )

    folded = fold_batchnorm(model)

    def standardise(x):
        return (x - norm_mean.reshape(1, -1, 1)) / norm_std.reshape(1, -1, 1)

    calib_n = standardise(np.asarray(calib, dtype=np.float64))
    layers, qmeta = quantize_model(
        folded, torch.from_numpy(calib_n).float()
    )

    # -- memories ----------------------------------------------------------
    manifest = []
    for i, L in enumerate(layers):
        w_file, b_file = f"layer_{i}_W.mem", f"layer_{i}_b.mem"
        write_mem_int8(os.path.join(out_dir, w_file), L.W_int8)
        write_mem_int32(os.path.join(out_dir, b_file), L.b_int32)
        manifest.append({
            "index": i, "name": L.name, "kind": L.kind,
            "W_file": w_file, "b_file": b_file,
            "W_shape": list(L.W_int8.shape), "b_shape": list(L.b_int32.shape),
            "s_W": [float(v) for v in L.s_W], "s_x": float(L.s_x),
            "stride": int(L.stride), "padding": int(L.padding),
            "is_last": bool(L.is_last),
            "note": ("acc_int32 = sum(W_int8 * x_int8) + b_int32; "
                     "y = s_W[o] * s_x * acc; relu unless is_last"),
        })

    fingerprint = blob.get("contract", "n/a")
    with open(os.path.join(out_dir, "scales.json"), "w") as fh:
        json.dump({
            "format": "hex_no_prefix_row_major_two_complement",
            "input": {"channels": C.N_INPUT_CHANNELS, "bins": C.N_ORDER_BINS},
            "classes": C.LABELS,
            "quantisation": {k: v for k, v in qmeta.items() if k != "accumulator"},
            "accumulator": qmeta["accumulator"],
            "layers": manifest,
        }, fh, indent=2)

    with open(os.path.join(out_dir, "input_norm.json"), "w") as fh:
        json.dump({"mean": norm_mean.tolist(), "std": norm_std.tolist(),
                   "applied": "x = (x - mean) / std, per channel, before layer 0"},
                  fh, indent=2)

    write_golden_reference(os.path.join(out_dir, "golden_reference.py"),
                           layers, ckpt_path, str(fingerprint))

    # -- verification ------------------------------------------------------
    report = {"n_weights": qmeta["n_weights"], "n_biases": qmeta["n_biases"],
              "accumulator": qmeta["accumulator"]}
    if verify is not None:
        vn = standardise(np.asarray(verify, dtype=np.float64))
        report["agreement"] = agreement(folded, layers, vn)
        if verify_labels is not None:
            f_pred = float_forward(folded, vn).argmax(1)
            q_pred = int8_forward(layers, vn).argmax(1)
            report["float_accuracy"] = float((f_pred == verify_labels).mean())
            report["int8_accuracy"] = float((q_pred == verify_labels).mean())
    with open(os.path.join(out_dir, "verification.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report
