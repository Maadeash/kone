"""
DriveSentinel v5 -- quantize.py
==============================
Post-training INT8 quantisation of the folded CNN, and the arithmetic the RTL
will have to match bit for bit.

THE SCHEME
----------
Symmetric, zero-point-free, so no offset arithmetic appears in the datapath:

    weights      per OUTPUT CHANNEL   s_W[o] = max|W[o]| / 127
    activations  per TENSOR           s_x    = percentile(|x|, 99.9) / 127
    bias         folded into the accumulator scale: s_b = s_W * s_x

    acc_int32 = conv_int8(x_int8, W_int8) + b_int32
    y_float   = s_W[o] * s_x * acc_int32

WHY PER-CHANNEL WEIGHTS
-----------------------
The old v2 export used per-tensor scaling for everything.  That is defensible
for a dense layer, where one output neuron's weights look much like another's.
It is not for convolutions: after BatchNorm folding the per-channel gain
gamma/sqrt(var+eps) varies by an order of magnitude between output channels, so
one shared scale wastes most of the INT8 range on every channel except the
largest.  Per-channel costs one extra scale per output channel -- 16, 32, 64, 64
and 3 floats in total -- and one multiply in the requantiser.

WHY A PERCENTILE AND NOT THE MAX FOR ACTIVATIONS
------------------------------------------------
A single spectral spike in one calibration window would otherwise set the scale
for the whole tensor and crush everything else toward zero.  The 99.9th
percentile clips that spike and keeps resolution where the data actually lives.
Clipping is applied explicitly at quantisation time so the golden reference and
the RTL agree about what happens to out-of-range values.

ACCUMULATOR WIDTH
-----------------
The widest layer accumulates in_channels * kernel = 64 * 3 = 192 products of two
INT8 values, worst case 192 * 127 * 127 = 3,096,768.  That needs 23 bits signed;
int32 leaves 8 bits of headroom, so no saturation logic is required in the
accumulator.  check_accumulator_headroom() asserts this rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .model import OrderSpectrumCNN, fold_batchnorm

INT8_MAX = 127


@dataclass
class QuantLayer:
    """One quantised layer, in the form the .mem writer and the RTL both need."""
    name: str
    kind: str                  # "conv" | "linear"
    W_int8: np.ndarray         # conv: (out, in, k)   linear: (out, in)
    b_int32: np.ndarray        # (out,)
    s_W: np.ndarray            # (out,) per-channel weight scales
    s_x: float                 # input activation scale
    stride: int
    padding: int
    is_last: bool

    @property
    def n_out(self) -> int:
        return self.W_int8.shape[0]

    @property
    def n_in(self) -> int:
        return self.W_int8.shape[1]

    @property
    def kernel(self) -> int:
        return self.W_int8.shape[2] if self.kind == "conv" else 1


def quantize_weights_per_channel(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """W -> (int8 weights, per-output-channel scales)."""
    flat = W.reshape(W.shape[0], -1)
    s = np.abs(flat).max(axis=1) / INT8_MAX
    s = np.where(s <= 0, 1.0, s)                      # a dead channel keeps scale 1
    q = np.clip(np.rint(W / s.reshape(-1, *([1] * (W.ndim - 1)))),
                -INT8_MAX - 1, INT8_MAX).astype(np.int8)
    return q, s.astype(np.float64)


def quantize_weights_per_tensor(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = float(np.abs(W).max() / INT8_MAX) or 1.0
    q = np.clip(np.rint(W / s), -INT8_MAX - 1, INT8_MAX).astype(np.int8)
    return q, np.full(W.shape[0], s, dtype=np.float64)


def activation_scale(x: np.ndarray, percentile: float = None) -> float:
    """Symmetric activation scale from a robust magnitude percentile."""
    percentile = percentile or C.QUANT_CONFIG["activation_percentile"]
    hi = float(np.percentile(np.abs(x), percentile))
    return (hi / INT8_MAX) if hi > 0 else 1.0


@torch.no_grad()
def collect_activation_scales(model: OrderSpectrumCNN,
                              calib: torch.Tensor) -> List[float]:
    """
    Input scale for every layer, measured on real calibration data.

    Hooks run on the FOLDED model, so the tensors seen here are exactly the ones
    the deployed graph will produce -- calibrating on the unfolded model would
    measure pre-BatchNorm activations that never exist at inference.
    """
    model.eval()
    scales: List[float] = [activation_scale(calib.cpu().numpy())]

    x = calib
    for stage in model.features:
        x = stage.conv(x)
        x = stage.act(x) if not isinstance(stage.bn, nn.Identity) else torch.relu(x)
        scales.append(activation_scale(x.cpu().numpy()))
    return scales


def check_accumulator_headroom(layers: List[QuantLayer], bits: int = 32) -> Dict:
    """Worst-case |accumulator| per layer against the int32 range."""
    limit = 2 ** (bits - 1) - 1
    worst = {}
    for L in layers:
        n_terms = L.n_in * L.kernel
        peak = n_terms * (INT8_MAX + 1) * INT8_MAX + int(np.abs(L.b_int32).max())
        worst[L.name] = {
            "terms": int(n_terms),
            "peak_abs": int(peak),
            "bits_needed": int(np.ceil(np.log2(peak + 1))) + 1,
            "fits_int32": bool(peak <= limit),
        }
        assert peak <= limit, f"{L.name} can overflow int{bits}: peak {peak}"
    return worst


@torch.no_grad()
def quantize_model(model: OrderSpectrumCNN, calib: torch.Tensor,
                   cfg: Dict = None) -> Tuple[List[QuantLayer], Dict]:
    """
    Folded float model + calibration batch -> list of QuantLayer.

    The model passed in MUST already be BatchNorm-folded; quantize_model folds
    defensively if it is not, because calibrating an unfolded model is a silent
    accuracy loss rather than an error.
    """
    cfg = cfg or C.QUANT_CONFIG
    if any(isinstance(s.bn, nn.BatchNorm1d) for s in model.features):
        model = fold_batchnorm(model)
    model.eval()

    scales_x = collect_activation_scales(model, calib)
    qfn = (quantize_weights_per_channel if cfg["per_channel_weights"]
           else quantize_weights_per_tensor)

    layers: List[QuantLayer] = []
    for i, stage in enumerate(model.features):
        conv = stage.conv
        W = conv.weight.detach().cpu().numpy().astype(np.float64)
        b = conv.bias.detach().cpu().numpy().astype(np.float64)
        W_q, s_W = qfn(W)
        s_x = scales_x[i]
        # Bias lives in the accumulator's own units, so it is scaled by the
        # PRODUCT of the weight and activation scales -- not by either alone.
        b_q = np.rint(b / (s_W * s_x)).astype(np.int64)
        b_q = np.clip(b_q, -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
        layers.append(QuantLayer(
            name=f"conv{i + 1}", kind="conv", W_int8=W_q, b_int32=b_q,
            s_W=s_W, s_x=s_x, stride=conv.stride[0], padding=conv.padding[0],
            is_last=False,
        ))

    fc = model.classifier
    W = fc.weight.detach().cpu().numpy().astype(np.float64)
    b = fc.bias.detach().cpu().numpy().astype(np.float64)
    W_q, s_W = qfn(W)
    s_x = scales_x[-1]
    b_q = np.clip(np.rint(b / (s_W * s_x)), -(2 ** 31), 2 ** 31 - 1).astype(np.int32)
    layers.append(QuantLayer(
        name="fc", kind="linear", W_int8=W_q, b_int32=b_q, s_W=s_W, s_x=s_x,
        stride=1, padding=0, is_last=True,
    ))

    meta = {
        "per_channel_weights": bool(cfg["per_channel_weights"]),
        "activation_percentile": cfg["activation_percentile"],
        "activation_scales": [float(s) for s in scales_x],
        "accumulator": check_accumulator_headroom(layers),
        "n_weights": int(sum(L.W_int8.size for L in layers)),
        "n_biases": int(sum(L.b_int32.size for L in layers)),
    }
    return layers, meta


# ===========================================================================
# reference INT8 inference -- the arithmetic the RTL must reproduce
# ===========================================================================

def _quantize_tensor(x: np.ndarray, s: float) -> np.ndarray:
    return np.clip(np.rint(x / s), -INT8_MAX - 1, INT8_MAX).astype(np.int32)


def int8_forward(layers: List[QuantLayer], x: np.ndarray) -> np.ndarray:
    """
    Pure-NumPy INT8 inference over a batch of NORMALISED inputs.

    Written the way hardware would do it -- integer accumulate, dequantise,
    ReLU, requantise -- so any divergence between this and the RTL is a bug in
    the RTL rather than a difference of formulation.  int8_export.py emits a
    standalone copy of this function with the weights inlined.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, ...]

    for L in layers:
        if L.kind == "conv":
            xq = _quantize_tensor(a, L.s_x)
            if L.padding:
                xq = np.pad(xq, ((0, 0), (0, 0), (L.padding, L.padding)))
            B, _, n = xq.shape
            out_len = (n - L.kernel) // L.stride + 1
            # explicit im2col so the loop order matches an RTL line buffer
            cols = np.stack(
                [xq[:, :, i * L.stride: i * L.stride + L.kernel] for i in range(out_len)],
                axis=1,
            )                                            # (B, out_len, in, k)
            acc = np.einsum("blik,oik->bol", cols, L.W_int8.astype(np.int32),
                            optimize=True)
            acc = acc + L.b_int32[None, :, None]
            a = acc.astype(np.float64) * (L.s_W[None, :, None] * L.s_x)
            a = np.maximum(a, 0.0)                       # ReLU
        else:
            a = a.mean(axis=2)                           # global average pool
            xq = _quantize_tensor(a, L.s_x)
            acc = xq.astype(np.int32) @ L.W_int8.astype(np.int32).T
            acc = acc + L.b_int32[None, :]
            a = acc.astype(np.float64) * (L.s_W[None, :] * L.s_x)
    return a


@torch.no_grad()
def float_forward(model: OrderSpectrumCNN, x: np.ndarray) -> np.ndarray:
    model.eval()
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if t.ndim == 2:
        t = t[None, ...]
    return model(t).cpu().numpy().astype(np.float64)


def agreement(model: OrderSpectrumCNN, layers: List[QuantLayer],
              x: np.ndarray) -> Dict:
    """
    Fraction of samples where INT8 and float pick the same class.

    The v3 methodology sets the bar at >= 98 %.  Reported alongside accuracy
    against the true labels, because agreement alone would look fine if both
    models were equally wrong.
    """
    f = float_forward(model, x)
    q = int8_forward(layers, x)
    fp, qp = f.argmax(1), q.argmax(1)
    return {
        "n": int(len(fp)),
        "argmax_agreement": float((fp == qp).mean()),
        "max_abs_logit_diff": float(np.abs(f - q).max()),
        "mean_abs_logit_diff": float(np.abs(f - q).mean()),
    }
