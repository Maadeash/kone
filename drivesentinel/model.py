"""
DriveSentinel v5 -- model.py
============================
The 1-D CNN that reads an order spectrum and returns a health verdict.

    input   (B, 2, 512)     current + vibration order spectra
    conv1   2  -> 16  k=9 s=2   -> 256
    conv2   16 -> 32  k=7 s=2   -> 128
    conv3   32 -> 64  k=5 s=2   ->  64
    conv4   64 -> 64  k=3 s=2   ->  32
    global average pool         ->  64
    linear  64 -> 3

    ~26.9k weights, ~1.58M MACs per inference.

DESIGN CONSTRAINTS THAT ARE NOT NEGOTIABLE
------------------------------------------
Every choice here exists so the network can be folded into a static INT8
datapath later.  Specifically:

  * BatchNorm sits immediately after each conv with no activation between, so
    it can be folded into the conv weights at export time and disappears from
    the deployed graph.  fold_batchnorm() does exactly that and is verified
    numerically by tests/test_model.py.
  * ReLU is the only non-linearity.  It is a comparison against zero -- free in
    hardware -- and it keeps activations non-negative, which lets the quantiser
    use the full unsigned INT8 range on activations if that is ever wanted.
  * Global average pooling replaces a flatten + large dense layer.  A flatten
    would make the classifier 64*32 = 2048 inputs wide and add 6k parameters
    for no accuracy; average pooling is a running sum and a shift.
  * No dropout.  It is a train-time-only op, but leaving it out keeps the
    train and deploy graphs identical, which removes one class of export bug.
  * Stride-2 convolutions instead of pooling layers: one operator, not two.

WHY THE NETWORK IS THIS SMALL
-----------------------------
The ceiling here is 29 physical bearings, not capacity.  A larger network fits
bearing identity faster than it fits fault physics, and leave-one-bearing-out
accuracy goes DOWN.  Capacity is not the lever; the input representation and
the augmentation are.
"""

from __future__ import annotations

import copy
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from . import config as C


class ConvStage(nn.Module):
    """conv -> batchnorm -> relu, in the order the folder expects."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int):
        super().__init__()
        # padding keeps the length arithmetic exact: out = ceil(in / stride)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              padding=kernel // 2, bias=True)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class OrderSpectrumCNN(nn.Module):
    def __init__(self, in_channels: int = None, n_classes: int = None,
                 conv_stages: Sequence[Tuple[int, int, int]] = None):
        super().__init__()
        in_channels = in_channels or C.MODEL_CONFIG["in_channels"]
        n_classes = n_classes or C.MODEL_CONFIG["n_classes"]
        conv_stages = conv_stages or C.MODEL_CONFIG["conv_stages"]

        stages: List[nn.Module] = []
        ch = in_channels
        for out_ch, kernel, stride in conv_stages:
            stages.append(ConvStage(ch, out_ch, kernel, stride))
            ch = out_ch
        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(ch, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    # -- introspection used by the export and the reports -------------------

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def mac_count(self, n_bins: int = None) -> int:
        """MACs for one inference, which is the number the FPGA budget uses."""
        n_bins = n_bins or C.N_ORDER_BINS
        macs, length = 0, n_bins
        for stage in self.features:
            conv = stage.conv
            length = (length + 2 * conv.padding[0] - conv.kernel_size[0]) \
                // conv.stride[0] + 1
            macs += conv.in_channels * conv.out_channels * conv.kernel_size[0] * length
        macs += self.classifier.in_features * self.classifier.out_features
        return macs


# ===========================================================================
# BatchNorm folding -- the bridge from the trained graph to the deployed one
# ===========================================================================

def fold_batchnorm(model: OrderSpectrumCNN) -> OrderSpectrumCNN:
    """
    Return a copy whose convolutions absorb their BatchNorm, with the BN layers
    replaced by identities.

    For y = gamma * (conv(x) - mu) / sqrt(var + eps) + beta, letting
    s = gamma / sqrt(var + eps):

        W' = W * s[:, None, None]
        b' = (b - mu) * s + beta

    This must happen before quantisation, not after: folding changes the weight
    distribution per output channel by a large factor, and calibrating scales on
    the unfolded weights would waste most of the INT8 range.
    """
    folded = copy.deepcopy(model).eval()
    for stage in folded.features:
        conv, bn = stage.conv, stage.bn
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        with torch.no_grad():
            conv.weight.mul_(s.reshape(-1, 1, 1))
            bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
            conv.bias = nn.Parameter((bias - bn.running_mean) * s + bn.bias)
        stage.bn = nn.Identity()
    return folded


def build_model(**kwargs) -> OrderSpectrumCNN:
    return OrderSpectrumCNN(**kwargs)


def describe(model: OrderSpectrumCNN) -> str:
    lines = [
        f"OrderSpectrumCNN  {model.parameter_count():,} params  "
        f"{model.mac_count():,} MAC/inference",
    ]
    length = C.N_ORDER_BINS
    for i, stage in enumerate(model.features):
        c = stage.conv
        length = (length + 2 * c.padding[0] - c.kernel_size[0]) // c.stride[0] + 1
        lines.append(
            f"  conv{i + 1}  {c.in_channels:3d} -> {c.out_channels:3d}  "
            f"k={c.kernel_size[0]} s={c.stride[0]}  out_len={length}"
        )
    lines.append(f"  gap    {model.classifier.in_features}")
    lines.append(f"  fc     {model.classifier.in_features} -> "
                 f"{model.classifier.out_features}")
    return "\n".join(lines)
