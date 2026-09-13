# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host half of the ``TT_FUSED`` serving path (torch only, no ttnn import). DEFAULT ON since the
2026-09-13 device validation (DEVICE_VALIDATION.md "Results"); ``TT_FUSED=0`` restores the legacy path.

The device half lives in ``models/tt/superpoint_ttnn.py`` (``TtSuperPoint.build_fused_graph``,
``capture_trace``, ``run_fused``). Everything here is either plumbing for the knob or a
*torch emulation of the exact device op sequence* so that the reformulations can be proven
against the legacy host math without hardware (``models/tests/test_fused_host.py``).

Knob (read ONCE at model build / server start, never per request):

    TT_FUSED              unset / empty / 1 (default) = the fused path: one metal trace over the
                          whole device graph (persistent input, resident outputs, ``execute_trace``
                          per request) plus the stages below. 0/false/no/off = the legacy untraced
                          path, untouched (the pre-2026-09-13 shipped behaviour).
    TT_FUSED_STAGES       comma list, default ``wide,nms,rms,rm`` (all). For device A/B only:
                          ``""`` = trace-only (legacy graph, traced); ``wide`` = 64-byte-page
                          input upload + in-trace reshape; ``nms`` = the device NMS-T chain;
                          ``rms`` = descriptor L2-norm as ``ttnn.rms_norm``; ``rm`` = untilize
                          the descriptor output on device.
    SP_TRACE_REGION       trace_region_size bytes for ``ttnn.CreateDevice`` (default 32 MiB).

Exactness classes (see reports/megakernel/superpoint-p150.md §4):
  * trace, wide, nms, rm: bit-identical to the legacy path on the same bf16 values;
  * rms: bf16-rounding-level (one final rounding instead of three), gated on device by
    descriptor PCC >= 0.999 and keypoint F1 >= 98.8 %.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

FUSED_ENV = "TT_FUSED"
STAGES_ENV = "TT_FUSED_STAGES"
TRACE_REGION_ENV = "SP_TRACE_REGION"

#: Default of the TT_FUSED knob when unset/empty (flipped to True after the p150a validation).
FUSED_DEFAULT = True
#: Every fused stage, in the A/B order of DEVICE_VALIDATION.md. Unknown names are an error.
ALL_STAGES: Tuple[str, ...] = ("wide", "nms", "rms", "rm")
_TRUE = ("1", "true", "yes", "on")

#: Wide-page upload: the 614 400 input bytes as [1, 1, N/32, 32] (64-byte RM pages) instead of
#: [1, 1, N, 1] (2-byte pages). The trace reshapes back to [1, 1, N, 1] before the first conv.
WIDE_PAGE_W = 32
#: NMS-T lane width: the dense 480x640 map is pooled as NHWC with C = 32 consecutive pixels of the
#: non-pooled axis, so each 1-D max-pool runs on 64-byte sticks (no 2-byte pages, no 32x padding).
NMS_LANES = 32
#: rms_norm gamma: x / ||x||_2 = x / (sqrt(256) * sqrt(mean(x^2))) = rms_norm(x) * (1/16). 1/16 is
#: a power of two, hence exact in bf16 (asserted in the host test).
RMS_GAMMA = 1.0 / 16.0
DESCRIPTOR_DIM = 256
#: trace_region_size when TT_FUSED=1 (port used 6 MiB for ~100 launches; the fused graph adds ~30).
DEFAULT_TRACE_REGION = 32 * 1024 * 1024


# ----------------------------------------------------------------------------- knob plumbing


def fused_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """``TT_FUSED`` unset or empty -> ``FUSED_DEFAULT`` (True); otherwise truthy ("1", "true",
    "yes", "on"; case-insensitive) -> fused, anything else ("0", "false", "no", "off") -> legacy."""
    env = os.environ if env is None else env
    raw = str(env.get(FUSED_ENV, "")).strip().lower()
    if raw == "":
        return FUSED_DEFAULT
    return raw in _TRUE


def parse_stages(spec: Optional[Iterable[str] | str]) -> FrozenSet[str]:
    """Normalise a stage list; ``None`` -> all stages, ``""`` -> none. Unknown -> ValueError."""
    if spec is None:
        return frozenset(ALL_STAGES)
    if isinstance(spec, str):
        items = [s.strip().lower() for s in spec.split(",")]
    else:
        items = [str(s).strip().lower() for s in spec]
    items = [s for s in items if s]
    unknown = sorted(set(items) - set(ALL_STAGES))
    if unknown:
        raise ValueError(f"{STAGES_ENV}: unknown stage(s) {unknown}; valid: {list(ALL_STAGES)}")
    return frozenset(items)


def fused_stages(env: Optional[Mapping[str, str]] = None) -> FrozenSet[str]:
    """Stages selected by ``TT_FUSED_STAGES`` (default: all). Independent of ``TT_FUSED`` itself
    so an explicit ``TtSuperPoint(..., fused=True)`` gets the full set."""
    env = os.environ if env is None else env
    return parse_stages(env.get(STAGES_ENV))


def trace_region_size(env: Optional[Mapping[str, str]] = None) -> int:
    env = os.environ if env is None else env
    raw = str(env.get(TRACE_REGION_ENV, "")).strip()
    return int(raw) if raw else DEFAULT_TRACE_REGION


def device_open_kwargs(
    device_id: int, l1_small_size: int, fused: bool, trace_region: Optional[int] = None
) -> Dict[str, int]:
    """kwargs for ``ttnn.CreateDevice``. Legacy: exactly ``device_id`` + ``l1_small_size`` (the
    shipped call). Fused: adds ``trace_region_size`` (default 0 in ttnn -> no capture possible)."""
    kwargs = {"device_id": int(device_id), "l1_small_size": int(l1_small_size)}
    if fused:
        kwargs["trace_region_size"] = int(DEFAULT_TRACE_REGION if trace_region is None else trace_region)
    return kwargs


# ----------------------------------------------------------------------------- wide-page upload


def wide_input_shape(batch_size: int, height: int, width: int) -> Tuple[int, int, int, int]:
    n = batch_size * height * width
    if n % WIDE_PAGE_W:
        raise ValueError(f"input volume {n} is not a multiple of {WIDE_PAGE_W}")
    return (1, 1, n // WIDE_PAGE_W, WIDE_PAGE_W)


def wide_input_view(nhwc_bf16: torch.Tensor) -> torch.Tensor:
    """[1, 1, N, 1] bf16 (the legacy host input) -> [1, 1, N/32, 32] view of the SAME bytes."""
    if nhwc_bf16.dim() != 4 or nhwc_bf16.shape[-1] != 1:
        raise ValueError(f"expected a [1, 1, N, 1] tensor, got {tuple(nhwc_bf16.shape)}")
    n = nhwc_bf16.numel()
    if n % WIDE_PAGE_W:
        raise ValueError(f"input volume {n} is not a multiple of {WIDE_PAGE_W}")
    return nhwc_bf16.contiguous().view(1, 1, n // WIDE_PAGE_W, WIDE_PAGE_W)


# ----------------------------------------------------------------------------- NMS-T emulation


def _maxpool_rows(x_flat: torch.Tensor, n: int, h: int, w: int, c: int, radius: int) -> torch.Tensor:
    """Emulates ``ttnn.max_pool2d(x, batch_size=n, input_h=h, input_w=w, channels=c,
    kernel_size=[2r+1, 1], stride=[1, 1], padding=[r, 0])`` on an NHWC tensor flattened to
    [1, 1, n*h*w, c]: a 1-D max along H for every (n, w, c) lane, -inf padded (pool_utils.cpp:40,
    same as ``F.max_pool2d``). Max of bf16 values is exact, so fp32 is used for the pool itself."""
    dtype = x_flat.dtype
    x = x_flat.reshape(n, h, w, c).permute(0, 3, 1, 2).float()  # NCHW for torch
    y = F.max_pool2d(x, kernel_size=(2 * radius + 1, 1), stride=1, padding=(radius, 0))
    return y.permute(0, 2, 3, 1).to(dtype).reshape(1, 1, n * h * w, c)


def nms_t_reference(s_sm: torch.Tensor, b: int, enc_h: int, enc_w: int, radius: int) -> torch.Tensor:
    """Torch emulation of ``TtSuperPoint._device_nms_t`` -- the SAME op sequence, step for step.

    ``s_sm``: device-softmaxed scores, logical [1, 1, b*enc_h*enc_w, 65] (any dtype; bf16 on
    device). Returns the post-NMS dense map [b, 8*enc_h, 8*enc_w] in ``s_sm.dtype``. Must equal
    ``postprocess.fold_scores(scores_nchw, radius)`` bit for bit -- the host test proves it.
    """
    H, W = enc_h * 8, enc_w * 8
    C = NMS_LANES
    if W % C or H % C:
        raise ValueError(f"NMS-T needs H and W multiples of {C}, got {H}x{W}")
    x = s_sm.reshape(1, 1, b * enc_h * enc_w, s_sm.shape[-1])
    # 1-2  slice(..., [1,1,N,64]) on the TILE tensor (tile-aligned), to_layout(ROW_MAJOR)
    x = x[..., :64].contiguous()
    # 3-5  reshape [b*enc_h, enc_w, 8, 8] -> permute(0,2,1,3) (= transpose_hc, RM) -> reshape
    #      [1, 1, b*H*(W/32), 32]: the dense map D[y, x] as NHWC (N=b, H=H, W=W/32, C=32)
    x = x.reshape(b * enc_h, enc_w, 8, 8).permute(0, 2, 1, 3).contiguous()
    D = x.reshape(1, 1, b * H * (W // C), C)
    # 6-7  max_pool2d k=[2r+1,1] p=[r,0] over H (the y direction) -> to DRAM
    My = _maxpool_rows(D, b, H, W // C, C, radius)
    # 8-11 reshape [b,1,H,W] -> tilize -> transpose(-2,-1) -> untilize -> reshape [1,1,b*W*(H/32),32]
    MyT = My.reshape(b, 1, H, W).transpose(-2, -1).contiguous().reshape(1, 1, b * W * (H // C), C)
    # 12-13 max_pool2d over H (now the x direction) -> to DRAM -> reshape [b,1,W,H] -> tilize
    WT = _maxpool_rows(MyT, b, W, H // C, C, radius).reshape(b, 1, W, H)  # = 9x9 window max, transposed
    # 14   D^T: reshape D [b,1,H,W] -> tilize -> transpose(-2,-1)
    DT = D.reshape(b, 1, H, W).transpose(-2, -1)
    # 15   eq(D^T, W^T) -> multiply(D^T, mask)   (x*1 = x, x*0 = +0 for x >= 0)
    nmsT = DT * (DT == WT).to(DT.dtype)
    # 16   transpose(-2,-1) -> untilize: natural [b, 1, H, W] ROW_MAJOR map, read back as-is
    return nmsT.transpose(-2, -1).contiguous().reshape(b, H, W)


# ----------------------------------------------------------------------------- rms_norm L2-norm


def l2norm_via_rms(x: torch.Tensor, gamma: float = RMS_GAMMA, eps: float = 0.0) -> torch.Tensor:
    """Emulates ``ttnn.rms_norm(x, epsilon=0, weight=gamma)`` with fp32 statistics and ONE final
    rounding to ``x.dtype`` -- the numerics the fused path asks for (HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True). For gamma = 1/sqrt(D) this is ``x / ||x||_2`` (``F.normalize``)."""
    xf = x.float()
    inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv_rms * gamma).to(x.dtype)


def l2norm_legacy_chain(x: torch.Tensor) -> torch.Tensor:
    """Emulates the legacy device chain with a bf16 rounding after every op: multiply(d, d) ->
    sum(-1) -> rsqrt -> multiply(d, inv). Only used to *report* the precision of the two
    formulations side by side in the host test; the device accumulates ``sum`` internally."""
    d_sq = (x.float() * x.float()).to(x.dtype)
    d_sum = d_sq.float().sum(dim=-1, keepdim=True).to(x.dtype)
    d_inv = torch.rsqrt(d_sum.float()).to(x.dtype)
    return (x.float() * d_inv.float()).to(x.dtype)


def bf16_ulp(v: torch.Tensor) -> torch.Tensor:
    """Spacing of bf16 values at |v| (8 significand bits): 2 ** (floor(log2|v|) - 7)."""
    a = v.detach().float().abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


# ----------------------------------------------------------------------------- result container


@dataclass
class FusedResult:
    """What ``TtSuperPoint.run_fused`` hands to the host post-processing.

    Exactly one of ``nms_map`` / ``scores_nchw`` is set: ``nms_map`` [B, H, W] fp32 (the device
    NMS map for the traced ``nms_radius``; feed ``postprocess.postprocess_from_nms_map``) or
    ``scores_nchw`` [B, 65, h, w] fp32 (device-softmaxed cell scores for any other radius; feed
    ``postprocess.postprocess_keypoints`` as on the legacy path). ``descriptors_nchw`` is the
    device-L2-normalised [B, 256, h, w] fp32 map in both cases.
    """

    descriptors_nchw: torch.Tensor
    nms_map: Optional[torch.Tensor] = None
    scores_nchw: Optional[torch.Tensor] = None
    nms_radius: int = 0

    @property
    def device_nms(self) -> bool:
        return self.nms_map is not None
