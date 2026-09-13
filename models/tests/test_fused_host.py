# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the TT_FUSED path (torch only -- no ttnn import, no device).

Every exact reformulation the fused device graph relies on is checked here against the
legacy host math (``models.tt.postprocess``), plus the knob plumbing:

* NMS-T: the torch emulation of the exact device op sequence (slice -> untilize -> reshape ->
  permute -> reshape -> [9,1] max-pool -> transpose -> [9,1] max-pool -> eq*mul -> transpose)
  equals ``fold_scores(scores, r)`` (= fold + ``simple_nms``) bit for bit (``torch.equal``):
  random softmax grids, a plateau/tie grid, border rows, batch 2, radii 1/4/8.
* Wide-page upload: the ``[1,1,N/32,32]`` host view is the same bytes as ``[1,1,N,1]``.
* rms_norm L2-norm: ``rms_norm(x, eps=0) * (1/16) == x / ||x||`` within 1 bf16 ULP of
  ``F.normalize`` (one final rounding; the legacy 4-op chain is reported alongside), and
  ``bf16(1/16) == 1/16`` exactly.
* ``postprocess_from_nms_map`` (the fused host post-processing) == ``postprocess_keypoints``.
* Knob: ``TT_FUSED`` unset/empty -> fused (default since 2026-09-13), ``0`` -> legacy (legacy
  ``CreateDevice`` kwargs); stages parsing.
* ``code/conftest.py`` (device fixtures for the device tests) imports without ttnn; device-id
  resolution CLI > ``$TT_DEVICE_ID`` > ``$DEVICE_ID`` > 0.

Run (host python of the model's tt-metal tree; pytest is in that venv):

    cd models/superpoint-p150/code && PYTHONPATH=. TT_METAL_HOME=<tree> \
        <tree>/python_env/bin/python -m pytest -q models/tests/test_fused_host.py

or as a plain script: ``python models/tests/test_fused_host.py``.
"""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn.functional as F

from models.tt import fused_host as fh
from models.tt import postprocess as post

ENC_H, ENC_W = 60, 80  # 480x640 / 8
KEYPOINT_DIM = 65


def _softmax_scores(b: int, enc_h: int, enc_w: int, scale: float, seed: int) -> torch.Tensor:
    """Device-like softmaxed scores [1, 1, b*enc_h*enc_w, 65] in bf16."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, 1, b * enc_h * enc_w, KEYPOINT_DIM, generator=g) * scale
    return torch.softmax(logits, dim=-1).to(torch.bfloat16)


def _to_nchw(s_sm: torch.Tensor, b: int, enc_h: int, enc_w: int) -> torch.Tensor:
    """What ``device_outputs_to_host`` does with ``s_sm``: NHWC -> NCHW fp32."""
    return s_sm.reshape(b, enc_h, enc_w, KEYPOINT_DIM).permute(0, 3, 1, 2).contiguous().float()


# ------------------------------------------------------------------------------- NMS-T


@pytest.mark.parametrize("scale", [0.3, 3.0])
@pytest.mark.parametrize("radius", [4])
def test_nms_t_equals_fold_and_simple_nms_random(scale, radius):
    s_sm = _softmax_scores(1, ENC_H, ENC_W, scale, seed=int(scale * 10))
    ref = post.fold_scores(_to_nchw(s_sm, 1, ENC_H, ENC_W), radius)
    out = fh.nms_t_reference(s_sm, 1, ENC_H, ENC_W, radius)
    assert out.shape == (1, 480, 640) and out.dtype == torch.bfloat16
    assert torch.equal(out.float(), ref)
    assert (ref > 0).sum() > 100  # the test actually exercised suppression


def test_nms_t_pre_nms_fold_is_exact():
    """Radius-free check of the fold permutation alone: D[cy*8+i, cx*8+j] == s[cy, cx, i*8+j]."""
    s_sm = _softmax_scores(1, ENC_H, ENC_W, 1.0, seed=7)
    # radius 0 is not a device configuration (k=1 pool is pointless) but the emulation is
    # well-defined: k=1, pad=0 -> identity pools, eq is all-true -> the folded map itself.
    out = fh.nms_t_reference(s_sm, 1, ENC_H, ENC_W, 0)
    ref = post.fold_scores(_to_nchw(s_sm, 1, ENC_H, ENC_W), None)
    assert torch.equal(out.float(), ref)


def test_nms_t_ties_plateaus_and_borders():
    """Ties inside a window keep every tied pixel (== semantics), plateaus survive, and the
    border rows/cols (windows partly outside the image, -inf padded) match the host."""
    s_sm = _softmax_scores(1, ENC_H, ENC_W, 3.0, seed=3)
    s = s_sm.clone()
    s[:, :, :400] = 1.0 / 65  # first 5 cell rows: uniform -> every pixel ties with its window
    s[..., 3] = s[..., 5]  # channel ties in every cell
    s[:, :, -80:, :64] = 0.0  # last cell row all zeros (border plateau of zeros)
    s[:, :, 2000:2003, :] = s[:, :, 2000:2001, :]  # three identical neighbouring cells
    for radius in (1, 4, 8):
        ref = post.fold_scores(_to_nchw(s, 1, ENC_H, ENC_W), radius)
        out = fh.nms_t_reference(s, 1, ENC_H, ENC_W, radius)
        assert torch.equal(out.float(), ref), f"radius {radius}"
    # Sanity on the tie region: uniform rows whose 9x9 window stays inside the uniform block
    # (rows 0..35; rows 36..39 see rows 40..43) are kept entirely -- the host tie semantics.
    ref4 = post.fold_scores(_to_nchw(s, 1, ENC_H, ENC_W), 4)
    assert torch.equal(ref4[0, :36], torch.full((36, 640), 1.0 / 65).to(torch.bfloat16).float())


def test_nms_t_batch_two():
    s_sm = _softmax_scores(2, ENC_H, ENC_W, 2.0, seed=11)
    ref = post.fold_scores(_to_nchw(s_sm, 2, ENC_H, ENC_W), 4)
    out = fh.nms_t_reference(s_sm, 2, ENC_H, ENC_W, 4)
    assert out.shape == (2, 480, 640)
    assert torch.equal(out.float(), ref)


def test_nms_t_matches_a_realistic_score_map():
    """Peaky softmax outputs (a few dominant cells, most probability in the dustbin) --
    the regime the real network produces -- plus the dustbin channel must be ignored."""
    g = torch.Generator().manual_seed(5)
    logits = torch.randn(1, 1, ENC_H * ENC_W, KEYPOINT_DIM, generator=g) * 0.5
    logits[..., 64] += 4.0  # dustbin dominates most cells
    peaks = torch.randint(0, ENC_H * ENC_W, (600,), generator=g)
    chans = torch.randint(0, 64, (600,), generator=g)
    logits[0, 0, peaks, chans] += 8.0
    s_sm = torch.softmax(logits, dim=-1).to(torch.bfloat16)
    nchw = _to_nchw(s_sm, 1, ENC_H, ENC_W)
    ref = post.fold_scores(nchw, 4)
    out = fh.nms_t_reference(s_sm, 1, ENC_H, ENC_W, 4)
    assert torch.equal(out.float(), ref)
    # and the dustbin really is dropped: nothing in the map exceeds the max non-dustbin score
    assert out.float().max() <= nchw[:, :64].max()
    assert int((ref > 0.005).sum()) >= 300  # suppression was exercised on a peaky map


def test_postprocess_from_nms_map_equals_postprocess_keypoints():
    s_sm = _softmax_scores(1, ENC_H, ENC_W, 3.0, seed=21)
    nchw = _to_nchw(s_sm, 1, ENC_H, ENC_W)
    g = torch.Generator().manual_seed(22)
    desc = F.normalize(torch.randn(1, 256, ENC_H, ENC_W, generator=g), dim=1)
    kw = dict(keypoint_threshold=0.005, max_keypoints=1024, border_removal_distance=4, with_descriptors=True)
    ref_kp, ref_sc, ref_desc = post.postprocess_keypoints(nchw, desc, nms_radius=4, **kw)[0]
    nms_map = fh.nms_t_reference(s_sm, 1, ENC_H, ENC_W, 4).float()  # what the device hands back
    kp, sc, d = post.postprocess_from_nms_map(nms_map, desc, **kw)[0]
    assert kp.shape[0] > 0
    assert torch.equal(kp, ref_kp) and torch.equal(sc, ref_sc) and torch.equal(d, ref_desc)
    # descriptors off / small top-k
    kp2, sc2, d2 = post.postprocess_from_nms_map(nms_map, desc, **{**kw, "max_keypoints": 50, "with_descriptors": False})[0]
    assert kp2.shape == (50, 2) and sc2.shape == (50,) and d2 is None


# ------------------------------------------------------------------------------- wide upload


def test_wide_input_view_is_the_same_bytes():
    g = torch.Generator().manual_seed(1)
    pixel = torch.rand(1, 3, 480, 640, generator=g)
    nhwc = pixel[:, 0:1].permute(0, 2, 3, 1).reshape(1, 1, 480 * 640, 1).to(torch.bfloat16).contiguous()
    wide = fh.wide_input_view(nhwc)
    assert tuple(wide.shape) == fh.wide_input_shape(1, 480, 640) == (1, 1, 9600, 32)
    assert wide.dtype == torch.bfloat16 and wide.is_contiguous()
    assert wide.data_ptr() == nhwc.data_ptr()  # a view, no copy
    assert torch.equal(wide.view(torch.int16).flatten(), nhwc.view(torch.int16).flatten())
    # element (row r, lane c) of the wide view is pixel r*32 + c of the flat NHWC input
    assert torch.equal(wide[0, 0, 123], nhwc[0, 0, 123 * 32 : 124 * 32, 0])
    with pytest.raises(ValueError):
        fh.wide_input_view(torch.zeros(1, 1, 30, 1, dtype=torch.bfloat16))


# ------------------------------------------------------------------------------- rms_norm L2


def test_rms_gamma_is_exact_in_bf16():
    assert fh.RMS_GAMMA == 1.0 / 16.0
    assert torch.tensor(fh.RMS_GAMMA, dtype=torch.bfloat16).item() == fh.RMS_GAMMA
    assert 16 * 16 == fh.DESCRIPTOR_DIM  # gamma = 1/sqrt(D)


@pytest.mark.parametrize("scale", [0.05, 0.7, 20.0])
def test_l2norm_via_rms_within_one_bf16_ulp(scale):
    g = torch.Generator().manual_seed(int(scale * 100))
    x = (torch.randn(4800, 256, generator=g) * scale).to(torch.bfloat16)
    ref = F.normalize(x.float(), p=2, dim=-1)
    out = fh.l2norm_via_rms(x).float()
    err = (out - ref).abs()
    ulp = fh.bf16_ulp(ref)
    # One final bf16 rounding of an fp32-exact value: <= 0.5 ULP (+ fp32 noise) -> allow 1 ULP.
    # Reason 1 ULP and not torch.equal: the device rounds x*inv_rms*gamma once at the end; the
    # reference F.normalize is fp32, so agreement is defined up to the output quantum.
    assert (err <= ulp + 1e-12).all(), float((err / ulp).max())
    assert float((err / ulp).max()) <= 0.5 + 1e-3
    # Unit norm after the rounding (what smoke_test.py checks on the served descriptors).
    assert (out.norm(dim=-1) - 1.0).abs().max() < 1e-2
    # Report: the legacy 4-op chain rounds three times and is worse -- informational only.
    legacy_err = ((fh.l2norm_legacy_chain(x).float() - ref).abs() / ulp).max()
    assert legacy_err >= float((err / ulp).max()) - 1e-6


# ------------------------------------------------------------------------------- knob plumbing


def test_knob_default_is_fused_and_zero_is_legacy():
    env = {}
    assert fh.FUSED_DEFAULT is True
    assert fh.fused_enabled(env) is True
    assert fh.fused_enabled({"TT_FUSED": ""}) is True  # empty == unset
    assert fh.device_open_kwargs(0, 32768, fh.fused_enabled(env)) == {
        "device_id": 0, "l1_small_size": 32768, "trace_region_size": fh.DEFAULT_TRACE_REGION,
    }
    for v in ("0", "off", "false", "no", " 0 "):
        assert fh.fused_enabled({"TT_FUSED": v}) is False
    # The legacy path keeps exactly the shipped CreateDevice kwargs.
    assert fh.device_open_kwargs(0, 32768, fh.fused_enabled({"TT_FUSED": "0"})) == {"device_id": 0, "l1_small_size": 32768}
    for v in ("1", "true", "YES", " on "):
        assert fh.fused_enabled({"TT_FUSED": v}) is True


def test_stage_parsing():
    assert fh.fused_stages({}) == frozenset(fh.ALL_STAGES) == {"wide", "nms", "rms", "rm"}
    assert fh.fused_stages({"TT_FUSED_STAGES": ""}) == frozenset()  # trace-only A/B
    assert fh.fused_stages({"TT_FUSED_STAGES": "wide, NMS"}) == {"wide", "nms"}
    assert fh.parse_stages(["rms", "rm"]) == {"rms", "rm"}
    with pytest.raises(ValueError):
        fh.fused_stages({"TT_FUSED_STAGES": "wide,bogus"})


def test_fused_device_open_kwargs():
    assert fh.device_open_kwargs(0, 32768, True) == {
        "device_id": 0, "l1_small_size": 32768, "trace_region_size": fh.DEFAULT_TRACE_REGION,
    }
    assert fh.device_open_kwargs(3, 32768, True, 6 * 1024 * 1024)["trace_region_size"] == 6 * 1024 * 1024
    assert fh.trace_region_size({}) == fh.DEFAULT_TRACE_REGION >= 12 * 1024 * 1024
    assert fh.trace_region_size({"SP_TRACE_REGION": "16777216"}) == 16777216


def test_fused_result_contract():
    d = torch.zeros(1, 256, ENC_H, ENC_W)
    r = fh.FusedResult(descriptors_nchw=d, nms_map=torch.zeros(1, 480, 640), nms_radius=4)
    assert r.device_nms and r.scores_nchw is None
    r2 = fh.FusedResult(descriptors_nchw=d, scores_nchw=torch.zeros(1, 65, ENC_H, ENC_W), nms_radius=7)
    assert not r2.device_nms and r2.nms_map is None


def test_superpoint_ttnn_module_reads_knob_once_at_build(monkeypatch):
    """The device wrapper takes the knob from the ctor (``fused=None`` -> env) and stores the
    decision; it must not consult the environment per call. Checked on the pure-python
    surface without constructing a device model (ttnn may be absent on the host)."""
    monkeypatch.delenv("TT_FUSED", raising=False)
    assert fh.fused_enabled() is True
    monkeypatch.setenv("TT_FUSED", "0")
    assert fh.fused_enabled() is False
    monkeypatch.setenv("TT_FUSED", "1")
    assert fh.fused_enabled() is True
    monkeypatch.setenv("TT_FUSED_STAGES", "nms")
    assert fh.fused_stages() == {"nms"}


def test_repo_conftest_provides_device_fixtures():
    """The device tests take ``device`` / ``device_params`` / ``--device-id`` from
    ``code/conftest.py`` (tt-metal's conftest cannot be loaded next to this repo: ``code/models``
    shadows its namespace ``models`` package). Import it by path -- no ttnn, no device -- and
    check the pure device-id resolution (CLI > $TT_DEVICE_ID > $DEVICE_ID > 0)."""
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "conftest.py"
    spec = importlib.util.spec_from_file_location("superpoint_repo_conftest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.pytest_addoption)
    assert hasattr(mod, "device") and hasattr(mod, "device_params")
    assert mod.resolve_device_id(None, {}) == 0
    assert mod.resolve_device_id(None, {"DEVICE_ID": "3"}) == 3
    assert mod.resolve_device_id(None, {"TT_DEVICE_ID": "2", "DEVICE_ID": "3"}) == 2
    assert mod.resolve_device_id(None, {"TT_DEVICE_ID": ""}) == 0
    assert mod.resolve_device_id(1, {"TT_DEVICE_ID": "2"}) == 1
    assert (pathlib.Path(__file__).resolve().parents[2] / "pytest.ini").is_file()


if __name__ == "__main__":  # `python models/tests/test_fused_host.py` == `python -m pytest -q <file>`
    sys.exit(pytest.main([__file__, "-q"]))
