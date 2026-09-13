# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side SuperPoint post-processing (torch only, no ttnn import).

This is the sequence the port validated in ``models/tests/test_superpoint.py``
(``_device_to_host_post`` -> ``_decode_keypoints(apply_nms=True)`` ->
``_extract_keypoints_single`` -> ``_sample_descriptors``), lifted out of the
``TtSuperPoint`` methods so that

* the serving app can call it with per-request ``nms_radius`` /
  ``keypoint_threshold`` / ``max_keypoints`` instead of the values frozen into
  the model config, and
* it stays importable without ttnn (unit tests, tooling).

The device already applied the 65-way softmax (``ttnn.softmax`` in
``TtSuperPoint.run_device_compute``); nothing here applies it again. The input
``scores_nchw`` is the *softmaxed* score tensor exactly as
``superpoint_ttnn.device_outputs_to_host`` returns it.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

DESCRIPTOR_SCALE = 8  # encoder stride: one descriptor cell per 8x8 pixels


def simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """Single-pass NMS: keep pixels whose score equals the local (2r+1)^2 max.

    The HF reference iterates a tie-expansion loop three times (~100 ms/iter on
    host at 480x640). The single pass costs one max-pool and preserved
    keypoint F1 98.8% @ top-500 / 2 px in the port's benchmark.
    """
    if nms_radius <= 0:
        return scores
    pooled = F.max_pool2d(scores, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)
    return torch.where(scores == pooled, scores, torch.zeros_like(scores))


def fold_scores(scores_nchw: torch.Tensor, nms_radius: int | None) -> torch.Tensor:
    """(B, 65, h, w) softmaxed cell scores -> (B, 8h, 8w) dense map.

    Drops the dustbin channel (64) and unfolds each 8x8 cell. ``nms_radius``
    ``None`` skips NMS (pre-NMS map); an int applies :func:`simple_nms`.
    """
    scores = scores_nchw[:, :-1]  # (B, 64, h, w)
    b, _, fh, fw = scores.shape
    scores = scores.permute(0, 2, 3, 1).reshape(b, fh, fw, 8, 8)
    scores = scores.permute(0, 1, 3, 2, 4).reshape(b, fh * 8, fw * 8)
    if nms_radius is not None:
        scores = simple_nms(scores, nms_radius)
    return scores


def extract_keypoints(
    scores_1hw: torch.Tensor,
    keypoint_threshold: float,
    border_removal_distance: int,
    max_keypoints: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Threshold, border-remove and top-k one (1, H, W) post-NMS map.

    Returns ``keypoints`` as (N, 2) float ``(x, y)`` pixel coordinates in the
    map's frame and ``scores`` as (N,). ``max_keypoints < 0`` keeps every point
    above the threshold.
    """
    _, height, width = scores_1hw.shape
    keypoints = torch.nonzero(scores_1hw[0] > keypoint_threshold)
    scores = scores_1hw[0][tuple(keypoints.t())]
    border = border_removal_distance
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    keypoints = keypoints[mask]
    scores = scores[mask]
    if max_keypoints >= 0 and keypoints.shape[0] > max_keypoints:
        scores, idx = torch.topk(scores, max_keypoints, dim=0)
        keypoints = keypoints[idx]
    keypoints = torch.flip(keypoints, [1]).to(scores.dtype)  # (y, x) -> (x, y)
    return keypoints, scores


def sample_descriptors(
    keypoints: torch.Tensor, descriptors: torch.Tensor, scale: int = DESCRIPTOR_SCALE
) -> torch.Tensor:
    """Bilinear-sample the (B, C, h, w) descriptor map at (B, N, 2) ``(x, y)`` points.

    Returns (B, C, N), L2-normalised along C (the HF reference's
    ``_sample_descriptors``).
    """
    batch_size, num_channels, height, width = descriptors.shape
    keypoints = keypoints - scale / 2 + 0.5
    divisor = torch.tensor([[(width * scale - scale / 2 - 0.5), (height * scale - scale / 2 - 0.5)]])
    divisor = divisor.to(keypoints)
    keypoints = keypoints / divisor
    keypoints = keypoints * 2 - 1
    keypoints = keypoints.view(batch_size, 1, -1, 2)
    descriptors = F.grid_sample(descriptors, keypoints, mode="bilinear", align_corners=True)
    descriptors = descriptors.reshape(batch_size, num_channels, -1)
    descriptors = F.normalize(descriptors, p=2, dim=1)
    return descriptors


def postprocess_keypoints(
    scores_nchw: torch.Tensor,
    descriptors_nchw: torch.Tensor,
    *,
    nms_radius: int,
    keypoint_threshold: float,
    max_keypoints: int,
    border_removal_distance: int,
    with_descriptors: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    """Full validated host post-processing for a batch.

    ``scores_nchw``: (B, 65, h, w) device-softmaxed scores as returned by
    ``superpoint_ttnn.device_outputs_to_host``. ``descriptors_nchw``:
    (B, 256, h, w) device-L2-normalised descriptor map.

    Returns one ``(keypoints (N, 2) xy, scores (N,), descriptors (N, 256) | None)``
    triple per image, keypoints in the network-input pixel frame (480x640).
    """
    scores_full = fold_scores(scores_nchw, nms_radius)
    return postprocess_from_nms_map(
        scores_full,
        descriptors_nchw,
        keypoint_threshold=keypoint_threshold,
        max_keypoints=max_keypoints,
        border_removal_distance=border_removal_distance,
        with_descriptors=with_descriptors,
    )


def postprocess_from_nms_map(
    nms_map: torch.Tensor,
    descriptors_nchw: torch.Tensor,
    *,
    keypoint_threshold: float,
    max_keypoints: int,
    border_removal_distance: int,
    with_descriptors: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    """Post-processing from an already folded + NMS'd dense map (the ``TT_FUSED`` path).

    ``nms_map``: (B, H, W) post-NMS scores -- either ``fold_scores(scores_nchw, r)`` (host) or
    the device NMS-T map ``TtSuperPoint.run_fused`` returns (bit-identical to it). Everything
    after the NMS is the legacy code path: :func:`extract_keypoints` + :func:`sample_descriptors`.
    """
    out = []
    for i in range(nms_map.shape[0]):
        kp, sc = extract_keypoints(
            nms_map[i : i + 1], keypoint_threshold, border_removal_distance, max_keypoints
        )
        desc = None
        if with_descriptors:
            if kp.shape[0] > 0:
                desc = sample_descriptors(kp[None], descriptors_nchw[i : i + 1])[0].transpose(0, 1)
            else:
                desc = torch.zeros((0, descriptors_nchw.shape[1]), dtype=descriptors_nchw.dtype)
        out.append((kp, sc, desc))
    return out
