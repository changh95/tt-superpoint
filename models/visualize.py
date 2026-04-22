# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Render SuperPoint keypoints produced by the tt-nn model onto the sample image.

Usage (run from repo root):
    TT_METAL_DIR=/path/to/tt-metal DEVICE_ID=3 \
    PYTHONPATH=.:$TT_METAL_DIR:$TT_METAL_DIR/ttnn \
    TT_METAL_HOME=$TT_METAL_DIR ARCH_NAME=blackhole \
    python models/visualize.py

Writes media/sample.png: the resized input with the top-500 keypoints overlaid
as cyan circles sized by relative score.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import ttnn

from models.reference.superpoint_reference import (
    DEFAULT_NATURAL_IMAGE,
    get_natural_input,
    load_reference_model,
)
from models.tt.superpoint_ttnn import TtSuperPoint


HEIGHT, WIDTH = 480, 640
TOP_K = 500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_NATURAL_IMAGE)
    parser.add_argument("--out", type=Path, default=Path("media/sample.png"))
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("DEVICE_ID", "0")))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    device = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=32 * 1024)

    torch_model = load_reference_model()
    pixel_values = get_natural_input(path=args.image, batch_size=1, height=HEIGHT, width=WIDTH)

    tt_model = TtSuperPoint(torch_model, device, input_height=HEIGHT, input_width=WIDTH)
    tt_in = tt_model.allocate_input(batch_size=1)
    tt_model.load_input(tt_in, pixel_values)
    s_sm, d_norm = tt_model.run_device_compute(tt_in, b=1)
    ttnn.synchronize_device(device)

    # Device softmax output + host fold + NMS.
    enc_h, enc_w = HEIGHT // 8, WIDTH // 8
    scores_nhwc = ttnn.to_torch(s_sm).reshape(1, enc_h, enc_w, 65)
    scores_nchw = scores_nhwc.permute(0, 3, 1, 2).contiguous().float()
    scores_pre = tt_model._decode_keypoints(scores_nchw, apply_nms=False)
    scores_nms = tt_model._simple_nms(scores_pre, tt_model.nms_radius)

    flat = scores_nms[0].flatten()
    k_eff = min(args.top_k, flat.numel())
    score_values, idx = torch.topk(flat, k_eff)
    ys = (idx // WIDTH).tolist()
    xs = (idx %  WIDTH).tolist()
    scores = score_values.tolist()

    # Draw on the original pixel values (resized 480×640 RGB in [0, 1]).
    img_t = pixel_values[0].permute(1, 2, 0).clamp(0, 1)
    img_np = (img_t.numpy() * 255.0).astype(np.uint8)
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil, "RGBA")

    if scores:
        s_min, s_max = min(scores), max(scores)
        span = max(s_max - s_min, 1e-9)
        for x, y, s in zip(xs, ys, scores):
            r = 2 + int(3 * (s - s_min) / span)  # 2..5 px radius
            draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 255, 255, 230), width=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(args.out)
    print(f"wrote {args.out} ({len(xs)} keypoints, top-K={args.top_k})")

    ttnn.deallocate(s_sm)
    ttnn.deallocate(d_norm)
    ttnn.deallocate(tt_in)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
