# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import SuperPointForKeypointDetection, AutoImageProcessor


MODEL_ID = "magic-leap-community/superpoint"
# Resolved relative to the repo root (one level above `models/`).
DEFAULT_NATURAL_IMAGE = Path(__file__).resolve().parents[2] / "sample_data" / "house_in_field_1080p.jpg"


def _weights_pointer(model_id, revision):
    """Repo id + revision: explicit args, else HF_MODEL / TT_WEIGHTS_REVISION, else MODEL_ID@main."""
    model_id = model_id or os.environ.get("HF_MODEL") or MODEL_ID
    revision = revision or os.environ.get("TT_WEIGHTS_REVISION") or None
    return model_id, revision


def load_reference_model(model_id: str | None = None, revision: str | None = None):
    model_id, revision = _weights_pointer(model_id, revision)
    model = SuperPointForKeypointDetection.from_pretrained(model_id, revision=revision)
    model.eval()
    return model


def load_image_processor(model_id: str | None = None, revision: str | None = None):
    model_id, revision = _weights_pointer(model_id, revision)
    return AutoImageProcessor.from_pretrained(model_id, revision=revision)


def get_dummy_input(batch_size: int = 1, height: int = 480, width: int = 640):
    torch.manual_seed(0)
    return torch.rand(batch_size, 3, height, width)


def get_natural_input(
    path: Path = DEFAULT_NATURAL_IMAGE,
    batch_size: int = 1,
    height: int = 480,
    width: int = 640,
) -> torch.Tensor:
    """Load and letterbox a natural image into a (B, 3, H, W) tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    transform = T.Compose([T.Resize((height, width)), T.ToTensor()])
    t = transform(img)  # (3, H, W) in [0, 1]
    return t.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
