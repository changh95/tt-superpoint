# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TT-NN SuperPoint port.

On-device: all convolutions + ReLU + MaxPool + softmax + simple_nms + L2-normalize.
Host: threshold / border-remove / top-k / grid_sample (variable-shape post-processing).

Two device paths share the layers:

* **legacy** (``TT_FUSED=0``): ``run_untraced`` -- H2D into the persistent
  ``[1, 1, H*W, 1]`` input, ``run_device_compute`` dispatched op by op (~100 launches),
  D2H of the TILE ``s_sm`` / ``d_norm``, host fold + 9x9 NMS. Unchanged.
* **fused** (default; ``TT_FUSED`` unset or 1, read once at build; device-validated
  2026-09-13 -- DEVICE_VALIDATION.md "Results"): ``build_fused_graph`` is the whole device
  graph -- in-trace reshape of a 64-byte-page ``[1, 1, H*W/32, 32]`` upload, encoder + heads,
  ``rms_norm`` descriptor L2-norm, the standard-op NMS-T chain (fold -> separable [9,1]
  max-pools on a 32-lane layout -> eq*mul), row-major outputs -- captured ONCE into a metal
  trace (``capture_trace``) and replayed per request (``run_fused``: one ``execute_trace``
  instead of ~100 host dispatches). ``TT_FUSED_STAGES`` isolates the stages for the device
  A/B (see ``models/tt/fused_host.py`` and DEVICE_VALIDATION.md).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import ttnn

from models.tt import fused_host as _fused
from models.tt import postprocess as _post

ENCODER_OUT_CHANNELS = (64, 64, 128, 128)
INTEREST_HIDDEN = 256
KEYPOINT_DIM = 65
DESCRIPTOR_HIDDEN = 256
DESCRIPTOR_DIM = 256


def _to_device_weight(weight: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(weight, dtype=dtype)


def _to_device_bias(bias: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    # Conv bias must be shape [1, 1, 1, out_channels].
    bias = bias.reshape(1, 1, 1, -1)
    return ttnn.from_torch(bias, dtype=dtype)


class TtConv2D:
    """Thin wrapper around ttnn.conv2d with weight/bias caching."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        device,
        activation: str | None = None,
        weights_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        num_slices: int = 1,
        act_block_h_override: int | None = None,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en: bool = False,
    ) -> None:
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (1, 1)
        self.padding = (padding, padding)
        self.dilation = (1, 1)
        self.groups = 1

        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32)
        self.bias = ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=ttnn.float32)

        act = None
        if activation == "relu":
            act = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            activation=act,
            shard_layout=shard_layout,
            deallocate_activation=False,
            output_layout=ttnn.TILE_LAYOUT,
        )
        if act_block_h_override is not None:
            self.conv_config.act_block_h_override = act_block_h_override
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )
        self.activation_dtype = activation_dtype
        if num_slices > 1:
            self.slice_config = ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dDRAMSliceHeight,
                num_slices=num_slices,
            )
        else:
            self.slice_config = ttnn.Conv2dL1FullSliceConfig

    def __call__(self, x, input_height: int, input_width: int, batch_size: int = 1):
        # DRAM-sliced conv requires the input tensor to live in DRAM.
        if self.slice_config is not ttnn.Conv2dL1FullSliceConfig:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x, [out_h, out_w], [self.weight, self.bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            conv_config=self.conv_config,
            slice_config=self.slice_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.activation_dtype,
        )
        return x, out_h, out_w


def device_outputs_to_host(s_sm, d_norm, b: int, h: int, w: int):
    """Device outputs of ``run_device_compute`` -> host NCHW fp32 tensors.

    ``s_sm`` is the device-softmaxed score tensor (TILE, logical
    ``(1, 1, b*h/8*w/8, 65)``) and ``d_norm`` the device-L2-normalised descriptor
    map (``(1, 1, b*h/8*w/8, 256)``). Returns ``(scores_nchw (b, 65, h/8, w/8),
    descriptors_nchw (b, 256, h/8, w/8))``. NO softmax is applied here -- it
    already ran on device. Identical to ``_device_to_host_post`` in
    ``models/tests/test_superpoint.py`` (the PCC/F1-validated harness).
    """
    enc_h, enc_w = h // 8, w // 8
    scores_nhwc = ttnn.to_torch(s_sm).reshape(b, enc_h, enc_w, KEYPOINT_DIM)
    descriptors_nhwc = ttnn.to_torch(d_norm).reshape(b, enc_h, enc_w, DESCRIPTOR_DIM)
    scores_nchw = scores_nhwc.permute(0, 3, 1, 2).contiguous().float()
    descriptors_nchw = descriptors_nhwc.permute(0, 3, 1, 2).contiguous().float()
    return scores_nchw, descriptors_nchw


class FusedDeviceOutputs:
    """Resident device outputs of ``TtSuperPoint.build_fused_graph`` (the trace's outputs).

    ``s_sm``: softmaxed scores, TILE, DRAM interleaved, logical ``[1, 1, B*h*w, 65]`` (kept for
    the host-NMS fallback when a request's ``nms_radius`` differs from the traced one).
    ``nms_map``: post-NMS dense map ``[B, 1, H, W]`` ROW_MAJOR bf16, or ``None`` when the ``nms``
    stage is off. ``d_out``: L2-normalised descriptors ``[1, 1, B*h*w, 256]``, ROW_MAJOR when the
    ``rm`` stage is on (D2H is then a plain copy), TILE otherwise.
    """

    def __init__(self, s_sm, nms_map, d_out):
        self.s_sm = s_sm
        self.nms_map = nms_map
        self.d_out = d_out

    def deallocate(self) -> None:
        for t in (self.s_sm, self.nms_map, self.d_out):
            if t is not None:
                ttnn.deallocate(t)
        self.s_sm = self.nms_map = self.d_out = None


class TtSuperPoint:
    """On-device SuperPoint inference (encoder + score/descriptor heads).

    Post-processing (NMS output extraction + grid_sample) is performed on host
    because the number of keypoints is data-dependent.
    """

    def __init__(
        self,
        torch_model,
        device,
        input_height: int = 480,
        input_width: int = 640,
        fused: bool | None = None,
        fused_stages=None,
    ):
        self.device = device
        self.input_height = input_height
        self.input_width = input_width
        self.nms_radius = torch_model.config.nms_radius
        self.keypoint_threshold = torch_model.config.keypoint_threshold
        self.max_keypoints = torch_model.config.max_keypoints
        self.border_removal_distance = torch_model.config.border_removal_distance

        # ---- TT_FUSED knob, read ONCE here (``fused=None`` -> env; default fused). 0 = legacy. ----
        self.fused = _fused.fused_enabled() if fused is None else bool(fused)
        if not self.fused:
            self.fused_stages = frozenset()
        elif fused_stages is None:
            self.fused_stages = _fused.fused_stages()
        else:
            self.fused_stages = _fused.parse_stages(fused_stages)
        #: NMS radius baked into the fused graph / trace (requests with another radius fall
        #: back to the host NMS from the traced ``s_sm``, same output, slower).
        self.nms_radius_traced = int(self.nms_radius)
        self._trace_id = None
        self._trace_outputs: FusedDeviceOutputs | None = None
        self._trace_batch = 0
        self._l2_gamma = None
        self._l2_compute_config = None
        if self.fused and "rms" in self.fused_stages:
            # x / ||x||_2 == rms_norm(x, eps=0) * (1/16) for D=256; 1/16 is exact in bf16.
            # TILE gamma [1,1,1,256] pads to [1,1,32,256]: layernorm validate wants padded H ==
            # tile height and padded W == the input's padded W (256). Created BEFORE any trace
            # capture (a host write). The default rms_norm compute config is HiFi4 with
            # math_approx_mode=True and no fp32 accumulate (rmsnorm.cpp) -> pass an explicit one.
            gamma = torch.full((1, 1, 1, DESCRIPTOR_DIM), _fused.RMS_GAMMA, dtype=torch.bfloat16)
            self._l2_gamma = ttnn.from_torch(
                gamma,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._l2_compute_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

        encoder = torch_model.encoder
        enc_hidden = torch_model.config.encoder_hidden_sizes

        # Encoder: 4 conv blocks of (conv_a -> relu -> conv_b -> relu [-> pool])
        # Per-block DRAM slicing to keep L1 circular buffers within budget at 480×640.
        # Block resolutions (H x W) with batch=1, image 480x640:
        #   block 0: 480x640, block 1: 240x320, block 2: 120x160, block 3: 60x80
        slice_per_block = (4, 2, 1, 1)
        # Encoder precision: HiFi2 + bfloat16 weights to keep PCC ≥ 99% through
        # 8 conv layers feeding softmax/L2-norm heads. bfloat8 + LoFi drops score
        # PCC to ~0.70.
        enc_kwargs = dict(
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
        )
        in_ch = 1
        self.enc_convs = []
        for block_idx, block in enumerate(encoder.conv_blocks):
            add_pooling = block.pool is not None
            ns = slice_per_block[block_idx]
            self.enc_convs.append(
                (
                    TtConv2D(
                        block.conv_a.weight,
                        block.conv_a.bias,
                        in_channels=in_ch,
                        out_channels=enc_hidden[block_idx],
                        kernel_size=3,
                        padding=1,
                        device=device,
                        activation="relu",
                        num_slices=ns,
                        **enc_kwargs,
                    ),
                    TtConv2D(
                        block.conv_b.weight,
                        block.conv_b.bias,
                        in_channels=enc_hidden[block_idx],
                        out_channels=enc_hidden[block_idx],
                        kernel_size=3,
                        padding=1,
                        device=device,
                        activation="relu",
                        num_slices=ns,
                        **enc_kwargs,
                    ),
                    add_pooling,
                )
            )
            in_ch = enc_hidden[block_idx]

        # Head configs: keep precision higher on heads (softmax + L2-norm are
        # numerically sensitive vs. the encoder ReLU chain).
        head_kwargs = dict(
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
        )

        # Score decoder
        kp = torch_model.keypoint_decoder
        self.conv_score_a = TtConv2D(
            kp.conv_score_a.weight,
            kp.conv_score_a.bias,
            in_channels=enc_hidden[-1],
            out_channels=torch_model.config.decoder_hidden_size,
            kernel_size=3,
            padding=1,
            device=device,
            activation="relu",
            **head_kwargs,
        )
        self.conv_score_b = TtConv2D(
            kp.conv_score_b.weight,
            kp.conv_score_b.bias,
            in_channels=torch_model.config.decoder_hidden_size,
            out_channels=torch_model.config.keypoint_decoder_dim,
            kernel_size=1,
            padding=0,
            device=device,
            activation=None,
            **head_kwargs,
        )

        # Descriptor decoder
        desc = torch_model.descriptor_decoder
        self.conv_desc_a = TtConv2D(
            desc.conv_descriptor_a.weight,
            desc.conv_descriptor_a.bias,
            in_channels=enc_hidden[-1],
            out_channels=torch_model.config.decoder_hidden_size,
            kernel_size=3,
            padding=1,
            device=device,
            activation="relu",
            **head_kwargs,
        )
        self.conv_desc_b = TtConv2D(
            desc.conv_descriptor_b.weight,
            desc.conv_descriptor_b.bias,
            in_channels=torch_model.config.decoder_hidden_size,
            out_channels=torch_model.config.descriptor_decoder_dim,
            kernel_size=1,
            padding=0,
            device=device,
            activation=None,
            **head_kwargs,
        )

    @staticmethod
    def _preprocess_host(pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract one channel & convert to NHWC layout for tt-nn conv input."""
        # HF model: (B, 3, H, W) -> first channel only -> (B, 1, H, W)
        one_ch = pixel_values[:, 0:1, :, :]
        # tt-nn conv expects NHWC flattened: [1, 1, B*H*W, C]
        b, c, h, w = one_ch.shape
        nhwc = one_ch.permute(0, 2, 3, 1).reshape(1, 1, b * h * w, c)
        return nhwc

    def _encoder_forward(self, tt_input, h: int, w: int, batch_size: int):
        x = tt_input
        cur_h, cur_w = h, w
        for conv_a, conv_b, add_pooling in self.enc_convs:
            x, cur_h, cur_w = conv_a(x, cur_h, cur_w, batch_size)
            x, cur_h, cur_w = conv_b(x, cur_h, cur_w, batch_size)
            if add_pooling:
                channels = x.shape[-1]
                x = ttnn.max_pool2d(
                    input_tensor=x,
                    batch_size=batch_size,
                    input_h=cur_h,
                    input_w=cur_w,
                    channels=channels,
                    kernel_size=[2, 2],
                    stride=[2, 2],
                    padding=[0, 0],
                    dilation=[1, 1],
                )
                cur_h //= 2
                cur_w //= 2
        return x, cur_h, cur_w

    def allocate_input(self, batch_size: int = 1) -> ttnn.Tensor:
        """Allocate a persistent DRAM tensor matching the expected input shape.

        Legacy: ``[1, 1, B*H*W, 1]`` ROW_MAJOR bf16 (2-byte pages, 307 200 of them). Fused
        ``wide`` stage: the same bytes as ``[1, 1, B*H*W/32, 32]`` (64-byte pages); the fused
        graph reshapes it back to ``[1, 1, B*H*W, 1]`` on device before the first conv (exact).
        """
        if self.fused and "wide" in self.fused_stages:
            shape = _fused.wide_input_shape(batch_size, self.input_height, self.input_width)
        else:
            shape = (1, 1, batch_size * self.input_height * self.input_width, 1)
        dummy = torch.zeros(shape, dtype=torch.float32)
        return ttnn.from_torch(
            dummy,
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def prepare_host_input(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """Host-side preprocess: extract first channel, NHWC-flatten, convert to
        bf16, and build a host ttnn tensor once. Reuse the return value across
        iterations — then ``load_input_prepared`` is pure H2D DMA (~0.1 ms).

        Without this, ``ttnn.from_torch(fp32 -> bf16)`` inside the hot loop
        costs ~10 ms/iter (dominant at trace speeds of ~3 ms forward).
        """
        nhwc = self._preprocess_host(pixel_values)
        # Pre-cast to bf16 on the CPU side — the expensive part.
        nhwc_bf16 = nhwc.to(torch.bfloat16).contiguous()
        if self.fused and "wide" in self.fused_stages:
            nhwc_bf16 = _fused.wide_input_view(nhwc_bf16)  # same bytes, 64-byte pages
        return ttnn.from_torch(nhwc_bf16, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def load_input_prepared(
        self,
        tt_in: ttnn.Tensor,
        host_tensor: ttnn.Tensor,
        cq_id: int = 0,
    ) -> None:
        """Issue the H2D copy of a pre-built host tensor into the device slot."""
        ttnn.copy_host_to_device_tensor(host_tensor, tt_in, cq_id=cq_id)

    def load_input(
        self,
        tt_in: ttnn.Tensor,
        pixel_values: torch.Tensor,
        cq_id: int = 0,
    ) -> None:
        """One-shot convenience: preprocess + build host tensor + H2D.

        Use ``prepare_host_input`` + ``load_input_prepared`` in hot loops —
        that split moves the fp32→bf16 cast out of the per-iteration cost.
        """
        host = self.prepare_host_input(pixel_values)
        ttnn.copy_host_to_device_tensor(host, tt_in, cq_id=cq_id)

    def run_device_compute(self, tt_in: ttnn.Tensor, b: int = 1, trace_nms: bool | None = None):
        """Device-only forward; returns (s_softmax, d_norm) in TILE layout.

        Softmax over the 65-dim channel axis is run on device.

        ``trace_nms`` selects the on-device fold + 9×9 max-pool NMS path, which
        returns a 3-tuple (s_softmax, s_pooled, d_norm) and needs the fused
        ``ttnn.experimental.sp_eq_mul_mask`` kernel (see kernels/). ``None``
        (the benchmark's behaviour) falls back to the SP_TRACE_NMS env var;
        callers with a fixed contract (the server) pass an explicit bool so a
        leaked env var cannot change the return arity. Pins intermediates to
        DRAM with explicit ``memory_config`` to avoid ttnn's implicit sharded
        reshape path.
        """
        if trace_nms is None:
            trace_nms = os.environ.get("SP_TRACE_NMS", "0") == "1"

        h, w = self.input_height, self.input_width
        # No-op for the legacy [1,1,N,1] input (``x is tt_in``, no device op); a fused wide-page
        # input is reshaped on device into a [1,1,N,1] copy that must be freed once the encoder
        # has consumed it (same as build_fused_graph) -- ``tt_in`` itself stays persistent.
        x, made_copy = self._input_as_conv_layout(tt_in, b)
        encoded, enc_h, enc_w = self._encoder_forward(x, h, w, b)
        if made_copy:
            ttnn.deallocate(x)

        s, _, _ = self.conv_score_a(encoded, enc_h, enc_w, b)
        s, _, _ = self.conv_score_b(s, enc_h, enc_w, b)
        s_sm = ttnn.softmax(s, dim=-1)
        ttnn.deallocate(s)

        d, _, _ = self.conv_desc_a(encoded, enc_h, enc_w, b)
        d, _, _ = self.conv_desc_b(d, enc_h, enc_w, b)
        d_sq = ttnn.multiply(d, d)
        d_sum = ttnn.sum(d_sq, dim=-1, keepdim=True)
        ttnn.deallocate(d_sq)
        d_inv = ttnn.rsqrt(d_sum)
        ttnn.deallocate(d_sum)
        d_norm = ttnn.multiply(d, d_inv)
        ttnn.deallocate(d_inv)
        ttnn.deallocate(d)
        ttnn.deallocate(encoded)

        if trace_nms:
            s_pooled = self._device_fold_and_nms(s_sm, b, enc_h, enc_w)
            return s_sm, s_pooled, d_norm
        return s_sm, d_norm

    def run_untraced(self, tt_in: ttnn.Tensor, pixel_values: torch.Tensor):
        """One untraced forward with host NMS: H2D -> device compute -> D2H.

        ``tt_in`` is the persistent device input from :meth:`allocate_input`
        (batch 1); ``pixel_values`` is fp32 (1, 3, input_height, input_width)
        in [0, 1]. Returns ``(scores_nchw (1, 65, h, w), descriptors_nchw
        (1, 256, h, w))`` host fp32 tensors -- the device-softmaxed scores and
        the device-L2-normalised descriptor map, i.e. exactly what
        ``tests/test_superpoint.py::_device_to_host_post`` hands to the
        validated post-processing (``models.tt.postprocess``). This is the
        serving path: pure ttnn (no custom kernel), no trace, host NMS.
        """
        b, _, h, w = pixel_values.shape
        if (h, w) != (self.input_height, self.input_width):
            raise ValueError(
                f"pixel_values is {h}x{w} but the device input is fixed at "
                f"{self.input_height}x{self.input_width}; resize on the host first"
            )
        self.load_input(tt_in, pixel_values)
        s_sm, d_norm = self.run_device_compute(tt_in, b=b, trace_nms=False)
        try:
            return device_outputs_to_host(s_sm, d_norm, b, h, w)
        finally:
            ttnn.deallocate(s_sm)
            ttnn.deallocate(d_norm)

    # ------------------------------------------------------------------ TT_FUSED path
    # Everything below is only reached when ``self.fused`` is True (TT_FUSED unset/1 or an explicit
    # ``fused=True``), except ``_input_as_conv_layout`` which is a pure shape check for the
    # legacy [1,1,N,1] input.

    @property
    def trace_id(self):
        return self._trace_id

    @staticmethod
    def _to_dram(t: ttnn.Tensor) -> ttnn.Tensor:
        """DRAM-interleaved copy of ``t`` (exact); deallocates the source when a copy was made,
        returns ``t`` itself when it already is DRAM interleaved (so the caller never frees it)."""
        mc = t.memory_config()
        if (not t.is_sharded()) and mc.buffer_type == ttnn.BufferType.DRAM:
            return t
        out = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(t)
        return out

    def _input_as_conv_layout(self, tt_in: ttnn.Tensor, b: int):
        """``[1, 1, N/32, 32]`` wide-page input -> ``[1, 1, N, 1]`` conv input (one in-trace
        ROW_MAJOR reshape, byte-identical data). Returns ``(tensor, made_copy)``; a legacy
        ``[1, 1, N, 1]`` input is returned unchanged with ``made_copy=False`` (no device op)."""
        n = b * self.input_height * self.input_width
        if tt_in.shape[-1] == 1:
            return tt_in, False
        x = ttnn.reshape(tt_in, [1, 1, n, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x, True

    def _pool_rows(self, x: ttnn.Tensor, n: int, h: int, w: int, c: int, radius: int) -> ttnn.Tensor:
        """1-D max over H of an NHWC tensor flattened to ``[1, 1, n*h*w, c]`` (kernel
        ``[2r+1, 1]``, stride 1, pad ``[r, 0]``; pool pads with -inf like ``F.max_pool2d``).
        ``max_pool2d`` auto-shards the interleaved input (reshard + halo + pool = 3 launches) and
        returns an L1 height-sharded ROW_MAJOR tensor; it is moved back to DRAM interleaved.
        Does NOT deallocate ``x``."""
        k = 2 * radius + 1
        y = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=n,
            input_h=h,
            input_w=w,
            channels=c,
            kernel_size=[k, 1],
            stride=[1, 1],
            padding=[radius, 0],
            dilation=[1, 1],
        )
        return self._to_dram(y)

    def _device_nms_t(self, s_sm: ttnn.Tensor, b: int, enc_h: int, enc_w: int) -> ttnn.Tensor:
        """NMS-T: fold + single-pass 9x9 NMS with ~28 standard ops (no custom kernel).

        ``s_sm``: softmaxed scores, TILE, DRAM interleaved, ``[1, 1, b*enc_h*enc_w, 65]``.
        Returns the post-NMS dense map ``[b, 1, 8*enc_h, 8*enc_w]`` ROW_MAJOR bf16 -- bit-identical
        to ``postprocess.fold_scores(scores_nchw, self.nms_radius_traced)`` on the same values
        (max/eq/x*1/x*0 are exact in bf16; scores >= 0; every window contains its centre).
        Torch emulation of this exact sequence: ``fused_host.nms_t_reference`` (host-tested).

        The 2-D max is separable: max_x(max_y(.)). Each 1-D pool runs on a layout whose 32
        channels are 32 consecutive pixels of the *other* axis (64-byte sticks) -- no 2-byte
        pages and none of the 32x zero-padding of the port's ``_device_fold_and_nms``. Step
        numbers match ``nms_t_reference`` and reports/megakernel/superpoint-p150.md §4C.
        """
        r = self.nms_radius_traced
        H, W = enc_h * 8, enc_w * 8
        C = _fused.NMS_LANES
        n = b * enc_h * enc_w
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT

        # 1-2  drop the dustbin channel (tile-aligned slice: n % 32 == 0, 64 % 32 == 0), untilize
        s64 = ttnn.slice(s_sm, [0, 0, 0, 0], [1, 1, n, 64], memory_config=DRAM)
        s64_rm = ttnn.to_layout(s64, RM, memory_config=DRAM)
        ttnn.deallocate(s64)
        # 3-5  fold: [b*enc_h, enc_w, 8, 8] (cy, cx, i, j) -> permute(0,2,1,3) = transpose_hc on
        #      ROW_MAJOR -> (cy, i, cx, j) -> D[y, x] laid out as NHWC (N=b, H=H, W=W/32, C=32)
        s4 = ttnn.reshape(s64_rm, [b * enc_h, enc_w, 8, 8], memory_config=DRAM)
        ttnn.deallocate(s64_rm)
        s4p = ttnn.permute(s4, (0, 2, 1, 3), memory_config=DRAM)
        ttnn.deallocate(s4)
        D = ttnn.reshape(s4p, [1, 1, b * H * (W // C), C], memory_config=DRAM)
        ttnn.deallocate(s4p)
        # 6-7  y-direction max: pool over H for every (x lane); back to DRAM
        My = self._pool_rows(D, b, H, W // C, C, r)
        # 8-11 to x-major: [b,1,H,W] -> tilize -> transpose(-2,-1) -> untilize -> [1,1,b*W*(H/32),32]
        My4 = ttnn.reshape(My, [b, 1, H, W], memory_config=DRAM)
        ttnn.deallocate(My)
        My_t = ttnn.to_layout(My4, TILE, memory_config=DRAM)
        ttnn.deallocate(My4)
        MyT = ttnn.transpose(My_t, -2, -1, memory_config=DRAM)
        ttnn.deallocate(My_t)
        MyT_rm = ttnn.to_layout(MyT, RM, memory_config=DRAM)
        ttnn.deallocate(MyT)
        MyT_l = ttnn.reshape(MyT_rm, [1, 1, b * W * (H // C), C], memory_config=DRAM)
        ttnn.deallocate(MyT_rm)
        # 12-13 x-direction max (= the 9x9 window max, transposed) -> [b,1,W,H] TILE
        WT = self._pool_rows(MyT_l, b, W, H // C, C, r)
        ttnn.deallocate(MyT_l)
        WT4 = ttnn.reshape(WT, [b, 1, W, H], memory_config=DRAM)
        ttnn.deallocate(WT)
        WT_t = ttnn.to_layout(WT4, TILE, memory_config=DRAM)
        ttnn.deallocate(WT4)
        # 14   D^T in the same layout
        D4 = ttnn.reshape(D, [b, 1, H, W], memory_config=DRAM)
        ttnn.deallocate(D)
        D_t = ttnn.to_layout(D4, TILE, memory_config=DRAM)
        ttnn.deallocate(D4)
        DT = ttnn.transpose(D_t, -2, -1, memory_config=DRAM)
        ttnn.deallocate(D_t)
        # 15   keep the pixels that equal their window max (eq -> 1.0/0.0; multiply is exact)
        mask = ttnn.eq(DT, WT_t, memory_config=DRAM)
        ttnn.deallocate(WT_t)
        nmsT = ttnn.multiply(DT, mask, memory_config=DRAM)
        ttnn.deallocate(mask)
        ttnn.deallocate(DT)
        # 16   natural orientation, ROW_MAJOR so the D2H is a plain copy: [b, 1, H, W]
        nms_t = ttnn.transpose(nmsT, -2, -1, memory_config=DRAM)
        ttnn.deallocate(nmsT)
        nms_rm = ttnn.to_layout(nms_t, RM, memory_config=DRAM)
        ttnn.deallocate(nms_t)
        return nms_rm

    def build_fused_graph(self, tt_in: ttnn.Tensor, b: int = 1) -> FusedDeviceOutputs:
        """The whole TT_FUSED device graph. Run it once eagerly (compiles every kernel and
        prepares the conv weights -- host writes that must happen OUTSIDE a capture), then again
        inside ``capture_trace``. No host writes in here: the only inputs are the persistent
        ``tt_in`` and the device-resident weights/gamma; intermediates are deallocated as they die.

        Stages (``self.fused_stages``): ``wide`` in-trace input reshape, ``rms`` rms_norm L2-norm
        (else the legacy multiply/sum/rsqrt/multiply chain), ``rm`` row-major descriptor output,
        ``nms`` NMS-T map. With all stages off this is exactly the legacy graph, traced.
        """
        if not self.fused:
            raise RuntimeError("build_fused_graph needs the fused path (TT_FUSED unset/1 or TtSuperPoint(..., fused=True))")
        st = self.fused_stages
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        h, w = self.input_height, self.input_width

        x, made_copy = self._input_as_conv_layout(tt_in, b)
        encoded, enc_h, enc_w = self._encoder_forward(x, h, w, b)
        if made_copy:
            ttnn.deallocate(x)

        # Descriptor head first: its L1 shards are gone before the NMS chain's ROW_MAJOR reshapes
        # stage through L1 (reshape_rm budgets its staging ring against live L1 tensors).
        d, _, _ = self.conv_desc_a(encoded, enc_h, enc_w, b)
        d, _, _ = self.conv_desc_b(d, enc_h, enc_w, b)
        if "rms" in st:
            # rms_norm rejects HEIGHT_SHARDED inputs -> DRAM interleaved first (exact copy).
            d_dram = self._to_dram(d)
            d_norm = ttnn.rms_norm(
                d_dram,
                epsilon=0.0,
                weight=self._l2_gamma,
                compute_kernel_config=self._l2_compute_config,
                memory_config=DRAM,
            )
            ttnn.deallocate(d_dram)
        else:
            # Legacy chain, verbatim (see run_device_compute).
            d_sq = ttnn.multiply(d, d)
            d_sum = ttnn.sum(d_sq, dim=-1, keepdim=True)
            ttnn.deallocate(d_sq)
            d_inv = ttnn.rsqrt(d_sum)
            ttnn.deallocate(d_sum)
            d_norm = ttnn.multiply(d, d_inv)
            ttnn.deallocate(d_inv)
            ttnn.deallocate(d)
        if "rm" in st:
            d_norm = self._to_dram(d_norm)
            d_out = ttnn.to_layout(d_norm, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
            ttnn.deallocate(d_norm)
        else:
            d_out = d_norm

        s, _, _ = self.conv_score_a(encoded, enc_h, enc_w, b)
        ttnn.deallocate(encoded)
        s, _, _ = self.conv_score_b(s, enc_h, enc_w, b)
        s_sm = ttnn.softmax(s, dim=-1)
        ttnn.deallocate(s)
        # The resident score output lives in DRAM (frees the L1 shards for the NMS chain).
        s_sm = self._to_dram(s_sm)

        nms_map = self._device_nms_t(s_sm, b, enc_h, enc_w) if "nms" in st else None
        return FusedDeviceOutputs(s_sm, nms_map, d_out)

    def capture_trace(self, tt_in: ttnn.Tensor, b: int = 1, cq_id: int = 0):
        """Capture ``build_fused_graph`` into a metal trace; the outputs stay resident and are
        re-read after every ``execute_trace``. Call after ONE eager ``build_fused_graph`` /
        ``run_fused`` (kernel compile + conv weight preparation happen there). The device must
        have been opened with ``trace_region_size > 0``."""
        if self._trace_id is not None:
            raise RuntimeError("trace already captured; call release() first")
        tid = ttnn.begin_trace_capture(self.device, cq_id=cq_id)
        outs = self.build_fused_graph(tt_in, b)
        ttnn.end_trace_capture(self.device, tid, cq_id=cq_id)
        ttnn.synchronize_device(self.device)
        self._trace_id, self._trace_outputs, self._trace_batch = tid, outs, b
        return tid

    def _read_fused(self, outs: FusedDeviceOutputs, b: int, h: int, w: int, nms_radius: int):
        enc_h, enc_w = h // 8, w // 8
        desc_nchw = (
            ttnn.to_torch(outs.d_out).reshape(b, enc_h, enc_w, DESCRIPTOR_DIM).permute(0, 3, 1, 2).contiguous().float()
        )
        if outs.nms_map is not None and nms_radius == self.nms_radius_traced:
            nms_map = ttnn.to_torch(outs.nms_map).reshape(b, h, w).float()
            return _fused.FusedResult(descriptors_nchw=desc_nchw, nms_map=nms_map, nms_radius=nms_radius)
        scores_nchw = (
            ttnn.to_torch(outs.s_sm).reshape(b, enc_h, enc_w, KEYPOINT_DIM).permute(0, 3, 1, 2).contiguous().float()
        )
        return _fused.FusedResult(descriptors_nchw=desc_nchw, scores_nchw=scores_nchw, nms_radius=nms_radius)

    def run_fused(self, tt_in: ttnn.Tensor, pixel_values: torch.Tensor, *, nms_radius: int | None = None):
        """One TT_FUSED forward: H2D -> ``execute_trace`` (or the eager graph before a capture)
        -> D2H. Returns ``fused_host.FusedResult``: the device NMS map when the ``nms`` stage is on
        and ``nms_radius`` equals the traced radius (default), otherwise the softmaxed scores for
        the legacy host post-processing; plus the descriptor map. Same contract as
        ``run_untraced`` for ``tt_in`` (from ``allocate_input``) and ``pixel_values``."""
        if not self.fused:
            raise RuntimeError("run_fused needs the fused path (TT_FUSED unset/1 or TtSuperPoint(..., fused=True))")
        b, _, h, w = pixel_values.shape
        if (h, w) != (self.input_height, self.input_width):
            raise ValueError(
                f"pixel_values is {h}x{w} but the device input is fixed at "
                f"{self.input_height}x{self.input_width}; resize on the host first"
            )
        radius = self.nms_radius_traced if nms_radius is None else int(nms_radius)
        self.load_input_prepared(tt_in, self.prepare_host_input(pixel_values))
        if self._trace_id is not None:
            if b != self._trace_batch:
                raise ValueError(f"trace was captured for batch {self._trace_batch}, got {b}")
            # Non-blocking: the readbacks below are queued behind the trace on the same CQ.
            ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
            return self._read_fused(self._trace_outputs, b, h, w, radius)
        outs = self.build_fused_graph(tt_in, b)
        try:
            return self._read_fused(outs, b, h, w, radius)
        finally:
            outs.deallocate()

    def release(self) -> None:
        """Release the trace and every device tensor the fused path keeps resident."""
        if self._trace_id is not None:
            ttnn.release_trace(self.device, self._trace_id)
            self._trace_id = None
        if self._trace_outputs is not None:
            self._trace_outputs.deallocate()
            self._trace_outputs = None
        if self._l2_gamma is not None:
            ttnn.deallocate(self._l2_gamma)
            self._l2_gamma = None

    def _get_nms_zero_pad(self, b: int) -> ttnn.Tensor:
        """Persistent zero-padding tensor reused across trace replays.

        ``ttnn.zeros`` performs a device write which trace capture rejects,
        so we materialise this exactly once during warmup and keep the
        handle alive for the trace to reference.
        """
        H, W = self.input_height, self.input_width
        cached = getattr(self, "_zeros_pad_cache", None)
        if cached is not None:
            return cached
        pad = ttnn.zeros(
            [1, 1, b * H * W, 31],
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._zeros_pad_cache = pad
        return pad

    def _device_fold_and_nms(self, s_sm: ttnn.Tensor, b: int, enc_h: int, enc_w: int):
        """Fold softmax (b,1,enc_h*enc_w,65) to dense (b,H,W,1) and run 9×9
        max-pool NMS, all on device. Keeps every reshape DRAM-interleaved to
        avoid ttnn's sharded page-alignment check.
        """
        H, W = enc_h * 8, enc_w * 8
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        s64 = ttnn.slice(s_sm, [0, 0, 0, 0], [b, 1, enc_h * enc_w, 64], memory_config=DRAM)
        s64_rm = ttnn.to_layout(s64, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(s64)
        s_nhwc = ttnn.reshape(s64_rm, [b, enc_h, enc_w, 64], memory_config=DRAM)
        ttnn.deallocate(s64_rm)
        s5d = ttnn.reshape(s_nhwc, [b, enc_h, enc_w, 8, 8], memory_config=DRAM)
        ttnn.deallocate(s_nhwc)
        s_perm = ttnn.permute(s5d, (0, 1, 3, 2, 4), memory_config=DRAM)
        ttnn.deallocate(s5d)
        s_dense = ttnn.reshape(s_perm, [b, H, W, 1], memory_config=DRAM)
        ttnn.deallocate(s_perm)
        s_flat = ttnn.reshape(s_dense, [1, 1, b * H * W, 1], memory_config=DRAM)
        ttnn.deallocate(s_dense)
        zeros_pad = self._get_nms_zero_pad(b)  # persistent, trace-compatible
        s_padded = ttnn.concat([s_flat, zeros_pad], dim=-1, memory_config=DRAM)
        ttnn.deallocate(s_flat)
        s_pooled = ttnn.max_pool2d(
            input_tensor=s_padded,
            batch_size=b,
            input_h=H,
            input_w=W,
            channels=32,
            kernel_size=[self.nms_radius * 2 + 1, self.nms_radius * 2 + 1],
            stride=[1, 1],
            padding=[self.nms_radius, self.nms_radius],
            dilation=[1, 1],
        )
        # Close the NMS loop on device with the fused C++ kernel:
        # output[i] = s_padded[i] if s_padded[i] == s_pooled[i] else 0.
        # max_pool2d returns a height-sharded tensor, so re-interleave both
        # operands into DRAM before TILE conversion (sharded->tile is rejected
        # unless every shard is tile-aligned).
        s_pooled_dram = ttnn.to_memory_config(s_pooled, DRAM)
        ttnn.deallocate(s_pooled)
        s_padded_tile = ttnn.to_layout(s_padded, ttnn.TILE_LAYOUT, memory_config=DRAM)
        s_pooled_tile = ttnn.to_layout(s_pooled_dram, ttnn.TILE_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(s_padded)
        ttnn.deallocate(s_pooled_dram)
        s_nms = ttnn.experimental.sp_eq_mul_mask(s_padded_tile, s_pooled_tile)
        ttnn.deallocate(s_padded_tile)
        ttnn.deallocate(s_pooled_tile)
        # Channels 1-31 of s_nms are exact zeros (s_padded was zero-padded
        # there), so only c0 carries information. Slicing to a 1-channel
        # row-major output shrinks the D2H payload by 32× and removes the
        # host tile-unpack cost.
        s_nms_rm = ttnn.to_layout(s_nms, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.deallocate(s_nms)
        s_nms_c0 = ttnn.slice(
            s_nms_rm, [0, 0, 0, 0], [1, 1, b * H * W, 1], memory_config=DRAM
        )
        ttnn.deallocate(s_nms_rm)
        return s_nms_c0

    def _run_device(self, pixel_values: torch.Tensor):
        """Untraced path: allocate input, compute, read back, host post-proc."""
        b, _, h, w = pixel_values.shape
        tt_in = self.allocate_input(batch_size=b)
        self.load_input(tt_in, pixel_values)

        s, d_norm = self.run_device_compute(tt_in, b, trace_nms=False)
        # Softmax already ran on device (ttnn.softmax in run_device_compute);
        # applying torch.softmax again here collapsed every cell towards 1/65
        # and defeated the keypoint threshold.
        scores_nchw, descriptors_nchw = device_outputs_to_host(s, d_norm, b, h, w)

        ttnn.deallocate(s)
        ttnn.deallocate(d_norm)
        ttnn.deallocate(tt_in)

        return scores_nchw, descriptors_nchw

    # --- Host-side post-processing (data-dependent; keypoint count varies) ---

    # The bodies live in models/tt/postprocess.py (torch only) so the server
    # can run them with per-request parameters and without importing ttnn.

    @staticmethod
    def _simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
        """Single-pass NMS (see ``models.tt.postprocess.simple_nms``)."""
        return _post.simple_nms(scores, nms_radius)

    def _decode_keypoints(self, scores_nchw: torch.Tensor, apply_nms: bool = True):
        # Drop the dustbin (last channel), fold 8x8 -> full-res, optional NMS.
        return _post.fold_scores(scores_nchw, self.nms_radius if apply_nms else None)

    def _extract_keypoints_single(self, scores_1hw: torch.Tensor):
        return _post.extract_keypoints(
            scores_1hw, self.keypoint_threshold, self.border_removal_distance, self.max_keypoints
        )

    @staticmethod
    def _sample_descriptors(keypoints, descriptors, scale: int = 8):
        return _post.sample_descriptors(keypoints, descriptors, scale)

    def forward(self, pixel_values: torch.Tensor):
        scores_nchw, descriptors_nchw = self._run_device(pixel_values)
        scores_pre_nms = self._decode_keypoints(scores_nchw, apply_nms=False)
        scores_full = self._simple_nms(scores_pre_nms, self.nms_radius)

        b, _, h, w = pixel_values.shape
        list_keypoints, list_scores, list_descriptors = [], [], []
        for i in range(b):
            kp, sc = self._extract_keypoints_single(scores_full[i : i + 1])
            list_keypoints.append(kp)
            list_scores.append(sc)
            d = self._sample_descriptors(kp[None], descriptors_nchw[i : i + 1], scale=8)[0]
            list_descriptors.append(d.transpose(0, 1))

        max_kp = max(k.shape[0] for k in list_keypoints)
        keypoints_t = torch.zeros((b, max_kp, 2))
        scores_t = torch.zeros((b, max_kp))
        descriptors_t = torch.zeros((b, max_kp, DESCRIPTOR_DIM))
        mask_t = torch.zeros((b, max_kp), dtype=torch.int)
        for i, (kp, sc, dc) in enumerate(zip(list_keypoints, list_scores, list_descriptors)):
            keypoints_t[i, : kp.shape[0]] = kp
            scores_t[i, : sc.shape[0]] = sc
            descriptors_t[i, : dc.shape[0]] = dc
            mask_t[i, : sc.shape[0]] = 1
        keypoints_t = keypoints_t / torch.tensor([w, h])
        return {
            "keypoints": keypoints_t,
            "scores": scores_t,
            "descriptors": descriptors_t,
            "mask": mask_t,
            "raw_scores_map": scores_full,
            "raw_scores_pre_nms": scores_pre_nms,
            "raw_descriptors_map": descriptors_nchw,
        }
