"""Smoke test for the new ttnn.experimental.sp_eq_mul_mask fused kernel.

Verifies output matches torch reference:
    out = a if a == b else 0
on a tile-aligned bfloat16 pair.
"""

import torch
import ttnn


def main() -> None:
    device = ttnn.CreateDevice(device_id=3, l1_small_size=32 * 1024)
    try:
        torch.manual_seed(0)
        # Tile-aligned shape: (1, 1, 307200, 32) — matches SuperPoint NMS mask
        # target; pooled tensor is 32-ch padded, input map is 1-ch broadcast.
        H, W = 480, 640
        N = H * W

        # Build an A tensor and a B tensor where ~5% of positions match exactly.
        a_t = torch.randn(1, 1, N, 32, dtype=torch.float32)
        b_t = a_t.clone()
        # Perturb 95% of values in B so only ~5% match.
        keep_mask = torch.rand(1, 1, N, 32) < 0.05
        b_t = torch.where(keep_mask, a_t, a_t + 1.0)

        a_bf = a_t.to(torch.bfloat16).contiguous()
        b_bf = b_t.to(torch.bfloat16).contiguous()

        a_tt = ttnn.from_torch(
            a_bf, dtype=ttnn.bfloat16, device=device,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b_tt = ttnn.from_torch(
            b_bf, dtype=ttnn.bfloat16, device=device,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        out_tt = ttnn.experimental.sp_eq_mul_mask(a_tt, b_tt)
        out = ttnn.to_torch(out_tt).float()

        # Torch reference (match bf16 rounding on both sides).
        a_ref = a_bf.float()
        b_ref = b_bf.float()
        expected = torch.where(a_ref == b_ref, a_ref, torch.zeros_like(a_ref))

        diff = (out - expected).abs().max().item()
        n_matches_out = (out != 0).sum().item()
        n_matches_ref = (expected != 0).sum().item()

        print(f"out shape: {out.shape}")
        print(f"max abs diff vs reference: {diff:.6f}")
        print(f"nonzero(out)={n_matches_out}  nonzero(ref)={n_matches_ref}")
        if diff < 1e-3 and n_matches_out == n_matches_ref:
            print("PROBE_OK")
        else:
            print("PROBE_FAIL")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
