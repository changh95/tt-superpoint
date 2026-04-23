"""Accuracy + throughput benchmark for ttnn.experimental.sp_eq_mul_mask.

Compares the fused single-kernel dispatch against the 2-op composed path
(ttnn.eq + ttnn.multiply) and the torch reference.
"""

import time
import torch
import ttnn

DEVICE_ID = 3
H, W = 480, 640
BHW = H * W
C_PAD = 32     # tile-aligned channel count matching the NMS use-case
N_ITER = 100


def build_pair(match_rate: float):
    """Construct A, B where roughly `match_rate` fraction of positions match."""
    torch.manual_seed(0)
    a = torch.randn(1, 1, BHW, C_PAD, dtype=torch.float32)
    keep = torch.rand(1, 1, BHW, C_PAD) < match_rate
    b = torch.where(keep, a, a + 1.0)
    return a.to(torch.bfloat16).contiguous(), b.to(torch.bfloat16).contiguous()


def to_device(t, device):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, device=device,
        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def fused(a_tt, b_tt):
    return ttnn.experimental.sp_eq_mul_mask(a_tt, b_tt)


def composed(a_tt, b_tt):
    mask = ttnn.eq(a_tt, b_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.multiply(a_tt, mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(mask)
    return out


def accuracy_check(device, match_rate: float) -> None:
    a_bf, b_bf = build_pair(match_rate)
    a_tt = to_device(a_bf, device)
    b_tt = to_device(b_bf, device)
    out_fused = ttnn.to_torch(fused(a_tt, b_tt)).float()
    out_comp = ttnn.to_torch(composed(a_tt, b_tt)).float()
    ref = torch.where(a_bf.float() == b_bf.float(), a_bf.float(),
                      torch.zeros_like(a_bf.float()))
    fuse_diff = (out_fused - ref).abs().max().item()
    comp_diff = (out_comp - ref).abs().max().item()
    fuse_nz = (out_fused != 0).sum().item()
    comp_nz = (out_comp != 0).sum().item()
    ref_nz = (ref != 0).sum().item()
    match = "MATCH" if (fuse_diff == 0 and fuse_nz == ref_nz) else "MISMATCH"
    print(f"  match_rate={match_rate:<6}  fused_maxabs={fuse_diff:.6f} "
          f"composed_maxabs={comp_diff:.6f}  nz fused={fuse_nz} composed={comp_nz} ref={ref_nz}  {match}")
    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)


def throughput(device, fn, label: str) -> float:
    a_bf, b_bf = build_pair(0.05)
    a_tt = to_device(a_bf, device)
    b_tt = to_device(b_bf, device)
    # Warmup / JIT compile
    warm = fn(a_tt, b_tt)
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm)

    t0 = time.perf_counter()
    outs = []
    for _ in range(N_ITER):
        outs.append(fn(a_tt, b_tt))
    ttnn.synchronize_device(device)
    dt = (time.perf_counter() - t0) / N_ITER * 1000.0
    for o in outs:
        ttnn.deallocate(o)
    ttnn.deallocate(a_tt)
    ttnn.deallocate(b_tt)
    print(f"  {label:<12} {dt:.3f} ms/iter  ({1000.0/dt:.0f} calls/s)")
    return dt


def main() -> None:
    device = ttnn.CreateDevice(device_id=DEVICE_ID, l1_small_size=32 * 1024)
    try:
        print("=== Accuracy (fused vs torch reference vs composed eq+mul) ===")
        for rate in (0.0, 0.01, 0.05, 0.20, 1.0):
            accuracy_check(device, rate)

        print()
        print("=== Throughput (N=100 async dispatches, single sync) ===")
        dt_fused = throughput(device, fused, "fused")
        dt_comp = throughput(device, composed, "composed")
        print(f"  speedup fused vs composed: {dt_comp / dt_fused:.2f}x")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
