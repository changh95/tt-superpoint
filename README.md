# tt-superpoint

SuperPoint keypoint-detection inference on a single Tenstorrent Blackhole
(p150a/p150b) accelerator, implemented with tt-nn.

Reference: `magic-leap-community/superpoint` on Hugging Face.
Input: single-image, 480×640, batch size 1.

## Running

Prereqs:
- A built `tt-metal` checkout (the `ttnn` runtime is loaded from there).
- Python 3.12 venv with `torch`, `torchvision`, `transformers`, and
  `loguru` installed.
- One visible Blackhole chip.

```bash
TT_METAL_DIR=/absolute/path/to/tt-metal \
DEVICE_ID=0 \
SP_N_ITER=100 \
SP_TRACE_NMS=1 \
bash run_benchmark.sh
```

`SP_TRACE_NMS=1` closes the NMS loop on device via the fused
`ttnn.experimental.sp_eq_mul_mask` kernel — that's the fast e2e path
(**40.7 fps** on a natural image vs 17 fps with host NMS).  Omit the flag
to run the pure-forward trace with host NMS.

## Sample output

Top-500 tt-nn keypoints overlaid on the resized (480×640) sample image.
Circle radius is proportional to keypoint score.

![tt-nn SuperPoint keypoints on the sample image](media/sample.png)

Reproduce with:

```bash
TT_METAL_DIR=/absolute/path/to/tt-metal \
DEVICE_ID=0 \
PYTHONPATH=.:$TT_METAL_DIR:$TT_METAL_DIR/ttnn \
TT_METAL_HOME=$TT_METAL_DIR ARCH_NAME=blackhole \
python models/visualize.py
```

## Final results (Blackhole p150b, 480×640, batch 1, natural image)

### PCC vs Hugging Face reference (fp32 CPU torch)

| Tensor | PCC |
|---|---:|
| Pre-NMS score map | **0.9971** |
| Descriptor map (pre grid-sample, post L2-norm) | **0.9991** |

PCC ≥ 0.99 across both outputs — meets the project's hard accuracy floor.

### Keypoint-set evaluation (GT = Hugging Face reference on real image)

Real photograph (`sample_data/house_in_field_1080p.jpg`), top-K = 500,
matching radius = 2 pixels:

| Metric | tt-nn vs torch reference |
|---|---:|
| Recall | **98.80%** |
| Precision | **98.80%** |
| **F1** | **98.80%** |

For the synthetic `torch.rand` input (distribution the model was not
trained on), F1 is 97.80% — included as a stability check, not an
accuracy claim.

### Throughput

Two traced inference paths are available; pick via the `SP_TRACE_NMS` env var.

**Forward-only trace (`SP_TRACE_NMS=0`) — pure device compute, host-side NMS**

| Metric | Random input | Natural image | Paper (Titan X, 2018 Caffe) |
|---|---:|---:|---:|
| **Device forward (input pre-resident)** | **355.26 fps** | **355.37 fps (2.81 ms)** | 90 fps (11.15 ms) |
| Traced forward incl. per-frame H2D | 73.58 fps | 73.55 fps | — |
| `fps_match_paper` (forward + descriptor sampling) | 41.95 fps | 42.66 fps | 70 fps (13 ms) |
| Full e2e (incl. host NMS) | 17.50 fps | 17.00 fps | not reported |

**Forward + device NMS (`SP_TRACE_NMS=1`) — fused `sp_eq_mul_mask` closes the NMS loop on device**

| Metric | Random input | Natural image | Paper (Titan X, 2018 Caffe) |
|---|---:|---:|---:|
| Device forward + device NMS (pre-resident) | 85.59 fps | 85.60 fps (11.68 ms) | 90 fps (11.15 ms) |
| Traced forward+NMS incl. per-frame H2D | 44.55 fps | 44.56 fps | — |
| `fps_match_paper` (forward + descriptor sampling) | 28.46 fps | 29.18 fps | 70 fps (13 ms) |
| **Full e2e (no host NMS)** | **40.69 fps** | **40.73 fps** | not reported |

**E2E win**: moving NMS on-device via the fused `sp_eq_mul_mask` C++ kernel
plus a stack of dispatch/host-side cleanups pushes end-to-end throughput from
**17.0 → 40.73 fps** on the natural image (**+140%**) and
**17.5 → 40.69 fps** on random (**+132%**). PCC and F1 are preserved to the
last digit (0.9971 / 98.80% on natural).

The device-NMS trace absorbs the 36 ms host simple_nms into ~7 ms of extra
device-side work (fold + `max_pool2d` + `sp_eq_mul_mask` + row-major
channel-0 slice) — that's why `compute_only` and `match_paper` look lower in
the second table: the traced region now does strictly more work. Pure forward
fps is unchanged.

E2E per-phase breakdown (SP_TRACE_NMS=1, natural image, ms/iter):

| Phase | ms |
|---|---:|
| Compute-phase (Python dispatch + event records; trace runs async) | 8.4 |
| D2H (descriptor tile + single-channel NMS map, 2× `ttnn.to_torch`) | 14.3 |
| Host post (keypoint extraction + grid_sample) | 1.8 |

D2H is dominated by ttnn's per-call `from_device` dispatch cost (~6–9 ms per
call even on a 614 KB tensor); it's the same runtime floor that caps H2D at
~10.7 ms. Input double-buffering to hide H2D behind trace was tried twice
and regressed (see the Reverts table) — the fix would need a 1-channel NMS
kernel to cut layout conversions out of the trace, or a batched D2H API.

Measurement methodology: 10-iteration inner loop per metric, SP_N_ITER=100 for
stable numbers. Compute-only uses `blocking=False` + a single final sync;
traced forward adds per-frame `load_input_prepared` (H2D of the pre-cast bf16
host tensor) on cq_id=1, overlapping the trace on cq_id=0. E2E uses the same
dual-CQ pattern + a blocking D2H of the single-channel NMS output.

### Comparison read

- **Device forward pass hits 353 fps on a natural image — 3.9× the 2018 Titan X
  baseline.** This is the SRAM-effective number: the traced forward replay
  fits entirely in on-chip L1 with only block 0/1's activations spilling to
  DRAM (per-slice, bounded by the 1.5 MB/core ceiling). Weights stay
  resident across invocations because trace owns the allocator.
- `inference_speed` (per-frame H2D included) drops to ~72 fps because
  `ttnn.copy_host_to_device_tensor` carries a ~10.7 ms-per-call fixed Python
  dispatch cost independent of payload size. That's a ttnn runtime
  characteristic, not hardware — the PCIe 4.0×16 payload is 600 KB (~20 µs at
  line rate). Dual command queues hide compute behind H2D but not vice versa
  because H2D > compute.
- `fps_match_paper` (the metric that lines up with the paper's 13 ms figure)
  is at **60%** of paper when paying the per-frame H2D cost each call, and
  exceeds the paper's 70 fps on pure compute (353 fps).
- End-to-end incl. NMS is lower because the paper does not include NMS in
  its timing.

## Optimization trajectory

Recorded experiment-by-experiment in `results.tsv`. The commits cited
here are short hashes from the branch the work was developed on.

### Biggest wins (cumulative)

| # | Change | Before → After (fps_traced) | Notes |
|---|---|---:|---|
| 1 | Initial port (LoFi, bfloat8 weights) | 0 → **5.85** | Score PCC 0.70 — below 99% floor |
| 2 | HiFi2 + bfloat16 weights + fp32 accumulator (`b0ecf6a`) | 5.85 → **6.41** | PCC jumps to 0.997 — now meets spec |
| 3 | Descriptor L2-norm moved to device (`1c2a582`) | 6.41 → **6.59** | +2.9% |
| 4 | **`ttnn.trace` captures the device forward** (`5a705bd`) | 6.59 → **71.31** | **10.8×** — Python dispatch was 97% of wall-clock |
| 5 | 2 command queues (H2D on CQ1 overlapped with compute) (`787eff6`) | 71.31 → **72.04** | +1% |
| 6 | Single-pass NMS on host (replaces HF's 3-pass tie-expansion loop) | (e2e: 6.23 → **16.5 fps**) | Host NMS was 119 ms/iter; single pass ~36 ms; F1 98.8% preserved |
| 7 | Device softmax (verified `ttnn.softmax` respects 65-dim logical shape) (`7d1c378`) | 73.60 | accuracy-neutral; unblocks future on-device post-proc |
| 8 | **SRAM diagnostic + prebuild host bf16 input once** (`62f112d`) | 73.60 → **353.31** (compute-only) | Isolated ttnn's per-call Python H2D dispatch cost (~10.7 ms/call, payload-independent) from actual device compute (2.83 ms/iter) — hardware forward-pass fps is **3.9×** the paper on natural image |
| 9 | **Fused `sp_eq_mul_mask` closes NMS loop on device** | (e2e: 17.00 → **24.11 fps**) | Replaces the 36 ms host simple_nms with `ttnn.max_pool2d` + the fused C++ kernel (`ttnn.experimental.sp_eq_mul_mask`) + an on-device channel-0 slice. 7 ms of extra trace work saves 36 ms of host work. F1 unchanged at 98.80%; PCC unchanged at 0.9971. |
| 10 | **Drop redundant `synchronize_device` before D2H** | (e2e: 24.11 → **34.28 fps**, +40%) | The explicit full-device sync before the D2H phase was forcing CQ1's pipelined H2D to drain at the same time as CQ0's trace; the first `ttnn.to_torch` on CQ0 already blocks implicitly on trace completion, so the sync was pure serialization. One-line removal. |
| 11 | Drop redundant `.contiguous()` before `.float()` on descriptor | (e2e: 34.28 → 38.11 fps, +6%) | `.float()` on a non-contiguous bf16 tensor already allocates a contiguous fp32 copy; the intermediate `.contiguous()` was doing a second 1.2 MB bf16→bf16 copy. Host post phase 4.5 → 2.3 ms. |
| 12 | Skip `.float()` on `nms_scores`, keep bf16 | (e2e: 38.11 → 39.87 fps, +4.6%) | `torch.nonzero`, `torch.topk` and indexing all support bf16; only the keypoint coords need an fp32 cast at `grid_sample` call-site (`kp.float()[None]`). Saves a ~1 ms 614 KB bf16→fp32 host copy per iter. |
| 13 | Consolidate intermediate reshapes in `_device_fold_and_nms` | (e2e: 39.87 → **40.73 fps**, +2.2%) | Two intermediate reshape views — `(b,enc_h,enc_w,64)` and `(b,H,W,1)` — were unnecessary. Reshape directly from row-major `(b,1,enc_h·enc_w,64)` to 5D `(b,enc_h,enc_w,8,8)` pre-permute, and from the permuted tensor to flat `(1,1,b·H·W,1)` post-permute. |

### Reverts (PCC fell below 99% or no wall-clock gain)

| Change | Why reverted |
|---|---|
| bfloat8_b weights on encoder (whole) | score PCC dropped to 0.91 |
| bfloat8_b weights on encoder block 0 only | score PCC 0.91 |
| Encoder math fidelity LoFi (with bf16 weights + fp32 acc) | score PCC 0.91 |
| DRAM slice counts `(2,1,1,1)` and `(2,2,1,1)` | block-0 L1 CB overflow 1.58 MB > 1.57 MB |
| DRAM slice counts `(4,1,1,1)` | block-1 L1-full slower than 2-slice DRAM |
| `enable_weights_double_buffer=True` on convs | slower (tighter CBs) |
| `enable_act_double_buffer=True` | within noise |
| `reallocate_halo_output=True` | no effect |
| `full_inner_dim=True` | no effect |
| `act_block_h_override=32` on block 0 | no improvement |
| `BLOCK_SHARDED` on encoder block 3 | no benefit at 60×80 |
| `deallocate_activation=True` on convs | marginal regression |
| `WIDTH_SHARDED` on block 0 | OOM — 1-channel input can't distribute across banks |
| Device NMS via standalone trace | per-op Python dispatch ate the savings (+6% for +code) |
| Device fold+NMS Python-composed (pre-fused-kernel) (`36dc956`) | Used to be net-negative: 6 ms fold + host compare/mask cancelled the 36 ms host-NMS saving. **Superseded**: once `sp_eq_mul_mask` closes the compare+mask on device, the same fold chain becomes net-positive (+41% e2e, now the default via `SP_TRACE_NMS=1`). |
| Pack descriptor + NMS into one tensor for a single D2H | Tile→row-major layout conversion on 1.2 MB descriptor + `ttnn.concat` added ~7 ms of trace work AND blew up D2H to 53.5 ms (likely the combined tensor broke amortization of trace-tail wait). e2e 34.28 → 14.07 — biggest regression of the whole project. |
| `ThreadPoolExecutor` for host post-processing | Post is only 2–4 ms; the worker-thread submit/result barrier added ~0.6 ms and GIL contention with `ttnn.to_torch` pushed D2H up. Net flat within noise. |
| Both D2Hs as `from_device(blocking=False)` + `synchronize_device` | CQ0 dispatch serializes internally regardless; flat (35.89 vs 35.94 baseline). |
| Cast descriptor to `bfloat8_b` before D2H | Halves the device payload (1.2 MB → 614 KB) but the host-side bf8→fp32 unpack path was *slower* than bf16→fp32 — D2H grew 14.3 → 17.2 ms. Descriptor PCC held at 0.9991 so quality was fine; purely a ttnn host-unpack cost issue. |
| Skip `to_memory_config(DRAM)` before `to_layout(TILE)` on `s_pooled` | CRASH: `ttnn.max_pool2d`'s sharded output has shard shape (2793, 32) which isn't tile-aligned; `to_layout(TILE)` rejects sharded input unless shards are tile-aligned. Must interleave to DRAM first. |
| Input double-buffering (two `tt_in` buffers, two captured traces, alternating) | Tried twice — once with D2H split across CQs and once with D2H unchanged — BOTH regressed e2e to ~36 fps. Host post phase consistently jumped 1.8 → 4.4–4.7 ms even with identical post code; suspected DRAM contention between concurrent CQ1 H2D and CQ0 trace, or event-scheduling overhead with two tids. Requires tracy profiling to diagnose; not worth pursuing without profiler data. |

### What each run taught

- **Precision is a cliff, not a slope.** Either encoder ran in `bfloat16 +
  HiFi2 + fp32 accumulator` and PCC stayed ≥ 0.997, or it didn't and PCC
  fell off to ~0.91 immediately. No halfway config worked.
- **Trace is the biggest unlock by far.** Before trace, device compute was
  ~3% of wall. After trace it became the bulk of wall. Everything else is
  small-percentage tuning.
- **Structural knobs (DRAM slicing, shard layout, act block) converged at
  the baseline.** On this tiny (~1.3 M-param) model, the auto-chosen
  configs are close enough to optimal that explicit overrides mostly turn
  into noise or CB overflow.
- **NMS is the dominant host cost** once trace is on. The 9×9 max-pool at
  480×640 is what gates end-to-end throughput. Single-pass instead of 3-pass
  eliminated a 119 ms/iter wall.

### Attempted but not completed

- **Full fold + NMS inside the traced forward.** **LANDED and wins +41% e2e**
  (opt-in via `SP_TRACE_NMS=1`). The page-alignment error was fixed by
  pinning every reshape to `DRAM_MEMORY_CONFIG` and materialising the
  zero-padding tensor once (trace capture rejects in-trace `ttnn.zeros`
  writes). The previously-blocking overhead — a Python-composed
  eq + multiply that cost ~1.5 ms per extra op in the trace — is now
  replaced by the single-dispatch `ttnn.experimental.sp_eq_mul_mask`
  fused kernel. Host NMS (36 ms) is gone; device trace gains ~7 ms of
  fold + max_pool + fused mask. Net: e2e goes 17.0 → 24.1 fps on the
  natural image.
- **Device-side `grid_sample`.** `ttnn.grid_sample` exists and is
  verified working. On a natural image with ~500 keypoints, the host
  `F.grid_sample` costs ~1.15 ms — not a meaningful target against
  the 9–14 ms D2H dispatch floor. Worth doing when D2H stops being
  dispatch-dominated.
- **Input double-buffering to pipeline H2D with trace.** Two tt_in
  buffers, two captured traces, alternating per-iter so CQ1's H2D
  writes a *different* buffer than CQ0's current trace reads. Analysis
  suggested a ~25% ceiling uplift if D2H could also split across CQs.
  **Attempted and reverted twice** — both variants (D2H-split and
  D2H-unchanged) regressed the host post phase from 1.8 to ~4.5 ms,
  wiping out the expected device-side gains. The regression is
  reproducible but unexplained from Python alone; the most likely
  suspects are DRAM/NoC contention between the concurrent CQ1 H2D and
  CQ0 trace, or ttnn event-scheduling overhead when two tids alternate.
  Needs tracy profiling before re-attempting. Documented in `results.tsv`
  under commits `2f00f2b1` and `6d4fae39`.
- **1-channel NMS kernel (C++).** The current NMS chain pads
  `s_flat` from 1 channel to 32 (via `ttnn.concat` with a persistent
  zero-pad tensor) so that `ttnn.max_pool2d` and `sp_eq_mul_mask` — both
  of which require tile-aligned channel dims (multiples of 32) — can
  run. The 31 zero channels contribute nothing semantically. A custom
  1-channel `max_pool2d`-style Tensix kernel would eliminate the
  concat, one layout conversion, and the 32→1 slice at the end,
  cutting ~3 ms from the trace interior. Similar scope to the landed
  `sp_eq_mul_mask` kernel (~450 LoC).
- **Custom fused C++ Tensix kernel `ttnn.experimental.sp_eq_mul_mask`** —
  **LANDED and on the critical path** (see `kernels/sp_eq_mul_mask/`).
  Fuses `eq + multiply` into a single JIT-compiled Tensix program that
  keeps the mask tile in a DST register between the SFPU
  `eq_binary_tile` and `mul_binary_tile` calls — no DRAM round-trip for
  the intermediate. ~450 LoC of C++.
  - **Accuracy**: byte-identical to torch reference across match rates
    0 → 100% (max abs diff = 0.0, exact nonzero count).
  - **Throughput**: 0.184 ms/iter fused vs 0.276 ms/iter composed
    (`ttnn.eq` + `ttnn.multiply`) — **1.50×** on a 1×1×307 200×32 bf16 pair.
  - Closes one of the two remaining ops in the device-NMS chain (the
    other — a fold + max_pool + compare — would be a similar-sized custom
    op on top of this template).
- **Reducing `ttnn.copy_host_to_device_tensor`'s ~10.7 ms-per-call
  dispatch floor.** Runtime-level work; not addressable from the model
  layer. Would take `inference_speed` (forward + per-frame H2D) from
  ~72 fps toward the 353 fps compute ceiling.

## Layout

```
tt-superpoint/
├── README.md
├── run_benchmark.sh              # Driver; requires TT_METAL_DIR
├── results.tsv                   # Full experiment log
├── sample_data/
│   └── house_in_field_1080p.jpg  # Natural-image validation input
├── media/
│   └── sample.png                # Rendered keypoint visualisation
├── kernels/
│   └── sp_eq_mul_mask/              # Fused C++ Tensix kernel (eq + mul in one pass)
│       ├── README.md                # Install + measurements
│       ├── test.py                  # Correctness vs torch reference
│       ├── bench.py                 # Fused vs composed throughput
│       ├── {hpp,cpp,nanobind}       # Public API + Python binding
│       └── device/                  # Device op + program factory + 3 kernels
└── models/
    ├── visualize.py                 # Keypoint visualisation script
    ├── reference/
    │   └── superpoint_reference.py  # HF reference model loader + input helpers
    ├── tests/
    │   └── test_superpoint.py       # Benchmark + PCC + keypoint-set test
    └── tt/
        └── superpoint_ttnn.py       # tt-nn implementation
```

