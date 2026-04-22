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
bash run_benchmark.sh
```

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

| Metric | Random input | Natural image | Paper (Titan X, 2018 Caffe) |
|---|---:|---:|---:|
| Traced forward pass | 73.57 fps | **73.60 fps** | 90 fps (11.15 ms) |
| `fps_match_paper` (forward + descriptor sampling) | 40.90 fps | **41.55 fps** | 70 fps (13 ms) |
| Full e2e (incl. host NMS) | 17.02 fps | **17.03 fps** | not reported |

Measurement methodology: 100 iterations after a warm-up replay;
traced device forward on cq_id=0 with input H2D overlapped on cq_id=1.

### Comparison read

- Traced forward pass is within **82%** of the 2018 Titan X paper baseline.
- `fps_match_paper` (the metric that actually lines up with the paper's
  published number) is at **59%** of the paper.
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

- **Full fold + NMS inside the traced forward.** Reshaping the
  `(1, 1, 4800, 64)` intermediate through `(1, enc_h, enc_w, 8, 8)` and
  a 5-D permute into `(1, 480, 640, 1)` triggered
  `Shard page size must currently have L1 aligned page size` errors from
  ttnn's internal `interleaved_to_sharded` step on the reshape output.
  Resolving it would likely require a custom C++ kernel that avoids the
  sharded intermediate.
- **Custom fused C++ Tensix kernel (conv + pool, or fold + max-pool).**
  Each experimental fused op in `tt-metal` is roughly 700–1000 LoC across
  reader/writer/compute kernels, program factory, and nanobind
  registration. Out of session scope.

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
└── models/
    ├── visualize.py                 # Keypoint visualisation script
    ├── reference/
    │   └── superpoint_reference.py  # HF reference model loader + input helpers
    ├── tests/
    │   └── test_superpoint.py       # Benchmark + PCC + keypoint-set test
    └── tt/
        └── superpoint_ttnn.py       # tt-nn implementation
```

