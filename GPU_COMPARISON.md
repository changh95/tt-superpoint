# superpoint-p150 — Blackhole p150a vs RTX 5090 (same host, same weights, same input)

Date 2026-09-14. Facts only; every number below is measured in this pass on the GPU or copied
(with its source line) from the p150a validation reports. The p150a was NOT touched.

## What was run

| | |
|---|---|
| Model | SuperPoint, the port's own torch reference: `transformers.SuperPointForKeypointDetection` (`models/superpoint-p150/code/models/reference/superpoint_reference.py::load_reference_model`, the same class `server/app.py::_load_reference` wraps into `TtSuperPoint`), 1,300,865 parameters, transformers 5.12.1 |
| Weights | `magic-leap-community/superpoint` @ `734450e9ffe229074f5998494ddc615475cdb20a` (tt-model.yaml `weights.revision` = `serve.env.TT_WEIGHTS_REVISION`), `model.safetensors` from the HF cache `~/.cache/huggingface/hub/models--magic-leap-community--superpoint/snapshots/734450e9…`, `HF_HUB_OFFLINE=1` |
| Input | `code/sample_data/house_in_field_1080p.jpg` (1600x900 RGB JPEG — the frame the p150a served numbers were taken on, 539 kp; `media/sample.png` is its 480x640 keypoint visualisation, not an input) -> `server/app.py::_preprocess` copied verbatim: PIL bilinear resize to 640x480, /255 -> `pixel_values [1,3,480,640]` fp32, batch 1; the model reads channel 0 (R) (`extract_one_channel_pixel_values`) |
| GPU | NVIDIA GeForce RTX 5090 (sm_120), driver 580.126.18, power limit 600 W, 32607 MiB; idle 29.8 W |
| venv | `/home/deepgadget/experiments/tt-models/.venv-gpu/main` — Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, torchvision 0.26.0, transformers 5.12.1, numpy 1.26.4, safetensors 0.8.0, huggingface_hub 1.31.0, pillow 12.3.0, triton 3.6.0 |
| Repo state | `models/superpoint-p150` @ `8528a70` (branch `tt-model-package`), read-only; the port's torch-only `models/tt/postprocess.py` (fold_scores / simple_nms / postprocess_from_nms_map) is imported unchanged |
| Script | `logs/gpu-vs-p150/superpoint/bench_superpoint_gpu.py` (uses `logs/gpu-vs-p150/bench_common.py`); logs `full_run.log` (eager + served-like), `compile_only_nocache_{tf32,bf16_autocast,fp16_autocast}.log` (the `torch.compile` rows), `probe_compile_*.{py,log}` (the inductor-cache investigation below); raw JSON `reports/gpu-vs-p150/superpoint.json` (= `logs/gpu-vs-p150/superpoint/result.json`); CPU reference tensors `cpu_fp32_reference.pt` |
| Commands | `HF_HUB_OFFLINE=1 .venv-gpu/main/bin/python bench_superpoint_gpu.py --iters 50 --warmup 10` then, one process per precision, `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 HF_HUB_OFFLINE=1 … bench_superpoint_gpu.py --iters 50 --warmup 10 --compile-only --compile-precisions <tf32|bf16_autocast|fp16_autocast>` |
| Loop | per precision 10 warm-ups + 50 timed iterations, `torch.cuda.synchronize()` before/after each; wall-clock (perf_counter) is the primary number, CUDA-event time recorded alongside (within 0.01 ms of wall) |
| p150a source | `reports/gpu-vs-p150/p150_numbers.json` -> `reports/megakernel/PUBLISH_SUMMARY.md:11` (Hub `tt serve`: device_forward 5.11 ms, total 23.94 ms), `models/superpoint-p150/DEVICE_VALIDATION.md` "## Results" (served table: fused 5.28 / 5.07 ms device_forward, preprocess 17.5-18.1 ms, postprocess 0.96-1.29 ms, total 23.79-24.71 ms; fused A/B `fused_h2d_plus_trace_ms` 3.99), `reports/megakernel/VALIDATION_SUMMARY.md:32` |

GPU forward variants (all the same weights, all on the GPU):

- **device_fwd** (the primary number) = encoder + keypoint head (65-ch softmax) + descriptor head (L2-normalised 256x60x80) + the port's `fold_scores` (drop dustbin, 8x8 unfold) + single-pass NMS radius 4 (`simple_nms`, one max-pool + eq) -> `nms_map [1,480,640]` + `descriptors [1,256,60,80]`. This is exactly the work inside the p150a `timing_ms.device_forward` key (whole-graph metal trace incl. on-device NMS-T, readback of the NMS map and the descriptor map). The host post-processing after it (threshold 0.005 / border 4 / top-1024 / grid_sample) is the same port code on both sides.
- **dense** = encoder + heads only (no fold/NMS): the tensors the port's PCC gate compares (informational).
- **hf_full** = `SuperPointForKeypointDetection.forward` as shipped by transformers: dense + HF 3-pass NMS + threshold/border/top-k (`max_keypoints=-1` in the config -> all) + `grid_sample`, on the GPU (informational; data-dependent shapes force a sync inside).

Timing definitions (they match the p150a `timing_ms` keys):

- **incl_h2d** = `pixel_values.to("cuda")` (3.7 MB, pageable host tensor as produced by the server preprocess) + device_fwd + `nms_map.cpu()` + `descriptors.cpu()` (1.2 + 4.9 MB). Compare with p150a `timing_ms.device_forward` = upload of the 480x640 frame + trace replay + readback of both maps (**5.11 ms**, PUBLISH_SUMMARY.md:11; 5.28 / 5.07 in the DEVICE_VALIDATION sessions).
- **excl_h2d** = device_fwd only, input already resident, outputs left on the device.
- **served-like** = base64 decode + JPEG decode (PIL, convert RGB) + `_preprocess` + incl_h2d device_fwd + `postprocess_from_nms_map` + `argsort` (the interval `server/app.py` reports as `timing_ms.total`; response JSON / npz encode is outside that key on both sides). Compare with p150a `timing_ms.total` (**23.94 ms**, PUBLISH_SUMMARY.md:11).

## Correctness check (GPU vs CPU fp32 reference)

CPU fp32 (same process, same `pixel_values`): dense 119 ms, device_fwd 150 ms, HF full forward 208 ms; served-path output 526 keypoints
(top-3 in original pixels `[610.0, 703.125] [1042.5, 446.25] [1122.5, 442.5]`, scores 0.6283 / 0.6044 / 0.5985 — the same three positions the
published p150a card lists, at bf16-rounded scores 0.6094 / 0.5898 / 0.5820). The p150a served 539 kp on this frame: the 13 extra points come from bf16
score rounding around the 0.005 threshold / NMS ties (its pre-NMS score PCC vs this reference is 0.9971, VS:32).

| GPU precision | pre-NMS score map PCC (p150a gate metric) | descriptor map PCC (p150a gate metric) | NMS map PCC | kp top-500 @2 px recall / precision / F1 | served output vs CPU (526 kp) |
|---|---:|---:|---:|---:|---|
| **fp32 strict** | **1.000000** (max abs diff 2.7e-6) | **1.000000** (1.1e-6) | 1.000000 | 0.998 / 0.998 / 0.998 | 526 kp, **identical keypoint set**, scores within 1.9e-6, descriptor cosine >= 0.9999993 |
| tf32 | 0.999994 (2.8e-3) | 0.999998 (1.4e-3) | 0.999937 | 1.000 / 1.000 / 1.000 | 526 kp; 99.62 % of ref keypoints at the exact pixel, 99.81 % within 1 px |
| bf16 autocast | 0.999155 (3.1e-2) | 0.999484 (1.9e-2) | 0.983589 | 0.992 / 0.994 / 0.993 | 530 kp; 95.8 % exact, 99.6 % within 1 px |
| fp16 autocast | 0.999963 (8.4e-3) | 0.999982 (5.4e-3) | 0.998310 | 0.996 / 1.000 / 0.998 | 527 kp; 99.0 % exact, 99.6 % within 1 px |
| p150a (bf16 device, VS:32 / DEVICE_VALIDATION Results) | 0.997109 | 0.999085 | n/a (NMS map bit-identical to the host fold+NMS of its own scores) | F1 0.98796 | 539 kp |

PCC > 0.999 holds for fp32 strict (1.000000 on both gate tensors; the served keypoint set is identical to the CPU one), so the GPU runs the right
model. The top-500 F1 of 0.998 rather than 1.0 in fp32 comes from one ~1e-6 score tie at rank 500 flipping the k-th point — the full served set is identical.
fp16 autocast is numerically fine (0.99996 / 0.99998, above the port's 0.997 / 0.999 gates); bf16 autocast (the p150a's own activation class) is
at 0.99916 / 0.99948 — above the p150a's own 0.9971 / 0.99909, with a slightly lower keypoint F1 (0.993 vs 0.988 for the p150a).

## GPU latency (batch 1, 480x640, median / min / p90 of 50 iterations, wall-clock ms)

Eager PyTorch, **device_fwd** (== p150a device_forward work):

| precision | incl_h2d median / min / p90 | excl_h2d median / min / p90 | CUDA-event excl | first call ms | power mean W (excl loop) | power mean W (incl loop) | GPU util % (excl) | peak mem alloc / reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | **2.938** / 2.908 / 2.964 | **2.259** / 2.238 / 3.516 | 2.247 | 2.3 (199.9 on the very first call incl. CUDA/cuDNN init) | 548.0 | 299.8 | 95.5 | 176 / 404 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | **2.097** / 2.072 / 2.118 | **1.425** / 1.416 / 1.434 | 1.415 | 1.5 | 409.5 | 423.7 | 88.7 | 327 / 406 |
| bf16 autocast (+TF32 remainder) | **1.586** / 1.559 / 1.616 | **0.913** / 0.901 / 0.921 | 0.905 | 0.9 | 397.5 | 346.3 | 85.4 | 175 / 406 |
| fp16 autocast (+TF32 remainder) | **1.526** / 1.502 / 1.555 | **0.857** / 0.847 / 0.864 | 0.848 | 0.9 | 396.3 | 335.8 | 84.2 | 175 / 406 |
| tf32, pinned host input (informational) | 2.046 / 2.023 / 2.065 | | | | | | | |
| bf16 autocast, pinned host input (informational) | 1.546 / 1.520 / 1.573 | | | | | | | |

The incl-excl gap of ~0.67 ms is the PCIe traffic (3.7 MB pageable upload + 6.1 MB readback of the two fp32 maps); the fp32-strict excl loop had a
few slow iterations (p90 3.52, max 4.66 ms) while its median/min are tight. The other two variants (same precisions, incl_h2d / excl_h2d medians):

| precision | dense (no fold/NMS) | hf_full (transformers forward incl. 3-pass NMS + top-k + grid_sample, kp count) |
|---|---:|---:|
| fp32 strict | 2.911 / 2.230 | 2.973 / 2.674 (581 kp) |
| tf32 | 2.072 / 1.397 | 2.129 / 1.833 (581 kp) |
| bf16 autocast | 1.563 / 0.892 | 1.654 / 1.350 (587 kp) |
| fp16 autocast | 1.512 / 0.837 | 1.588 / 1.296 (582 kp) |

The port's fold + single-pass NMS costs ~0.03 ms on the GPU; the HF 3-pass NMS + host-style keypoint extraction adds ~0.45 ms and returns every
point above threshold (581, no top-k, `max_keypoints=-1`).

`torch.compile` (inductor, `dynamic=False`, device_fwd; each precision compiled in its own fresh process with `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`
— see the note below; compile time 2-3.4 s, far under the 5-min budget):

| variant | compile s | incl_h2d median / min / p90 | excl_h2d median / min / p90 | power W (excl loop) | peak mem MiB | NMS map PCC vs CPU fp32 | kp F1 top-500 | served kp (exact-pixel match to CPU) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tf32 + compile default | 3.4 | 1.832 / 1.802 / 1.863 | 1.134 / 1.127 / 1.142 | 395.4 | 251 | 0.999937 | 1.000 | 526 (99.6 %) |
| tf32 + compile reduce-overhead (CUDA graphs) | 1.8 | **1.776** / 1.750 / 1.801 | **1.051** / 1.045 / 1.058 | 421.0 | 18 | 0.999937 | 1.000 | 526 (99.6 %) |
| bf16 autocast + compile default | 3.4 | 1.359 / 1.320 / 1.387 | 0.647 / 0.636 / 0.659 | 362.1 | 138 | 0.979078 | 0.994 | 527 (95.1 %) |
| bf16 autocast + compile reduce-overhead | 1.8 | **1.291** / 1.269 / 1.326 | **0.588** / 0.581 / 0.600 | 381.3 | 18 | 0.979078 | 0.994 | 527 (95.1 %) |
| fp16 autocast + compile default | 3.3 | 1.286 / 1.256 / 1.328 | 0.588 / 0.580 / 0.596 | 361.6 | 100 | 0.996632 | 0.998 | 526 (98.9 %) |
| fp16 autocast + compile reduce-overhead | 1.9 | **1.211** / 1.187 / 1.238 | **0.505** / 0.497 / 0.514 | 388.7 | 18 | 0.996632 | 0.998 | 526 (98.9 %) |

Compiled descriptor-map PCC: tf32 0.999998, bf16 0.999678, fp16 0.999987 (pre-NMS score map is not an output of the compiled graph; its eager value applies).
Compile note (`probe_compile_single.py`, `probe_compile_graph.py`): with inductor's default on-disk cache the fp16 compile returned the *bf16*
kernels (bit-identical outputs and latency to the bf16 compile in fresh processes, "compile" 0.1 s) although the dynamo/AOT graph carried
`_to_copy(dtype=float16)` casts — a cache-key collision across autocast dtypes in torch 2.11.0+cu128. The rows above were therefore re-measured
with `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`, one precision per process; with caches off the compiled fp16 output tracks eager fp16 (NMS-map PCC 0.998
vs eager) and differs from bf16. The bf16 rows were identical with and without the cache.

Other facts: model to cuda 0.002 s (5 MB of weights; CPU load from safetensors 0.02 s); first fp32 call 200 ms (CUDA/cuDNN init); idle GPU power
29.8 W; GPU utilisation 84-96 % in the eager excl loops (this graph is 12 small convs — kernel-launch-bound at batch 1, which is why CUDA graphs and
autocast both help and why fp32 strict draws the most power).

Served-like loop (same host work as `server/app.py::predict`, 50 iterations after 10 warm-ups, medians ms):

| GPU precision | decode (base64 + JPEG 1600x900) | preprocess (PIL resize + /255) | device_forward (incl_h2d) | postprocess | **total** | p90 total | power W (loop) | keypoints |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp32 strict | 11.74 | 6.18 | 3.00 | 1.52 | **22.75** | 24.33 | 235.2 | 526 |
| tf32 | 11.93 | 6.03 | 2.15 | 2.21 | **22.21** | 23.15 | 103.5 | 526 |
| bf16 autocast | 11.90 | 5.96 | 1.64 | 2.26 | **21.45** | 22.13 | 93.6 | 530 |
| fp16 autocast | 12.05 | 6.15 | 1.59 | 2.19 | **21.90** | 23.36 | 91.0 | 527 |
| p150a (PUBLISH_SUMMARY.md:11; split DEVICE_VALIDATION.md Results, Hub run) | ~17.5-18.1 (decode + resize as one key) | | 5.11 | 0.96-1.29 | **23.94** | | not measured | 539 |

The host stages dominate on both sides: the same PIL JPEG decode + resize takes ~18 ms here (11.9 + 6.0; the p150a container reports 17.5-18.1 ms for the
same two steps as one key), so the served totals differ by roughly the inference term only. The GPU postprocess (torch CPU ops in this venv) is
1.5-2.3 ms vs 0.96-1.29 ms in the p150a container — same code, different host process / torch build; not a device difference.

## Comparison with the p150a (matching definitions)

Ratio = p150a ms / GPU ms (> 1 means the GPU is faster). p150a precision: bf16 activations, HiFi2 + fp32 accumulate on device, one metal trace with
device NMS-T (DEVICE_VALIDATION.md Results). GPU precision per row as stated.

| row | p150a (definition) | GPU precision | GPU ms | ratio p150a/GPU |
|---|---:|---|---:|---:|
| device forward (p150a `timing_ms.device_forward` = upload + trace incl. NMS + readback vs GPU device_fwd incl_h2d) | 5.11 (PUBLISH_SUMMARY.md:11, Hub tt serve; 5.28 / 5.07 DEVICE_VALIDATION) | fp32 strict | 2.938 | **1.74** |
| | | tf32 | 2.097 | **2.44** |
| | | bf16 autocast (the p150a's precision class) | 1.586 | **3.22** |
| | | fp16 autocast | 1.526 | **3.35** |
| | | tf32 + compile reduce-overhead | 1.776 | **2.88** |
| | | bf16 autocast + compile reduce-overhead | 1.291 | **3.96** |
| | | fp16 autocast + compile reduce-overhead | 1.211 | **4.22** |
| GPU forward only (excl_h2d, no PCIe) vs the same p150a 5.11 (which cannot exclude its transfers) | 5.11 | fp32 strict / tf32 / bf16 / fp16 | 2.259 / 1.425 / 0.913 / 0.857 | 2.26 / 3.59 / 5.60 / 5.96 |
| | | tf32 / bf16 / fp16 + compile reduce-overhead | 1.051 / 0.588 / 0.505 | 4.86 / 8.69 / 10.12 |
| GPU forward only (excl_h2d) vs p150a `fused_h2d_plus_trace_ms` 3.99 (DEVICE_VALIDATION Results, host harness: upload + trace, no readback/convert) | 3.99 | fp32 strict / tf32 / bf16 / fp16 | 2.259 / 1.425 / 0.913 / 0.857 | 1.77 / 2.80 / 4.37 / 4.66 |
| served e2e (p150a `timing_ms.total` vs GPU served-like total) | 23.94 (PUBLISH_SUMMARY.md:11) | fp32 strict | 22.75 | **1.05** |
| | | tf32 | 22.21 | **1.08** |
| | | bf16 autocast | 21.45 | **1.12** |
| | | fp16 autocast | 21.90 | **1.09** |

Reading: on the device-forward definition (upload + network + NMS + readback) the eager RTX 5090 is 1.7x faster than the p150a's fused trace in strict
fp32 (2.94 vs 5.11 ms) and 3.2x faster in the p150a's own precision class (bf16 autocast, 1.59 ms); the best GPU configuration measured (fp16 autocast +
inductor + CUDA graphs) is 4.2x faster (1.21 ms) with better agreement to the fp32 reference than the p150a (kp F1 0.998 vs 0.988). About 0.7 ms of every
GPU incl_h2d number is PCIe traffic for the 6 MB of fp32 maps — the fold/NMS itself is ~0.03 ms. End-to-end the ~18 ms of identical host JPEG decode /
resize work on both sides compresses the gap to 1.05-1.12x (22-23 vs 23.94 ms): for this model the served latency is host-bound on either accelerator.

Not measured / not claimed: p150a power (not measured in any pass -> no power or efficiency comparison; the GPU drew 360-550 W mean during the dense
excl loops, 90-235 W in the host-bound served-like loops, 30 W idle). p150a numbers were not re-measured. The GPU numbers exclude HTTP/JSON framing
and the npz/base64 response encode, as do the p150a `timing_ms` keys. Keypoint counts differ across precisions (526-530 here, 539 on the p150a)
because points near the 0.005 threshold / NMS ties flip with rounding; the top-ranked keypoints and their positions agree.

## Reproduce

```bash
cd /home/deepgadget/experiments/tt-models/logs/gpu-vs-p150/superpoint
HF_HUB_OFFLINE=1 /home/deepgadget/experiments/tt-models/.venv-gpu/main/bin/python bench_superpoint_gpu.py --iters 50 --warmup 10 | tee full_run.log
for P in tf32 bf16_autocast fp16_autocast; do   # one autocast dtype per process, inductor caches off (see the compile note)
  TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 HF_HUB_OFFLINE=1 /home/deepgadget/experiments/tt-models/.venv-gpu/main/bin/python \
    bench_superpoint_gpu.py --iters 50 --warmup 10 --compile-only --compile-precisions $P | tee compile_only_nocache_$P.log
done
# outputs: /home/deepgadget/experiments/tt-models/reports/gpu-vs-p150/superpoint.json, ./result.json, ./cpu_fp32_reference.pt
```

GPU released after the run: `nvidia-smi --query-compute-apps=pid --format=csv,noheader` -> empty.
