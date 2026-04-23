# sp_eq_mul_mask — fused Tensix kernel

`ttnn.experimental.sp_eq_mul_mask(A, B)` → `A` where `A == B`, zero elsewhere.
Tile-elementwise, bf16, interleaved memory.

Replaces a two-kernel Python-composed chain (`ttnn.eq(A, B)` + `ttnn.multiply(A, mask)`)
with a single JIT-compiled Tensix program. The `eq` result stays in a DST
register and is consumed by the `multiply` in the same kernel — no DRAM
round-trip for the intermediate mask.

## Measured (N=100 async, DRAM-interleaved, 1×1×307 200×32 bf16 pair on p150b)

| Path | ms/iter | calls/s |
|---|---:|---:|
| `ttnn.experimental.sp_eq_mul_mask` (fused) | **0.184** | 5 447 |
| `ttnn.eq` + `ttnn.multiply` (composed) | 0.276 | 3 619 |

**1.50× throughput.** Byte-identical to the torch reference across
match-rate sweeps 0 → 100% (max abs diff = 0.0, nonzero count exact).

## Files

```
sp_eq_mul_mask.{hpp,cpp}                          public API
sp_eq_mul_mask_nanobind.{hpp,cpp}                 Python binding
device/sp_eq_mul_mask_device_operation.{hpp,cpp}  validation + launch
device/sp_eq_mul_mask_device_operation_types.hpp  attribute + input structs
device/sp_eq_mul_mask_program_factory.{hpp,cpp}   work-split, CB setup, kernel dispatch
device/kernels/sp_eq_mul_mask_compute.cpp         compute kernel (~50 LoC LLK)
device/kernels/sp_eq_mul_mask_reader.cpp          2-input sequential reader
device/kernels/sp_eq_mul_mask_writer.cpp          1-output sequential writer
```

~450 LoC total.

## Installation (in tt-metal checkout)

```
cp -r kernels/sp_eq_mul_mask ttnn/cpp/ttnn/operations/experimental/ssm/
# Then edit:
#   ttnn/cpp/ttnn/operations/experimental/ssm/CMakeLists.txt        — add sources + kernel glob
#   ttnn/CMakeLists.txt                                              — add _nanobind.cpp to source list
#   ttnn/cpp/ttnn/operations/experimental/experimental_nanobind.cpp  — #include + call bind_sp_eq_mul_mask(mod)
# Then:
(cd build_Release && ninja ttnncpp _ttnn.so)
```

## Test

```
python kernels/sp_eq_mul_mask/test.py   # correctness vs torch reference
python kernels/sp_eq_mul_mask/bench.py  # throughput fused vs composed
```

## Compute kernel core logic

```cpp
for (tile in num_tiles) {
    copy_tile(cb_a, 0, 0);     // DST[0] = A
    copy_tile(cb_b, 0, 1);     // DST[1] = B
    eq_binary_tile_init();
    eq_binary_tile(0, 1, 2);   // DST[2] = (A == B) ? 1 : 0   (SFPU)
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);  // DST[0] = A * mask            (SFPU, in-register)
    pack_tile(0, cb_out);
}
```
