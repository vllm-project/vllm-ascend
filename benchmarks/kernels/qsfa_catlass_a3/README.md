# CATLASS QSFA decode on Atlas A3

This single-operator benchmark uses the GLM-5.4 QSFA shape at TP1, batch 10,
MTP5 (60 query rows), 128K KV tokens and Top-2048. Queries are BF16 with 64
heads, 512 latent channels and 64 RoPE channels. The paged KV cache uses packed
INT8. The baseline is CANN's `torch_npu.npu_kv_quant_sparse_flash_attention`.

The A3 kernel gathers and dequantizes each sparse KV tile once, stages it in NZ
layout, and uses CATLASS `TileMmad` for QK and PV. It keeps Q in L1 and retains
the first 384 latent channels of the last 128-token stripe of each 512-token
QK tile. Two task-parity L1 banks let PV reuse that stripe while the two
scratch banks service the other loads. Online softmax keeps FP32 statistics.

## One logical NPU on Atlas A3 560T

CANN 9.2.1, PyTorch 2.10.0 and torch_npu 2.10.0.post7. Each seed used 23
paired ABBA/BAAB measurements, with nine operator calls per ACL graph replay.
Latency and useful MFU are per operator call.

| Seed | P50, µs (stock → CATLASS) | P95, µs (stock → CATLASS) | Paired speedup (95% CI) | Useful MFU (stock → CATLASS) |
| --- | ---: | ---: | ---: | ---: |
| 93211 | 288.76 → 262.25 (−9.18%) | 289.65 → 263.32 | 1.102× [1.094, 1.105] | 21.17% → 23.30% |
| 93212 | 289.03 → 266.20 (−7.90%) | 290.59 → 267.39 | 1.085× [1.080, 1.088] | 21.15% → 22.96% |

Useful MFU counts 17.113 GFLOP of QK and PV per call against a 280 BF16 TFLOP/s
peak per logical NPU. A separate HBM PMU capture with seed 93211 used four
windows per implementation. Physical MBU rose from **14.47% to 15.47%** at a
1,600 GB/s peak. Read plus write bandwidth rose from 231.56 to 247.60 GB/s and
estimated traffic per call fell from 67.79 to 66.79 MB.

Both seeds matched stock exactly, with zero maximum absolute error in the
graph-to-eager check. A causal-boundary fixture with eight filtered indices
per query row also matched stock exactly. The maximum absolute error versus
the FP32 reference across those three checks was 0.00105 for both implementations.
The measurements cover B10 single-operator decode. [B10 measurements](results-b10.csv)

## A3 and A5 data paths

On A3, Cube and Vector exchange QK scores, softmax probabilities and PV
partials through global memory. Kernel fusion can reduce scheduling overhead
and L1 reuse saves some KV reloads, while the intermediate handoffs still
require GM transfers and synchronization. A5 supports L0C-to-UB and UB-to-L1
transfers, so QK scores and probabilities can stay on chip between Cube and
Vector. This explains one part of the gap. Quantifying its contribution
requires a matched cross-platform benchmark. [A3 architecture][a3] · [A5 architecture][a5]

## Reproduce the B10 latency and accuracy check

From the repository root on A3, with CANN initialized and the bundled CATLASS
submodule checked out at `41bf90da655bba3c66d0acd7e00abe33960ecfd6`:

```bash
git submodule update --init csrc/third_party/catlass
python benchmarks/kernels/qsfa_catlass_a3/catlass_qsfa.py \
  --catlass-include csrc/third_party/catlass/include
python benchmarks/kernels/qsfa_catlass_a3/benchmark.py --seed 93211
python benchmarks/kernels/qsfa_catlass_a3/benchmark.py --seed 93212
python benchmarks/kernels/qsfa_catlass_a3/benchmark.py --seed 93211 --causal-boundary
```

The graph runner reports latency and useful MFU. Physical MBU in the CSV was
measured with `msprof` HBM counters at 50 Hz over four 450 ms graph-replay
windows per implementation, discarding 50 ms at each edge. The kernel includes code derived from
CANN ops-transformer revision `8fbf1ffea0cf0c401407fdf202743a75e0a427b7`.
[CANN Open Software License Agreement Version 2.0][cann-license]

[a3]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_00031.html
[a5]: https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00065.html
[cann-license]: LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt
