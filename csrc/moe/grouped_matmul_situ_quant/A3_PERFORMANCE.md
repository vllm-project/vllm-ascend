# A3 SiTU integration and performance investigation

## Scope and status

The A3 INT8/INT4 MSD path uses the same `DeviceOperator`, four Torch dispatcher
entries, and GE `GroupedMatmulSituQuant` registration as the A5 MX path.
The A5 computation is unchanged. Both the custom OPP and framework extension
must be rebuilt because the shared native signature has additional inputs.

The optimization reduces A3 launch overhead and workspace requests. It does
**not** establish a universal speedup over stock GMM followed by SiTU.
Compact three-expert inputs still regress. This change should remain a draft
until the default dispatch policy and representative K3.1 serving performance
are reviewed; operator tests do not establish full-model accuracy or throughput.

Baseline source: `76ea9d24f155afe032ad9ac6c98717f1f7334f36`.
The performance baseline is the initial A3 integration before the optimizations
below, rebuilt with the same toolchain and preserved in a separate OPP package.
It is distinct from the stock split-path reference.

## Concrete changes

| Area | Files | Behavior |
| --- | --- | --- |
| Device selection | `vllm_ascend/device/hardware_profile.py`, `device_op.py` | Advertise A3 SiTU capability and route through the existing device interface. |
| MoE integration | `vllm_ascend/quantization/methods/w4a8/w4a8.py` | Select A3 fusion for SiTU, per-channel weights, compensation, supported group-list types and aligned K/N; retain split fallback for unsupported cases. |
| Native interface | `grouped_matmul_situ_quant_torch_adpt.h`, `op_host/op_api/`, `csrc/torch_binding_meta.cpp` | Validate A3 metadata, support tensor/list carriers and INT8 output Meta inference; preserve A5 MX dispatch. |
| INT4 descriptors | Torch adaptor | Convert carrier view offsets, storage offsets, outer strides and storage shape consistently; sliced experts retain the correct underlying addresses. |
| Registration/build | `op_host/`, `csrc/build_aclnn.sh` | Add arch32 tiling/kernel selection to the shared operator package. |
| MSD arithmetic | `op_kernel/arch32/` | Split INT8 into signed INT4 partials, accumulate through FP16, restore compensation, apply SiTU and per-token INT8 quantization. Zero rows avoid a reciprocal of zero. |
| Tiling initialization | `op_kernel/grouped_matmul_situ_quant_apt.cpp` | Decode one enclosing tiling record and keep it alive until `Process()` finishes. The Matmul implementation retains a pointer to its selected member. |
| Single-chunk synchronization | `arch32/grouped_matmul_situ_quant_a8w4_msd_pipeline.h` | Use directional producer notifications instead of two complete bidirectional barriers. Empty routes return early; multiple chunks retain the existing synchronized pipeline. |
| Workspace sizing | `op_host/grouped_matmul_situ_quant_a8w4_base_tiling.cpp` | Allocate each workspace half for `min(input_capacity, chunk_limit)` rows. The chunk limit and double-buffer indexing are unchanged. |
| Regression coverage | Native A3 suite, W4A8/device unit tests, benchmark script | Check real execution, graph route changes, descriptor views, failures, numerical bounds and reproducible paired timings. |

The single-chunk dependency chain is:

```text
all AIV pre-process stores (MTE3)
    -> AIV-only barrier (flag 4)
    -> paired AIV-to-AIC notification (flag 5)
    -> Cube matmul and stores (FIX)
    -> AIC-only barrier (flag 6)
    -> paired AIC-to-AIV notification (flag 7)
    -> SiTU and quantization
```

Every core participates in its barrier, including cores with no rows or tiles.
Flags 4 through 7 do not overlap the CANN synchronization flags 11 through 14.
No device routing tensor is read back to the host. The fast path is selected
from live rows on device, so graph replay can switch between zero, one and
multiple chunks without recapture.

## Profiling evidence

Validation used A3 SoC 253, NPU 0, CANN 9.2.1, PyTorch 2.10.0+cpu,
torch_npu 2.10.0.post7 and Python 3.12.16. A5 regression used SoC 260.
Both used isolated build directories and private `ASCEND_CUSTOM_OPP_PATH`;
no system OPP or serving deployment was replaced.

`torch_npu.profiler` collected Level1 `PipeUtilization` and a separate `L2Cache`
run. Each graph contains 20 calls, replayed three times for profiling, yielding
60 device records per implementation. Raw evidence IDs are directory names
relative to the saved evidence bundle.

| Evidence | Workload | Observation |
| --- | --- | --- |
| `profile-baseline-small` | E=3, counts=[3,0,2], K=7168, N=4096 | Fused kernel median 65.291 us; stock GMM 37.401 us and SiTU 5.780 us. Fused/stock Cube MAC activity is essentially identical (2.496/2.502 us), while scalar activity is 28.477/16.699 us. |
| `profile-baseline-medium` | counts=[32,32,64] | Fused kernel 100.212 us; GMM 49.961 us, SiTU 15.300 us. Cube activity spans 80.990/44.855 us but MAC activity is 13.204/12.572 us. |
| `profile-baseline-l2` | Small workload | Both paths have near-complete Cube read cache hits; removing the weight cache hint did not improve latency. |
| `profile-final-small` | Small workload, final candidate | Fused kernel 51.571 us, versus baseline 65.291 us; Cube scalar activity 22.460 us, versus 28.477 us. |
| `profile-final-medium` | Medium workload, final candidate | Fused kernel 92.262 us, versus baseline 100.212 us; AIV MTE2 activity 5.993 us, versus 14.687 us. |
| `profile-directional-medium` | Medium workload, directional synchronization | Fused kernel 91.492 us; AIV MTE2 activity fell from 14.687 to 6.323 us. Waiting and scheduling overhead remain substantial. |

These observations support synchronization and scheduling overhead as major
contributors, rather than additional matrix arithmetic or weight-cache misses.
Pipe counters overlap and are averaged across cores; they must not be summed
as a critical-path decomposition. The stock GMM implementation also overlaps
per-tile MSD post-processing with Cube work, whereas this fused implementation
waits for the chunk's Cube outputs before full-row SiTU reduction. This is a
remaining structural optimization opportunity, not a measured attribution of
every microsecond of the gap.

The profiler could not connect some ACL graph replay events to NPU flow links.
The device `op_summary.csv` records were present and used for the comparisons.
Profiler-instrumented durations above are separate from event-timed latency.

Rejected experiments: disabling the weight cache hint, changing NZ tiles to
64x128x512 or 64x256x256, and manually loading selected tiling fields did not
provide stable gains. These experiments are not in the delivered code.

## Unprofiled performance

The benchmark fixes seed 721, creates native NZ INT8 storage with INT32 carrier
views, warms each graph five times, and measures six ABBA rounds with NPU
events: 12 samples per implementation, 20 calls per sample. Both sides include
the same input clone because stock GMM may mutate its input. Tensor-list
weights own their per-expert INT8 storage before creating INT32 views.

The final baseline/candidate comparison uses separate processes and OPP paths
in baseline/candidate, candidate/baseline, baseline/candidate order. Each process
also measures the stock split reference. Timing samples and fixed precision
checks are saved as JSON. No profiler is enabled during these runs.

| Case (E=3) | Initial fused | Optimized fused | Split in candidate run | Fused reduction | Remaining gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tensor, M=5 | 66.690 | 54.326 | 46.478 | 18.5% | +16.9% |
| Tensor, M=128 | 115.898 | 102.606 | 74.735 | 11.5% | +37.3% |
| List, M=5 | 69.096 | 55.823 | 51.897 | 19.2% | +7.6% |
| List, M=128 | 110.841 | 103.131 | 75.459 | 7.0% | +36.7% |

Values are medians of 36 samples per code state and case. Split timing also
varies across processes (for example, tensor M=128: 79.157 us in baseline
processes versus 74.735 us in candidate processes). Raw fused reductions
therefore include run-to-run variation; the split-normalized improvement for
that case is about 6.2%. All three run medians and every sample are retained
in `paired-summary.json` and `paired-*.json`.

Additional candidate measurements (microseconds; separate runs):

| E / live rows / capacity | Fused | Split | Fused versus split |
| --- | ---: | ---: | ---: |
| 3 / 2048 / 2048, counts=[512,512,1024] | 647.144 | 462.193 | +40.0% |
| 16 / 128 / 128, eight rows per expert | 200.685 | 228.498 | -12.2% |
| 3 / 5 / 2048, counts=[3,0,2] | 63.366 | 72.484 | -12.6% |

All use K=7168 and N=4096. These are synthetic operator workloads, not measured
K3.1 request distributions. No whole-model speedup is claimed.

## Workspace

The original host tiling requested approximately 64 MiB of user workspace for
every nonempty call, plus 16 MiB of system workspace. For supplied compensation,
the new user request is:

```text
2 * min(input_capacity, chunk_limit) * (K + 4*N) bytes
```

For K=7168 and N=4096, this is 235,520 bytes at capacity 5, and 6,029,312 bytes
at capacity 128. Capacities above the chunk limit retain the bounded two-half
workspace. The no-compensation path additionally reserves one auxiliary row
per expert in each half. These are host tiling allocation requests, not measured
peak device-memory savings. Torch graph-memory counters do not include every
allocation made by the native executor and cannot establish that peak saving.

## Correctness and build coverage

- A3 native suite: **44 passed**, including ND/NZ, tensor/list, sliced storage,
  supplied/computed compensation, zero rows, Meta/Fake inference, invalid
  metadata, actual W4A8 method calls, graph replay and workspace boundaries.
- New boundary tests use capacity 1536, K=7168, N=4096, both group-list formats,
  both compensation modes and two calls in one graph. Routes transition across
  zero, one and two chunks. CPU reference rows include both sides of the chunk
  boundary and expert boundaries.
- CPU reference thresholds remain scale rtol=1e-3/atol=1e-5; maximum INT8
  difference 1; differing INT8 fraction below 2%; dequantized output
  rtol=2e-2/atol=2e-2.
- The stock BF16 split reference has its own predeclared bounds: maximum INT8
  difference 2, fraction above one below 0.5%, scale rtol=2e-2/atol=1e-5,
  dequantized output rtol=3e-2/atol=2e-2. Additional cases observed at most two
  INT8 units and scale relative error below 0.73%. Thresholds were not relaxed.
- A3 custom OPP rebuilt successfully; the full framework extension was built
  during integration. The optimization only changes the A3 kernel and tiling.
- A5 original native suite: **4 passed**, including both group-list formats
  and dynamic route graph replay with byte-identical output and scales.
- CPU-isolated W4A8/device unit tests: **147 passed**. NPU backend autoload and
  the `npu-smi` probe are disabled only in this CPU mock test process.
- Formatter coverage and binary/source identities are recorded in the
  accompanying validation report and PR.

CANN 9.2.1 injects `round_mode=rint` into the existing A5 `SituMxQuant` reference
compiler wrapper, causing an argument-count failure. For the reference build
only, the generated wrapper drops that default argument. Neither the reference
compute kernel nor delivered source is changed by this test-build workaround.

Full K3.1 distributed serving accuracy, throughput and long-duration stability
remain untested. Required repository CI must pass before merge.

## Reproduction

Build both `grouped_matmul_situ_quant` and `dequant_situ_quant`, and rebuild the
framework extension. Point `ASCEND_CUSTOM_OPP_PATH` at the selected isolated
OPP package, then run from the checkout:

```bash
export OMP_NUM_THREADS=8 ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH="$PWD"
python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_gmm_situ_quant_msd.py
python benchmarks/benchmark_gmm_situ_quant_msd.py --out small.json
python benchmarks/benchmark_gmm_situ_quant_msd.py --counts 32,32,64 --out medium.json
python benchmarks/benchmark_gmm_situ_quant_msd.py --as-list --out list.json
python benchmarks/benchmark_gmm_situ_quant_msd.py --capacity 2048 --out padded.json
python benchmarks/benchmark_gmm_situ_quant_msd.py --profile profile-small
python benchmarks/benchmark_gmm_situ_quant_msd.py --metric L2Cache --profile profile-cache
```

The script depends on the native test fixture in this checkout. Keep raw device
CSVs and JSON samples when comparing another code state; do not substitute
profiler durations or Python enqueue time for graph execution latency.
