# A5 serial compressor

The A5 device path follows the public Compressor mathematical specification:

```text
AscendC cast -> torch.matmul -> AscendC weighted softmax pooling
                            -> AscendC RMSNorm + RoPE + cache update
```

The two projections share one matmul with concatenated static weights. Input
casting, matmul and both vector tasks execute on every invocation. Projections
and the persistent cache are FP32. Preparation converts weights and allocates
scratch buffers before capture. Pooling finishes before any cache write, so
current tokens cannot overwrite history still read by another core. Every valid
token updates the cache, including tokens outside the last compression window.

## Measured status

Ten optimization rounds completed on Ascend950DT_9581V120, architecture 3510,
with CANN 9.2, PyTorch 2.10 and matching torch_npu. All six exported workload
classes passed CPU formula, fused black-box and Graph replay comparisons in
every valid round. Four additional edge cases cover FP16/BF16, BSH/TND,
paged/ring caches, both RoPE modes, optional metadata and cache stride padding.
Independent ASC compilation, parent-project CMake compilation, installation and
automatic extension loading passed. A complete vLLM wheel build was not run.

The requested fused/serial >= 0.8 target **was not achieved for all cases**.
Final complete Graph task sums reach 0.811 for prefill r128/D512 and 0.805 for
decode r4/D128. Three other stable cases reach 0.407 to 0.697. Decode r4/D512
has excessive fused sampling variation in both verification attempts and has
no valid performance acceptance result. Its observed ratios are retained.
See the [full measurement report](../../../benchmarks/results/compressor_serial_a5/20261026/README.md)
for raw exports, every round, exact precision settings and C/(C+V).

The extension uses the repository's `_C_ascend` registration namespace and
`op_host/`, `op_kernel/arch35/` layout. It is a direct ASC Torch extension with
a prepared Python execution plan. The main A5 CMake build includes its shared
library. It does not provide an aclnn vendor-package interface or change model
dispatch; use `CompressorSerialPlan` explicitly.

## Build and call

Use an A5 development container with CANN 9.2 and its ASC compiler, PyTorch 2.10
and matching torch_npu. Verify architecture 3510. From the repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash tools/build_compressor_serial.sh
cmake --install build/compressor_serial --prefix "$PWD"
```

The install command places `libcompressor_serial_ops.so` in `vllm_ascend/`.
`prepare` loads it automatically. Alternatively, explicitly load the library
from the build directory before preparing a plan.

```python
import torch
import torch_npu
from vllm_ascend.ops.compressor_serial import CompressorSerialPlan

plan = CompressorSerialPlan.prepare(
    x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
    state_block_table, cu_seqlens, seqused, start_pos,
    cmp_ratio=4, coff=2,
)
out = plan(x)
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    out = plan(x)
graph.replay()
```

Use one plan per concurrent stream. Its output is reused. Rebuild the plan when
weights, shapes, dtype or device change. Device metadata may be updated in
place between replays. The caller supplies valid sequence lengths, nonnegative
start positions, sufficient output rows and a ring buffer of at least
`coff * ratio + max_sequence_length - 1` rows. Metadata arithmetic and all buffer
offsets must fit signed 32-bit indices. Do not modify the internal parameter
tensor. Only stride(0) cache padding is supported. Weights remain immutable.

The plan does not change process-wide matmul precision flags. The final
benchmark explicitly sets `allow_hf32=True` and `cube_math_type=USE_HF32` before
preparation/capture for both implementations. Revalidate accuracy if changing
precision. Default launch choices depend on token count; explicit
`CompressorSerialConfig` overrides them.

## Precision and Graph benchmark

The fixtures preserve six tensor-shape classes exported by Zhanlu for
DeepSeek-V4-Flash, with node multiplicities 21, 21 and 20 per phase. Sequence
lengths, starts and page mappings are synthetic, rather than a model trace.
The installed fused binary requires TND output capacity derived from X. For
prefill r4, runtime RoPE capacity is therefore 264 rows instead of the exported
175; there are still 175 valid output rows. Original and runtime shapes are
recorded separately. These results are not full-model throughput measurements.

```bash
python -m pytest --noconftest     tests/ut/ops/test_compressor_serial_contract.py     tests/ut/ops/test_compressor_serial_profile.py -q
python -m pytest --noconftest tests/ut/ops/test_compressor_serial_npu.py -q
python -m benchmarks.benchmark_compressor_serial     --serial-library build/compressor_serial/libcompressor_serial_ops.so     --fused-library /path/to/installed/vllm_ascend_C.so     --iteration 10 --hypothesis final-verification     --hf32 --cube-math-type USE_HF32 --auto-launch     --output results/fresh-verification.json
```

The benchmark checks outputs and the entire cache against the CPU oracle, then
the installed fused operator as a black box, and finally captured Graph replay.
A failed precision check aborts measurement. Fused implementation source is not
required. Use a fresh output path; recorded evidence is never overwritten.

Event timing reports complete Graph latency as diagnostic information. For
device task timing and C/(C+V), profile one case per application with
`--profile serial` or `--profile fused`. This emits exactly ten adjacent pairs
of ArgMax on 256 MiB FP16 data and complete Graph replay. Retain every sample,
sum every replay task before taking the median, and reject performance
acceptance if `(max - min) / median > 0.10` on either path. C is the complete
matmul task duration; V includes cast, pooling, normalization, RoPE and cache
update. C/(C+V) is the median of the ten per-replay fractions. It is a task-time
split, not Cube hardware pipe utilization. Whole mixed fused kernels are not
classified as pure Cube time.
