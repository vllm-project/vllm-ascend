# A5 serial compressor

This development implementation follows the public Compressor mathematical
specification. Its device path is:

```text
AscendC input cast -> torch.matmul -> AscendC weighted pool
                                  -> AscendC RMSNorm + RoPE + cache update
```

The two projections share one matmul with statically concatenated weights.
The matmul produces FP32 intermediates. Preparation caches converted weights
and allocates scratch tensors before graph capture. Each invocation includes
input casting, matmul and both vector tasks. Pool and cache update have a task
boundary so that one core cannot overwrite history still read by another.

## Status

The implementation is under development. CPU formula checks pass; AscendC
compilation, NPU correctness, graph replay correctness and performance acceptance
are pending. No performance ratio or completed optimization round is claimed.
The intended acceptance criterion is fused time / serial time >= 0.8 for every
reported case, with ten measured optimization rounds even if the target is
reached earlier. Environment setup and compiler repairs do not count as rounds.

The extension uses `_C_ascend` Torch registration and the `op_host/`,
`op_kernel/arch35/` layout. It is a separate direct ASC extension; it is not yet
an aclnn vendor-package operator. Existing model dispatch is unchanged.

## Build and call

Use an A5 development container with CANN 9.2, its ASC compiler, PyTorch 2.10
and matching torch_npu. Verify the full SoC and NPU architecture 3510 before
execution. From the repository root, after sourcing the CANN environment:

```bash
bash tools/build_compressor_serial.sh
```

```python
import torch
import torch_npu
from vllm_ascend.ops.compressor_serial import CompressorSerialPlan

torch.ops.load_library("build/compressor_serial/libcompressor_serial_ops.so")
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

Use one plan per concurrent stream. Its output is reused on every call. Rebuild
the plan when weights, shapes, dtype or device change. Device metadata may be
updated in place between replays. The caller is responsible for valid sequence
lengths, nonnegative start positions, sufficient output capacity and a ring
buffer of at least `coff * ratio + max_sequence_length - 1` elements. Do not
modify the internal parameter tensor. Only stride(0) cache padding is supported.
The module does not change global HF32 flags; benchmark results must record the
chosen flag. Prepared weight conversion is excluded from steady state timing.

## Precision and graph benchmark

The six fixtures preserve the tensor shapes exported by the official Zhanlu
tool for DeepSeek-V4-Flash, with three distinct compressor shapes in each of
prefill and decode. Node multiplicities are 21, 21 and 20. These are template
expansions, not measured NPU results. The exported CSV does not establish runtime
metadata values; benchmark sequence lengths, starts and page mappings are
synthetic and explicitly recorded as such.

```bash
python -m pytest tests/ut/ops/test_compressor_serial_contract.py -q
python -m benchmarks.benchmark_compressor_serial \
    --serial-library build/compressor_serial/libcompressor_serial_ops.so \
    --fused-library /path/to/installed/vllm_ascend_C.so \
    --iteration 1 --hypothesis 'Initial serial baseline' \
    --output results/iteration-01.json
```

The benchmark checks output and the entire cache against a CPU formula oracle,
then compares with the installed fused operator as a black box, then checks graph
replay. A failed precision check aborts performance acceptance. The source of
the fused kernel is not needed by this workflow.

Event timing reports graph latency samples in microseconds. For task timing and
`C/(C+V)`, run one case per `msprof` application with `--profile serial` or
`--profile fused`; this emits exactly ten adjacent ArgMax warmup / graph replay
pairs. Retain the raw profiler export. Sum every task of each replay before
taking the median. For the serial path, C is measured matmul task time and V
includes the input cast, pooling, normalization, RoPE and cache-update tasks.
Do not estimate C/V by timing isolated stages or by subtracting graph latencies.
Mixed fused kernels require pipe metrics for a separate analysis; their whole
duration cannot be counted as pure Cube time.
