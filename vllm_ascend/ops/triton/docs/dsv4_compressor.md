# DeepSeek V4 projected compressor

## Description

- **Function:** Compress adapter-aware FP32 KV and gate projections for the
  DeepSeek V4 main attention and indexer. The production caller is
  `models/deepseek_v4/compressor.py`; it selects this path when either projection
  has a LoRA wrapper. Models without compressor adapters retain the fused CANN
  compressor.
- **Formula:** For each completed compression group, apply a softmax over
  `score + APE` along its token axis, then sum the weighted KV values. C4 joins
  the previous group's first projection head and the current group's second
  head. C128 uses the current group. Apply FP32 RMSNorm and interleaved RoPE,
  then cast the result to the activation dtype.
- **Algorithm flow:** Pool current projections and cached leading history into
  FP32 scratch, normalize/rotate output, then write updated state. The three
  launches preserve history reads before writes. Each state row writes KV and
  scores together in one contiguous store. The pooling head tile is 32 features,
  bounding the C128 intermediate size.
- **Supported modes:** Atlas A2, Atlas A3, and 950PR/950DT; eager and ACLGraph.
  Actual device evidence is recorded in the accompanying PR.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `kv`, `scores` | Input | Adapter-aware projections `[tokens, coefficient * head_dim]` | FP32 | Contiguous ND |
| `state_cache` | Input/Output | Paged state `[pages, block_size, 2 * coefficient * head_dim]` | FP32 | ND; outer page stride may include padding |
| `ape` | Input | Positional score bias `[ratio, coefficient * head_dim]` | FP32 | Contiguous ND |
| `norm_weight` | Input | RMSNorm weight `[head_dim]` | FP32 | Contiguous ND |
| `rope_sin`, `rope_cos` | Input | Selected compressed-position RoPE `[output_rows, 64]` | FP32/BF16/FP16 | Contiguous ND |
| `state_block_table` | Input | Request-to-state-page mapping `[requests, logical_pages]` | INT32 | Contiguous rows |
| `cu_seqlens` | Input | Request query offsets `[requests + 1]` | INT32 | Contiguous ND |
| `start_pos` | Input | Request start positions `[requests]` | INT32 | Contiguous ND |
| `compress_ratio` | Attribute | Compression ratio, 4 or 128 | Integer | N/A |
| `norm_eps` | Attribute | Positive normalization epsilon | Float | N/A |
| `output_dtype` | Attribute | Output activation dtype | BF16/FP16 | N/A |
| Return value | Output | Compressed request-major rows, followed by zero padding | BF16/FP16 | Contiguous ND |

## Constraints

All tensors reside on the same NPU. `head_dim` is 128 or 512; production
combinations are C4/128, C4/512, and C128/512. C4 has coefficient 2; C128 has
coefficient 1. RoPE covers the final 64 features. The number of allocated output
rows comes from the RoPE metadata, as in the fused compressor. Metadata must
cover every completed group and every referenced logical page.

Query offsets are monotonic and bounded by the projection rows. A zero-length
query or negative start denotes padding. Page IDs refer to allocated state;
page zero denotes absent state and is never written. Existing history scores
already include APE. State writes store `[kv, scores + APE]` in FP32 and modify
only current query rows with nonzero page IDs. Other pages and page padding are
unchanged. Pooling never transfers device metadata to the CPU.

JIT specialization depends on head size, ratio, state block size, the next
power of two of request count, RoPE width, and pointer dtypes. Token count,
output count, strides, epsilon and request count use runtime scalars with value
specialization disabled. Changing adapter IDs or query metadata contents does
not specialize. This operator does not register explicit JIT warmup. Before
capturing each graph descriptor, the model runner's normal capture warmup runs
the same forward path with dummy LoRAs and graph mode disabled. This exercises
the selected compressor wrapper before capture. Different request-count buckets
may compile additional variants.

## Origin and Differences

- **Origin:** New Ascend implementation of the C4/C128 pooling equations used
  by the existing fused `compressor` operator.
- **Differences:** Accepts precomputed FP32 projections so the model's LoRA
  wrappers can apply each token's adapter before the nonlinear softmax. Uses
  Ascend paged state and supplied compressed RoPE tables. Unlike the V4.1 C2
  ring compressor, it supports C4 overlap and C128 paged history.

## Test Cases

The independent CPU reference explicitly gathers each compression group,
applies PyTorch softmax, RMSNorm and RoPE, and updates a separate state copy.
Cases cover all three production shapes, BF16/FP16 outputs, chunk boundaries,
leading history, padded requests, absent pages, padded page strides, empty
inputs, C4 prefill with 384 tokens to cover multiple programs per core, and
ACLGraph replay after changing projection values. State updates and
untouched page padding are compared exactly. BF16 output uses `rtol=atol=2^-6`;
FP16 output uses `rtol=atol=2^-9`.

The compressor integration test uses actual LoRA wrappers and NPU Punica,
mixes two nonzero adapters with a base request, and checks both projections of
the main and indexer paths. It also replays changed adapter mappings under
ACLGraph. FP32 projected state uses `rtol=2^-10, atol=2^-16`.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dsv4_compressor.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dsv4_compressor_lora.py
```

## Performance measurement

Run the module benchmark on one NPU from the repository root:

```bash
python benchmarks/benchmark_dsv4_compressor.py
```

The benchmark uses BF16 TP1, hidden size 4096, three requests of 128 tokens,
and all three production compressor shapes. It checks unchanged fused outputs
without LoRA, compares zero-adapter outputs with the fused operator, and confirms
that nonzero adapters affect only their requests. It reports synchronized module
wall-clock latency after ten warmup calls and thirty measured calls. The
adapter-aware path includes unfused FP32 projections, Punica, and pooling;
it costs more than the original fused path. These measurements do not represent
full-model throughput or latency.
