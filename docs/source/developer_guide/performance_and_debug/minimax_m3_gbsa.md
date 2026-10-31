# MiniMax-M3 sparse attention on A3

The A3 hardware profile selects CANN GenericBlockSparseAttention (GBSA).
A2, A5, and compatibility profiles retain the repository's SparseAttentionScore
(SAS) path. Its C++ source, build configuration, bindings, and fake/meta
registration remain available. This selection does not change index scoring.
CANN GBSA bindings are imported lazily only when the GBSA path executes.

## Decode preparation

For supported BF16 decode with 128-token blocks and uniform query lengths 1
or 4, the model-specific metadata builder prepares one GBSA schedule per
matching layer configuration before IndexScore. Layers share the schedule,
while their Q/K/V, selected blocks, counts, and block tables remain independent.
The schedule is refreshed every forward into persistent 1024-element int32
buffers. Uniform query prefixes also have persistent storage. FULL graph
replay uses the padded capture shape, including MRV2 request padding.
Other shapes and direct callers retain per-call preparation.

The A3 decode indexer requests ordered block IDs from its TopK postprocessor.
Filtering invalid IDs, computing valid counts, FP32 block ordering, and int32
conversion happen in one kernel. TopK=16 uses four rows per program and 32
internal sorting slots. Consumers declare the indices already sorted and avoid
repeating normalization. Block IDs must be exactly representable in FP32.
Neither `torch.topk` nor the CANN TopKV2 operator is modified.

Direct GBSA callers and the GBSA prefill fallback normalize only the valid
prefix selected by `select_num_idx`, preserve trailing `-1` padding, and order
logical block IDs. This prevents invalid final-page KV tails from affecting
partial-page attention. The optional split-KV prefill implementation is retained.

A5 keeps the original SAS operands, precision settings, FP8 conversion and
dequantization scales. Its decode indexer does not opt into GBSA ordering, and
its metadata builder allocates no GBSA schedule or uniform-prefix buffers.
The A5 Triton implementation is unchanged.

## Validation

CPU regressions cover backend dispatch, schedule lifetime, padded graph
shapes, selected-block ordering, and the retained A5 BF16/FP8 SAS ABI:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q \
    tests/ut/models/minimax_m3/test_msa_m3.py \
    tests/ut/device/test_hardware_profile.py
```

Run device correctness and graph-replay checks on A3:

```bash
pytest -q tests/e2e/pull_request/one_card/test_minimax_m3_sparse_attn.py \
    --msa-m3-sparse-backend=torch_npu
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_msa_m3_index_postprocess.py
```

Repeat the sparse-attention E2E file on A5 to exercise the retained SAS path.
GBSA-specific tests explicitly skip other profiles. Existing profiling of
the parent implementation does not replace hardware validation of a new head;
kernel timing alone does not establish full-model TPOT or throughput gains.
