# MsaIndexScore

## Product Support

| Product                                                               | Supported |
| --------------------------------------------------------------------- | :-------: |
| <term>Atlas A2 Products</term>                                       |     √     |
| <term>Atlas A3 Products</term>                                       |     √     |
| <term>950PR&950DT Products</term>                                    |     √     |

## Function Description

`MsaIndexScore` computes block scores for the Index Branch of MiniMax Sparse
Attention (MSA). For each query token and sparse KV block, it performs matrix
multiplication followed by max pooling over all causally visible tokens in the
block. The scores are consumed by the subsequent TopK stage. Prefill and decode
use the same interface.

The non-quantized and INT8-quantized paths are:

$$
score = Maxpool[Q_{idx}@K_{idx}^{T}]
$$

$$
score = Maxpool[scale \cdot Q_{idx}@K_{idx}^{T}]
$$

The complete formula is:

$$
score = Maxpool[(scale \cdot)Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
$$

Max pooling reduces the KV-token dimension within each sparse block of length
$block\_size$. `start_loc`, `init_blocks`, and `local_blocks` generate
$local\_mask`, which assigns high scores to leading blocks and blocks around the
current query so that TopK always retains them. Set both block-count attributes
to 0 to disable this behavior and match the Triton raw-score kernel.
When the two windows overlap, the local-window score (`1e29`) overrides
the leading-block score (`1e30`).

## Parameters

Notation:

- B is the batch size.
- S1 and S2 are the query and key sequence lengths.
- T1 and T2 are the sums of query and key sequence lengths across the batch.
- N1 and N2 are the query-head and key-head counts.
- D is the head dimension.
- In PageAttention, `block_num` is the number of physical pages, `block_size`
  is the token count per page, and `maxBlockNumPerSeq` is the width of
  `block_table`.

| Parameter | ACLNN Name | Kind | Description | Data Type | Format |
| --------- | ---------- | ---- | ----------- | --------- | ------ |
| `query` | `query` | Input | Query tensor in TND layout, shape `[T1, N1, D]`. | BFLOAT16, FLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | ND |
| `key` | `key` | Input | Key tensor in TND `[T2, N2, D]`, BNBD `[block_num, N2, block_size, D]`, or BBND `[block_num, block_size, N2, D]` layout. | BFLOAT16, FLOAT16, INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | ND |
| `block_table` | `blockTableOptional` | Optional input | PageAttention logical-block-to-physical-page mapping, shape `[B, maxBlockNumPerSeq]`. Required for BBND and BNBD. | INT32 | ND |
| `scale` | `scaleOptional` | Optional input | INT8 dequantization scale. PageAttention shape: `[block_num, N2, block_size]` or `[block_num, block_size, N2]`; TND shape: `[T2, N2]`, or `[T2]` when N2 is 1. | FLOAT | ND |
| `atten_mask` | `attenMaskOptional` | Optional input | Base causal mask used by `sparse_mode=3`, shape `[2048, 2048]`. A value of 1 excludes a position and 0 includes it. | INT8 | ND |
| `actual_seq_qlen` | `actualSeqQlenOptional` | Input | Non-decreasing query prefix sums, shape `[B+1]`. | INT32 | ND |
| `actual_seq_klen` | `actualSeqKlenOptional` | Input | TND key prefix sums `[B+1]`, or visible key lengths `[B]` for PageAttention. | INT32 | ND |
| `start_loc` | `startLoc` | Input | Logical-block index containing the current query, shape `[B]`. | INT32 | ND |
| `layout_key` | `layoutKeyOptional` | Attribute | Key layout: `"TND"`, `"BBND"`, or `"BNBD"`. The aclnn parameter is `layoutKeyOptional` and defaults to `"BBND"`. | STRING | - |
| `sparse_mode` | `sparseMode` | Attribute | 0: `defaultMask`; 3: `rightDownCausal`. | INT64 | - |
| `init_blocks` | `initBlocks` | Attribute | Number of leading blocks assigned `1e30`. Default: 0. | INT64 | - |
| `local_blocks` | `localBlocks` | Attribute | Size of the local window `[max(0, start_loc+1-local_blocks), start_loc]`, assigned `1e29`. Default: 1. | INT64 | - |
| `score` | `score` | Output | Block scores, shape `[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]`. | FLOAT | ND |

The defaults above describe the operator schema. The ACLNN C++ calls take
explicit attribute arguments. The vLLM binding in
[msa_index_score_torch_adpt.h](./msa_index_score_torch_adpt.h) supports the
BBND non-INT8 path and uses `init_blocks=0, local_blocks=0` by default, as
registered in [torch_binding.cpp](../../torch_binding.cpp). Keep both at
zero for TP-sharded scoring, where the subsequent TopK stage applies global
block forcing.

## Constraints

- Only `block_size=128` is supported.
- `layout_key` must match the actual shape of `key`.
- PageAttention requires `block_table`. TND must omit `block_table` and use
  `[B+1]` prefix sums for `actual_seq_klen`.
- On the non-quantized path, `query` and `key` must have the same dtype and
  `scale` must be absent. A2/A3 support FLOAT16 and BFLOAT16. 950PR&950DT Products also
  supports HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN.
- The quantized path supports a FLOAT16 query, an INT8 key, and a required
  FLOAT scale. Native FP8 is not an INT8 quantized path: on 950PR&950DT Products, query
  and key must use the same FP8 dtype and `scale` must be absent.
- `sparse_mode=0` requires no `atten_mask`. `sparse_mode=3` requires an INT8
  mask with shape `[2048, 2048]`.
- `init_blocks` and `local_blocks` must be non-negative and cannot exceed the
  logical block width. Setting both to 0 disables `local_mask`.
- `block_table` may be wider than the actual logical KV block count. The score
  width is `RoundUp(block_table.shape[1], 16)`. The 950PR&950DT Products C2UB path flushes
  widths greater than 256 in 256-column windows.
- `q_len` and `kv_len` may be zero, including for the entire batch. Empty query
  requests skip QK computation; empty KV requests produce fill scores; an
  all-empty query batch launches with one block.
- PageAttention BBND/BNBD keys may be non-contiguous on the physical-page axis
  on A2/A3 and 950PR&950DT Products. All inner axes must remain contiguous. TND keys must
  be contiguous, and `scale` remains tightly packed by logical page.
- A2/A3 and 950PR&950DT Products size the MIX launch from the estimated M-task count.
  When short-M decode cannot fill the 950PR&950DT Products AICs and spans multiple visible
  KV S-tiles, `kvChunks` partitions the visible KV range across additional MIX
  tasks. Short-KV and wide-table inputs keep a single KV chunk.
- The operator returns block scores only and does not perform TopK.

## ACLNN Interface

### Function Prototypes

This operator uses a two-stage interface. Call
`aclnnMsaIndexScoreGetWorkspaceSize` to validate the inputs and obtain an
executor and the required workspace size. Then call `aclnnMsaIndexScore` on the
same stream context to execute the computation.

```cpp
aclnnStatus aclnnMsaIndexScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *blockTableOptional,
    const aclTensor *scaleOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *actualSeqQlenOptional,
    const aclTensor *actualSeqKlenOptional,
    const aclTensor *startLoc,
    char            *layoutKeyOptional,
    int64_t          sparseMode,
    int64_t          initBlocks,
    int64_t          localBlocks,
    const aclTensor *score,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);

aclnnStatus aclnnMsaIndexScore(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

### Workspace Query Outputs

The input tensors and attributes of `aclnnMsaIndexScoreGetWorkspaceSize`
are described in [Parameters](#parameters). The caller supplies `score`
with the documented output shape and receives:

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `workspaceSize` | `uint64_t*` | Required device workspace size in bytes. |
| `executor` | `aclOpExecutor**` | Executor passed to the second-stage call. |

### Workspace Query Return Values

| Return Code | Error Code | Description |
| ----------- | ---------- | ----------- |
| `ACLNN_SUCCESS` | 0 | Validation succeeded. |
| `ACLNN_ERR_PARAM_NULLPTR` | 161001 | A required input or output is null. |
| `ACLNN_ERR_PARAM_INVALID` | 161002 | A dtype, format, dimension, stride, or value violates a constraint. |

### Execution: aclnnMsaIndexScore

#### Parameters

| Parameter | Kind | Description |
| --------- | ---- | ----------- |
| `workspace` | Input | Device workspace address. |
| `workspaceSize` | Input | Workspace size returned by the first-stage interface. |
| `executor` | Input | Executor returned by the first-stage interface. |
| `stream` | Input | ACL stream used to execute the operator. |

#### Return Values

| Return Code | Error Code | Description |
| ----------- | ---------- | ----------- |
| `ACLNN_SUCCESS` | 0 | Execution succeeded. |
| `ACLNN_ERR_PARAM_INVALID` | 161002 | A parameter is invalid. |

### Invocation Example

The following excerpt shows the two-stage invocation for a non-quantized BBND
PageAttention input. See
[test_aclnn_msa_index_score.cpp](./examples/test_aclnn_msa_index_score.cpp)
for complete BBND, BNBD, TND, INT8, FP8, empty-sequence, strided-page, and wide
block-table accuracy cases.

```cpp
static char layoutKey[] = "BBND";
int64_t sparseMode = 3;
int64_t initBlocks = 0;
int64_t localBlocks = 1;
void *workspace = nullptr;
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;

aclnnStatus ret = aclnnMsaIndexScoreGetWorkspaceSize(
    queryTensor,
    keyTensor,
    blockTableTensor,
    nullptr,  // scaleOptional: null for non-quantized input
    attenMaskTensor,
    actualSeqQlenTensor,
    actualSeqKlenTensor,
    startLocTensor,
    layoutKey,
    sparseMode,
    initBlocks,
    localBlocks,
    scoreTensor,
    &workspaceSize,
    &executor);

if (ret == ACLNN_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
if (ret == ACLNN_SUCCESS) {
    ret = aclnnMsaIndexScore(workspace, workspaceSize, executor, stream);
}
```

For BNBD, set `layoutKey="BNBD"` and use a
`[block_num, N2, block_size, D]` key. For TND, set `layoutKey="TND"`, omit
`blockTableOptional`, and provide `[B+1]` key-length prefix sums.

Keep attribute storage alive through execution and graph capture. The
complete example also handles synchronization and resource cleanup.

## Implementation Notes

- `layout_key` selects PageAttention BBND/BNBD or packed TND. TND does not use
  `block_table`.
- For `sparse_mode=3`, the host validates `atten_mask[2048,2048]`; the device
  derives right-down-causal visibility without loading the mask element by
  element.
- The 950PR&950DT Products implementation is under `op_kernel/arch35`. It uses native
  Cube FP8 tiling keys 4/5/6 without a scale or an intermediate FP16 cast.
- 950PR&950DT Products uses the operator-private Catlass snapshot under
  `op_kernel/catlass`, derived from v1.3.1-notla. A2/A3 continue to use the
  repository Catlass submodule. The `msa_` prefix isolates only the A5-specific
  snapshot because its interfaces and implementation differ.
- For 950PR&950DT Products short-M/long-KV decode, host tiling derives `kvChunks` from
  visible KV S-tiles. Both arch22 and arch35 schedulers split the S range and
  only the final chunk writes the aligned tail fill.

## Testing

### Standalone ACLNN Build and Run

Run these commands from the operator build tree (`csrc/` in this checkout),
with the matching CANN build environment prepared. Use the generated `.run`
package for your host architecture; the filenames below illustrate x86_64.
These are standalone numerical checks, not model-level GPQA evaluation.

For Atlas A2/A3:

```bash
bash build.sh --pkg --soc=ascend910b --ops=msa_index_score -j32
bash ./build_out/cann-ops-transformer-custom_linux-x86_64.run \
  --quiet --install-path=/tmp/msa_opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust \
  --vendor_name=custom --soc=ascend910b
```

For 950PR&950DT Products:

```bash
bash build.sh --pkg --soc=ascend950 --ops=msa_index_score -j32
bash ./build_out/cann-ops-transformer-custom_linux-x86_64.run \
  --quiet --install-path=/tmp/msa_opp
source /tmp/msa_opp/vendors/custom_transformer/bin/set_env.bash
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust \
  --vendor_name=custom --soc=ascend950
```

The expected result is 50/50 cases on 950PR&950DT Products. A2/A3 skip the ten FP8
cases and run 40 cases. See [Acceptance Criteria](#acceptance-criteria) for tolerances.

### Standalone Test Matrix

`start_loc` is a logical-block index, and `sparse_mode=3` applies
right-down-causal masking.

| Test Case | Scenario | Coverage |
| --------- | -------- | -------- |
| `L0-debug-trace` | Minimal dimensions | Main path and trace |
| `L0-int8-dequant-trace` | INT8 key with scale | Fused dequantization |
| `L0-prefill-aligned` | Aligned chunked prefill | Causal and local masks |
| `L1-prefill-unaligned` | Variable-length batch | Boundary-block mask |
| `L1-prefill-multi-mtile` | Row count greater than M-tile | M-tile partitioning |
| `L1-decode-lq1` | Decode with `q_len=1` | Multiple sequence lengths |
| `L1-decode-speculative` | Decode with `q_len>1` | Speculative decoding |
| `L1-long-seq-multi-stile` | `kv_len=4096` | Multiple S-tiles |
| `L1-bf16` / `L1-int8-dequant` | Data type | Non-quantized and quantized paths |
| `L2-tiny-kv` | Minimal KV length | Tail padding |
| `L1-bnbd` / `L1-bnbd-int8` | PageAttention BNBD | `[NP, N2, P, D]` layout |
| `L1-tnd-unaligned` / `L1-tnd-int8` / `L0-tnd-tiny` | Packed TND | No block table and key-length prefix sums |
| `L0-fp8-e4m3fn` / `L0-fp8-e5m2` / `L1-fp8-e4m3fn-prefill` | 950PR&950DT Products FP8 | Native E4M3FN/E5M2 Cube paths; HIFLOAT8 is kernel-only |
| `L1-pad-q0` / `L1-pad-kv0` | Empty request in a mixed batch | Skip empty query or fill empty KV scores |
| `L1-pad-q0-kv0` / `L1-tnd-pad-q0-kv0` / `L1-pad-mid-q0` | Empty request at an edge or in the middle | PageAttention and TND padding |
| `L0-all-q0` / `L1-all-q0` | Entire batch has `q_len=0` | Host acceptance and skipped computation |
| `L0-all-kv0` / `L0-all-q0-kv0` | Entire batch has empty KV | Fill scores and fully empty input |
| `L0-tnd-all-q0` / `L0-tnd-all-kv0` / `L0-tnd-all-q0-kv0` | Empty packed TND batch | Empty query and key tensors |
| `L0-stride-bbnd` / `L1-stride-bbnd` / `L1-stride-bnbd` | Page axis has a gap of two | Non-contiguous physical-page addressing |
| `L1-stride-int8` | INT8 page axis has a gap of two | Quantized page copy with a stride |
| `L0-wide-table-257` / `L1-wide-table-257-bf16` | Block-table width 257 | 950PR&950DT Products C2UB windowed flush |
| `L0-fp8-wide-table-257` | Width 257 with FP8 | Aligned score width and fill positions |
| `L0-decode-q4-kv4` / `L0-decode-q4-kv4-b2` | Short decode with four query heads | MIX launch sized from estimated M tasks |
| `L0-decode-q4-kv4-table275` | Short decode with a wide block table | Aligned score tail and windowed flush |
| `L0-decode-q4-kv275` | Long-KV decode with 275 visible pages | 950PR&950DT Products KV S-range splitting |
| `L0-fp8-decode-*` | FP8 short and long-KV decode | E4M3FN/E5M2 compact, wide-table, and KV-split paths |

The full matrix runs by default. The key layout is selected by `layout_key`
(`layoutKeyOptional` in aclnn) and is not inferred from tensor rank.

### CPU Reference

The NumPy CPU reference is in
[msa_index_score_golden.py](./tests/golden/msa_index_score_golden.py):

```python
inputs = MsaIndexScoreGoldenInputs(
    query=query,
    key=key,
    block_table=block_table,
    actual_seq_qlen=actual_seq_qlen,
    actual_seq_klen=actual_seq_klen,
    start_loc=start_loc,
    sparse_mode=3,
    scale=None,
)
score = msa_index_score_golden(inputs)
```

### vLLM Single-Operator Precision Tests

From the vllm-ascend repository root, with the custom operator installed, run:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_msa_index_score.py
```

The [nightly test](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_msa_index_score.py)
uses a self-contained CPU FP32 reference. Its eight scenarios cover prefill,
decode, non-contiguous page axes, long KV with a wide block table, wide-table
padding, empty query/KV requests, forced blocks, and dense TP chunks. Each is
parameterized over FLOAT16, BFLOAT16, and FLOAT8_E4M3FN, giving 24 cases. FP8
cases skip when `HardwareCapability.FP8_ATTENTION` is unavailable.

The existing [PR operator test](../../../tests/e2e/pull_request/one_card/test_msa_index_score.py)
also compares operator outputs against the NumPy reference. The
[MiniMax unit tests](../../../tests/ut/models/minimax_m3/test_msa_m3.py)
cover integration and dispatch behavior. These checks do not need model
weights and do not measure GPQA answer accuracy.

### Acceptance Criteria

- Masked and padded score positions must match the reference fill positions.
- Blocks forced by `local_mask` must be at least `1e28` on both sides.
- FLOAT16, BFLOAT16, and INT8 standalone checks use `atol=rtol=1e-3`;
  FP8 uses `atol=rtol=2e-2`. The C++ example requires every ordinary score to
  meet its tolerance and rejects mismatched fill or forced-block positions.
- The NumPy reference's optional `compare` helper allows an error ratio no
  greater than `1e-3`, with an error threshold of
  `atol + rtol * max(abs(golden), 1)` for ordinary score values.
- The nightly test checks fill/forced-block masks exactly and uses
  `torch.testing.assert_close` for every remaining score, with `atol=rtol=1e-3`
  for FLOAT16/BFLOAT16 and `2e-2` for FP8. It uses inputs exactly representable
  in the tested dtypes to isolate computation and scheduling differences.

## References

- [Standalone ACLNN example](./examples/test_aclnn_msa_index_score.cpp)
- [Upstream ops-transformer implementation](https://gitcode.com/cann/ops-transformer/tree/master/attention/msa_index_score)
- [Upstream Python interface](https://gitcode.com/cann/ops-transformer/blob/master/attention/msa_index_score/docs/torchapi_msa_index_score.md)
