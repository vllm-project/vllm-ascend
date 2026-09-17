# aclnnMsaIndexScore

[View upstream source](https://gitcode.com/cann/ops-transformer/tree/master/attention/msa_index_score)

## Product Support

| Product                                                               | Supported |
| --------------------------------------------------------------------- | :-------: |
| <term>Atlas A2 Training Series/Atlas A2 Inference Series</term>       |     √     |
| <term>Atlas A3 Training Series/Atlas A3 Inference Series</term>       |     √     |
| <term>Ascend 950PR/Ascend 950DT</term>                                |     √     |

## Function Description

`aclnnMsaIndexScore` computes per-block importance scores for the Index Branch
of MiniMax Sparse Attention (MSA). For every query token and sparse KV block,
the operator performs matrix multiplication and max pooling over the causally
visible tokens in that block. It optionally dequantizes an INT8 key. The result
is consumed by a subsequent TopK operation; TopK itself is not part of this
operator.

The complete formula is:

$$
score = Maxpool[(scale \cdot)Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
$$

`local_mask` is generated from `startLoc`, `initBlocks`, and `localBlocks`.
Logical blocks in `[0, initBlocks)` receive `1e30`. Blocks in
`[max(0, startLoc+1-localBlocks), startLoc]` receive `1e29`, overriding an
init-block score at the same position. Setting both block-count attributes to
zero disables `local_mask`.

Notation used below:

- B is the batch size.
- S1 and S2 are the query and key sequence lengths.
- T1 and T2 are the sums of query and key lengths across the batch.
- N1 and N2 are the query-head and key-head counts.
- D is the head dimension.
- `block_num` is the number of physical PageAttention pages.
- `maxBlockNumPerSeq` is the width of `blockTableOptional`.

## Function Prototypes

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

## aclnnMsaIndexScoreGetWorkspaceSize

### Parameters

| Parameter | Kind | Description | Data Type | Format | Shape |
| --------- | ---- | ----------- | --------- | ------ | ----- |
| `query` | Input | Query in TND layout. | BFLOAT16, FLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | ND | `[T1, N1, D]` |
| `key` | Input | Key in TND, BNBD, or BBND layout. | BFLOAT16, FLOAT16, INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | ND | `[T2, N2, D]`, `[block_num, N2, block_size, D]`, or `[block_num, block_size, N2, D]` |
| `blockTableOptional` | Optional input | Logical-block-to-physical-page mapping. Required for PageAttention and omitted for TND. | INT32 | ND | `[B, maxBlockNumPerSeq]` |
| `scaleOptional` | Optional input | INT8 dequantization scale. Pass `nullptr` for non-quantized and native FP8 inputs. | FLOAT | ND | PageAttention: `[block_num, N2, block_size]` or `[block_num, block_size, N2]`; TND: `[T2, N2]` or `[T2]` when N2 is 1 |
| `attenMaskOptional` | Optional input | Base causal mask for `sparseMode=3`. A value of 1 excludes a position and 0 includes it. | INT8 | ND | `[2048, 2048]` |
| `actualSeqQlenOptional` | Input | Non-decreasing query-length prefix sums. | INT32 | ND | `[B+1]` |
| `actualSeqKlenOptional` | Input | TND key-length prefix sums, or visible key lengths for PageAttention. | INT32 | ND | TND: `[B+1]`; PageAttention: `[B]` |
| `startLoc` | Input | Logical-block index containing the current query. | INT32 | ND | `[B]` |
| `layoutKeyOptional` | Attribute | `"TND"`, `"BBND"`, or `"BNBD"`. Defaults to `"BBND"` when omitted or empty. | CHAR* | - | - |
| `sparseMode` | Attribute | 0 selects `defaultMask`; 3 selects `rightDownCausal`. | INT64 | - | - |
| `initBlocks` | Attribute | Number of leading blocks assigned `1e30`. Default: 0. | INT64 | - | - |
| `localBlocks` | Attribute | Local-window length assigned `1e29`. Default: 1. | INT64 | - | - |
| `score` | Output | Per-block importance scores. | FLOAT | ND | `[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]` |
| `workspaceSize` | Output | Required workspace size in bytes. | uint64_t | - | - |
| `executor` | Output | Operator executor returned by the first-stage interface. | aclOpExecutor** | - | - |

### Return Values

| Return Code | Error Code | Description |
| ----------- | ---------- | ----------- |
| `ACLNN_SUCCESS` | 0 | Validation succeeded. |
| `ACLNN_ERR_PARAM_NULLPTR` | 161001 | A required input or output is null. |
| `ACLNN_ERR_PARAM_INVALID` | 161002 | A dtype, format, dimension, stride, or value violates a constraint. |

## aclnnMsaIndexScore

### Parameters

| Parameter | Kind | Description |
| --------- | ---- | ----------- |
| `workspace` | Input | Device workspace address. |
| `workspaceSize` | Input | Workspace size returned by the first-stage interface. |
| `executor` | Input | Executor returned by the first-stage interface. |
| `stream` | Input | ACL stream used to execute the operator. |

### Return Values

| Return Code | Error Code | Description |
| ----------- | ---------- | ----------- |
| `ACLNN_SUCCESS` | 0 | Execution succeeded. |
| `ACLNN_ERR_PARAM_INVALID` | 161002 | A parameter is invalid. |

## Constraints

- Only `block_size=128` is supported.
- `layoutKeyOptional` must match the key shape. BBND is
  `[block_num, block_size, N2, D]`, BNBD is
  `[block_num, N2, block_size, D]`, and TND is `[T2, N2, D]`.
- PageAttention requires `blockTableOptional`. TND requires a null
  `blockTableOptional` and `[B+1]` prefix sums in `actualSeqKlenOptional`.
- For non-quantized input, query and key must use the same dtype and
  `scaleOptional` must be null. A2/A3 support FLOAT16 and BFLOAT16. Ascend 950
  additionally supports HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN.
- The quantized path supports a FLOAT16 query, an INT8 key, and a required
  FLOAT `scaleOptional`. Native FP8 is a non-quantized Ascend 950 path: query
  and key must use the same FP8 dtype and `scaleOptional` must be null.
- `sparseMode=0` requires a null `attenMaskOptional`. `sparseMode=3` requires
  an INT8 `[2048, 2048]` mask.
- `initBlocks` and `localBlocks` must be non-negative and no greater than the
  logical block width. Setting both to zero disables `local_mask`.
- `q_len` and `kv_len` may be zero, including for the entire batch. The kernel
  skips empty-query computation and fills scores for empty KV requests. An
  all-empty query batch launches with one block.
- A PageAttention block table may be wider than the actual logical KV length.
  Score width is `RoundUp(blockTableOptional.shape[1], 16)`. On Ascend 950,
  widths above 256 are flushed in 256-column windows.
- PageAttention BBND/BNBD keys may be non-contiguous only on the physical-page
  axis. The operator reads the first-axis element stride from tensor metadata;
  all inner axes must be contiguous. TND keys must be contiguous, and
  `scaleOptional` remains tightly packed by logical page.

## Invocation Example

The following excerpt shows the two-stage invocation for a non-quantized BBND
PageAttention input. See
[test_aclnn_msa_index_score.cpp](../examples/test_aclnn_msa_index_score.cpp)
for complete BBND, BNBD, TND, INT8, FP8, empty-sequence, strided-page, and wide
block-table accuracy cases.

```cpp
char layoutKey[] = "BBND";
int64_t sparseMode = 3;
int64_t initBlocks = 0;
int64_t localBlocks = 1;
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

## Validation Matrix

The standalone example runs 40 cases on Ascend 950: 36
FLOAT16/BFLOAT16/INT8 cases plus four FP8 cases. A2/A3 skip FP8 and run 36
cases. The matrix includes all supported layouts, empty sequences,
non-contiguous PageAttention page axes, and a block-table width of 257.

FLOAT16/BFLOAT16/INT8 use `atol=rtol=1e-3`; FP8 uses `atol=rtol=2e-2`.
