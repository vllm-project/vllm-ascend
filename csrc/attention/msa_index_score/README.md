# MsaIndexScore

## Product Support

| Product                                                               | Supported |
| --------------------------------------------------------------------- | :-------: |
| <term>Atlas A2 Training Series/Atlas A2 Inference Series</term>       |     √     |
| <term>Atlas A3 Training Series/Atlas A3 Inference Series</term>       |     √     |
| <term>Ascend 950PR/Ascend 950DT</term>                                |     √     |

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

| Parameter | Kind | Description | Data Type | Format |
| --------- | ---- | ----------- | --------- | ------ |
| `query` | Input | Query tensor in TND layout, shape `[T1, N1, D]`. | BFLOAT16, FLOAT16, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | ND |
| `key` | Input | Key tensor in TND `[T2, N2, D]`, BNBD `[block_num, N2, block_size, D]`, or BBND `[block_num, block_size, N2, D]` layout. | BFLOAT16, FLOAT16, INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN | ND |
| `block_table` | Optional input | PageAttention logical-block-to-physical-page mapping, shape `[B, maxBlockNumPerSeq]`. Required for BBND and BNBD. | INT32 | ND |
| `scale` | Optional input | INT8 dequantization scale. PageAttention shape: `[block_num, N2, block_size]` or `[block_num, block_size, N2]`; TND shape: `[T2, N2]`. | FLOAT | ND |
| `atten_mask` | Optional input | Base causal mask used by `sparse_mode=3`, shape `[2048, 2048]`. A value of 1 excludes a position and 0 includes it. | INT8 | ND |
| `actual_seq_qlen` | Input | Non-decreasing query prefix sums, shape `[B+1]`. | INT32 | ND |
| `actual_seq_klen` | Input | TND key prefix sums `[B+1]`, or visible key lengths `[B]` for PageAttention. | INT32 | ND |
| `start_loc` | Input | Logical-block index containing the current query, shape `[B]`. | INT32 | ND |
| `layout_key` | Attribute | Key layout: `"TND"`, `"BBND"`, or `"BNBD"`. The aclnn parameter is `layoutKeyOptional` and defaults to `"BBND"`. | STRING | - |
| `sparse_mode` | Attribute | 0: `defaultMask`; 3: `rightDownCausal`. | INT64 | - |
| `init_blocks` | Attribute | Number of leading blocks assigned `1e30`. Default: 0. | INT64 | - |
| `local_blocks` | Attribute | Size of the local window `[max(0, start_loc+1-local_blocks), start_loc]`, assigned `1e29`. Default: 1. | INT64 | - |
| `score` | Output | Block scores, shape `[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]`. | FLOAT | ND |

## Constraints

- Only `block_size=128` is supported.
- `layout_key` must match the actual shape of `key`.
- PageAttention requires `block_table`. TND must omit `block_table` and use
  `[B+1]` prefix sums for `actual_seq_klen`.
- On the non-quantized path, `query` and `key` must have the same dtype and
  `scale` must be absent. A2/A3 support FLOAT16 and BFLOAT16. Ascend 950 also
  supports HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN.
- The quantized path supports a FLOAT16 query, an INT8 key, and a required
  FLOAT scale. Native FP8 is not an INT8 quantized path: on Ascend 950, query
  and key must use the same FP8 dtype and `scale` must be absent.
- `sparse_mode=0` requires no `atten_mask`. `sparse_mode=3` requires an INT8
  mask with shape `[2048, 2048]`.
- `init_blocks` and `local_blocks` must be non-negative and cannot exceed the
  logical block width. Setting both to 0 disables `local_mask`.
- `block_table` may be wider than the actual logical KV block count. The score
  width is `RoundUp(block_table.shape[1], 16)`. The Ascend 950 C2UB path flushes
  widths greater than 256 in 256-column windows.
- `q_len` and `kv_len` may be zero, including for the entire batch. Empty query
  requests skip QK computation; empty KV requests produce fill scores; an
  all-empty query batch launches with one block.
- PageAttention BBND/BNBD keys may be non-contiguous on the physical-page axis
  on A2/A3 and Ascend 950. All inner axes must remain contiguous. TND keys must
  be contiguous, and `scale` remains tightly packed by logical page.
- The operator returns block scores only and does not perform TopK.

## Build and Run

For Atlas A2/A3:

```bash
bash build.sh --pkg --soc=ascend910b --ops=msa_index_score -j32
bash ./build_out/cann-ops-transformer-custom_linux-x86_64.run \
  --quiet --install-path=/tmp/msa_opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust \
  --vendor_name=custom --soc=ascend910b
```

For Ascend 950:

```bash
bash build.sh --pkg --soc=ascend950 --ops=msa_index_score -j32
bash ./build_out/cann-ops-transformer-custom_linux-x86_64.run \
  --quiet --install-path=/tmp/msa_opp
source /tmp/msa_opp/vendors/custom_transformer/bin/set_env.bash
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust \
  --vendor_name=custom --soc=ascend950
```

The expected result is 40/40 cases on Ascend 950. A2/A3 skip the four FP8
cases and run 36 cases. FLOAT16/BFLOAT16/INT8 use a tolerance of `1e-3`; FP8
uses `2e-2`.

## References

- [aclnn interface documentation](./docs/aclnnMsaIndexScore.md)
- [End-to-end example](./examples/test_aclnn_msa_index_score.cpp)
- [Test guide](./tests/README.md)
- [torch extension documentation](../../torch_extension/cann_ops_transformer/docs/zh/msa_index_score.md)

## Implementation Notes

- `layout_key` selects PageAttention BBND/BNBD or packed TND. TND does not use
  `block_table`.
- For `sparse_mode=3`, the host validates `atten_mask[2048,2048]`; the device
  derives right-down-causal visibility without loading the mask element by
  element.
- The Ascend 950 implementation is under `op_kernel/arch35`. It uses native
  Cube FP8 tiling keys 4/5/6 without a scale or an intermediate FP16 cast.
- `op_kernel/catlass` is an operator-private Catlass snapshot derived from the
  v1.3.1-notla implementation. It is intentionally isolated from the repository
  Catlass submodule because its interfaces and implementation differ.
