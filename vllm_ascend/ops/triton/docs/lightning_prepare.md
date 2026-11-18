# QSA Lightning input preparation

## Description

- **Location**: `vllm_ascend/ops/triton/qwen4_exp/lightning_prepare.py`.
- **Function**: `prepare_qsa_lightning_inputs` constructs the five inputs needed
  by `qsa_select_paged_tokens_lightning` before `npu_lightning_indexer` scoring.
- **Formula**: `visible = min(floor((position + 1)/4), floor(sequence_length/4))`;
  valid heads have weight `1/sqrt(128)`, padded heads have weight zero.
- **Algorithm flow**: one program per query row normalizes request ownership,
  gathers the request length and block table, computes visible complete groups,
  writes query end `row + 1`, generates weights, and optionally copies/pads Q.
- **Supported modes**: model-specific BF16 NPU preparation. A3 tests are provided;
  final-head execution and other applicable devices remain unverified.

## Parameters

| Parameter | Role | Type/layout |
| --- | --- | --- |
| `query` | Input | BF16 `[rows, heads, 128]`, last dimension contiguous |
| `query_positions` | Input | int32/int64 `[rows]` logical positions |
| `sequence_lengths` | Input | int32/int64 `[requests]` |
| `token_to_req` | Input | int32/int64 `[rows]` request owners |
| `block_table` | Input | int32/int64 `[requests, columns]`, arbitrary row/column strides |
| Returned query | Output | `[rows, padded_heads, 128]`; aliases input when padding is unnecessary |
| Returned weights | Output | BF16 `[rows, padded_heads]` |
| Returned visible groups | Output | int32 `[rows]` |
| Returned query ends | Output | int32 `[rows]`, values `1..rows` |
| Returned row table | Output | Same dtype as table, `[rows, columns]` |

## Constraints

All metadata is on the query device with matching row counts. Head counts are
1 through 64, rounded up to the next of `(8, 16, 24, 32, 64)`. Nonempty query
inputs require a nonempty request table. The compression ratio is fixed at 4.
Positions, lengths and resulting group counts must fit the downstream int32
metadata contract; this helper is not an arbitrary-width sequence API.

Negative owners in `[-requests, -1]` follow PyTorch indexing semantics. Other
invalid owners produce zero visible groups and a zero table row, using bounded
pointers before masked loads. Signed division preserves floor behavior for
negative test inputs. Head padding does not change the number of query rows.

**Side effects**: no inputs are mutated. The returned query aliases the original
when its head count is supported; consumers must not assume an independent copy.
All other outputs are allocated by the wrapper. Empty input launches no kernel.
Every graph replay reads current owners, lengths, positions and table contents;
no scheduler values are cached on the host. Indexer scoring and subsequent
packing of group/tail indices are outside this operator.

## Origin and Differences

Replaces the model's PyTorch gathers, arithmetic, arange, full, zeros and cat
preparation chain from PR #21957/#20947 with one launch. Output semantics for
valid owners are preserved; invalid owners are explicitly masked instead of
allowing invalid addresses. The public wrapper owns validation and launch sizes.

## Test Cases

The independent PyTorch reference requires exact equality of all five outputs.
Parameterized heads are `1, 4, 8, 9, 17, 24, 33, 64`, rows `0, 1, 32`;
tests also cover strided inputs, int32/int64 owners, invalid owners, input
contract errors and graph replay with changing metadata. These head counts
include synthetic padding boundaries, not a claim of model support for each.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qsa_lightning_prepare.py
pytest -sv tests/ut/ops/test_qsa_lightning_prepare_contract.py
```
