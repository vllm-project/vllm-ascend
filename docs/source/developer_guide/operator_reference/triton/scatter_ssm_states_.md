# scatter_ssm_states_

Write updated SSM rows back into cache.

[Implementation of `scatter_ssm_states_`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/mamba/state_index.py) · [Caller `_forward_core` in gdn.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/gdn.py)

## Calling convention

`scatter_ssm_states_(state, indices, source)`

The Ascend GDN prefill path uses this operation to write updated recurrent state back to the original cache.

## Parameters and results

| Name | Contract |
| --- | --- |
| `state` | Floating-point cache of shape `[N, *inner_dims]`. Each logical row has contiguous inner dimensions. The stride between rows may include padding. |
| `indices` | One-dimensional INT32 or INT64 tensor of physical row IDs on the same device as the cache. Indices may be unsorted and strided. |
| `source` | Updated rows of shape `[B, *inner_dims]` on the cache device, with contiguous inner dimensions. |

Writes each source row to its indexed destination, converting values to the cache dtype. Returns `None`. Use unique destination IDs to avoid concurrent writes. Unselected rows and physical padding are preserved.

## Constraints

The caller supplies row IDs in `[0, N)`. The cache has at least one physical row and at least two dimensions. A launched copy requires a positive logical row size. Only the row stride may include padding and it must span the full logical row. With an empty selection, the wrapper returns without writing.

## Origin and validation

The Triton implementation replaces indexed write-back in Ascend GDN fused prefill. The [nightly state-index tests](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_state_index.py) compare against PyTorch across dense and padded storage, INT32/INT64 indices, tile boundaries (1023, 1024 and 1025 elements) and the production GDN state shape. The tests exercise invalid layouts. Device-family and graph coverage depend on the deployed backend.

See also [gather_ssm_states](gather_ssm_states.md).
