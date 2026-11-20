# gather_ssm_states

Gather selected physical SSM cache rows.

[Implementation of `gather_ssm_states`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/mamba/state_index.py) · [Caller `_forward_core` in gdn.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/gdn.py)

## Calling convention

`gather_ssm_states(state, indices, has_initial_state, *, output_dtype=None)`

The Ascend GDN prefill path uses this operation to read initial recurrent state into dense working storage.

## Parameters and results

| Name | Contract |
| --- | --- |
| `state` | Floating-point cache of shape `[N, *inner_dims]`. Each logical row has contiguous inner dimensions. The stride between rows may include padding. |
| `indices` | One-dimensional INT32 or INT64 tensor of physical row IDs on the same device as the cache. Indices may be unsorted and strided. |
| `has_initial_state` | Boolean mask matching the shape and device of the indices. A false entry produces zeros without reading the selected cache row. |
| `output_dtype` | Optional output dtype. Defaults to the cache dtype. |

Returns a new contiguous tensor of shape `[B, *inner_dims]`, where `B = indices.numel()`. Repeated IDs are valid. The cache is read-only.

## Constraints

The caller supplies row IDs in `[0, N)`. The cache has at least one physical row and at least two dimensions. A launched copy requires a positive logical row size. Only the row stride may include padding and it must span the full logical row. With an empty selection, the wrapper returns an empty output with the requested dtype.

## Origin and validation

The Triton implementation replaces indexed reads and a separate initial-state clear in Ascend GDN fused prefill. The [nightly state-index tests](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_state_index.py) compare against PyTorch across dense and padded storage, INT32/INT64 indices, tile boundaries (1023, 1024 and 1025 elements) and the production GDN state shape. The tests exercise invalid layouts. Device-family and graph coverage depend on the deployed backend.

See also [scatter_ssm_states_](scatter_ssm_states_.md).
