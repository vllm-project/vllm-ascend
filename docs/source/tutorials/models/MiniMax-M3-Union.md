# MiniMax-M3 FP8 Union attention on Ascend 950

MiniMax-M3 runs its supported FP8 sparse attention through UnionSparseGen and
GenericBlockSparseAttention (GBSA) exactMask. Eight adjacent queries share the
union of their selected KV blocks. An exact per-query mask preserves the original
selection and causal boundary. The indexer and TopK selection are unchanged.
This default applies to both prefill and decode; it does not introduce a policy
or switch between attention implementations at runtime.

## Requirements

- Ascend 950 with FP8 attention support.
- BF16 model activations, FP8 E4M3 K/V caches, head dimension 128,
  128-token cache pages, one local KV head and at most 16 local query heads.
  MiniMax-M3.1-Flash with TP4 satisfies these attention geometry constraints.
- Matching CANN custom operator packages from
  [ops-nn PR #14010](https://gitcode.com/cann/ops-nn/pull/14010)
  (`UnionSparseGen`) and
  [ops-transformer PR #15473](https://gitcode.com/cann/ops-transformer/pull/15473)
  (`GenericBlockSparseAttention`, `mask_mode=3`, FP8 `quant_mode=5`).

Build/install the two packages using `index/union_sparse_gen/docs/build_install_and_call.md`
and `index/union_sparse_gen/docs/model_integration_and_build.md` in the ops-nn PR.
The Python packages must expose `cann_ops_nn_custom` and
`cann_ops_transformer_custom`. Set `PYTHONPATH`, `ASCEND_CUSTOM_OPP_PATH` and
`LD_LIBRARY_PATH` according to those build instructions before starting workers.
vLLM Ascend does not change operator search paths or install these packages.
The standard non-Union CANN packages remain required by the model's indexer.

## Deployment

Follow the [MiniMax-M3.1-Flash deployment guide](MiniMax-M3.1-Flash.md), use
`--kv-cache-dtype fp8_e4m3`, and install the matching CANN packages above.
Union is selected automatically for the supported hardware, cache type and
attention geometry; no additional configuration flag is needed. Other hardware,
cache types and attention geometries retain the native attention path without
loading the Union extensions. Missing or incompatible Union packages fail during
model initialization for configurations that select Union.

The extensions load before graph capture. Output capacities come from host
attention metadata, including per-request groups and incomplete groups of fewer
than eight queries. Union uses `softmax_precision=0` as required by the exactMask
operator contract; the native non-Union FP8 precision setting is unchanged.
Cache policy, model runner, graph mode and distributed execution remain governed
by the normal serving configuration.
