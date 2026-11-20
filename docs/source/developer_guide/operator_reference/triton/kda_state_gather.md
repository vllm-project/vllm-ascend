# torch.ops.vllm.kda_state_gather

Gather selected KDA cache pages into packed state.

[Implementation of `_kda_state_gather`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/kda_state_copy_plan.py) · [Caller `KDAStateCopyPlan.gather` in kda_state_copy_plan.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/kda_state_copy_plan.py)

## Calling convention

```python
torch.ops.vllm.kda_state_gather(state, indices, flags, layer_name)
```

Supply a bound cache shaped `[N,H,V,K]`, one-dimensional int32 or int64 indices and optional boolean flags. The result has shape `[len(indices),H,V,K]` and the cache dtype. Invalid indices and false flags yield zero rows. Repeated indices are valid. Pass `None` for flags to gather every valid row. Empty selections launch no copy kernel. Cache page gaps are preserved. Cache and packed storage must be disjoint.

## Validation

[`test_plan_matches_pr17301_native_operator`](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_state_copy_integration.py) compares the prepared plan with native state copy across cache and index dtypes.
