# torch.ops.vllm.kda_state_scatter

Write packed final KDA states back to selected cache pages.

[Implementation of `_kda_state_scatter`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/kda_state_copy_plan.py) · [Production caller in kda_state_copy_plan.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/kda_state_copy_plan.py)

## Calling convention

```python
torch.ops.vllm.kda_state_scatter(state, packed, indices, layer_name)
```

Supply packed states and one int32 or int64 index per row. Invalid destinations are skipped and valid destinations must be unique. The state-copy plan converts final states to contiguous cache dtype before dispatch. Initial-state flags do not suppress final writes. The registered operator mutates the cache and returns `None`. Empty selections launch no copy kernel. Cache page gaps are preserved. Cache and packed storage must be disjoint.

## Validation

[Integration and semantic tests](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_kda_state_copy_integration.py)
