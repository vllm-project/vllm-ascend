# V1 rejection sampling

`vllm_ascend.ops.triton.v1.reject_sample` contains the legacy Ascend rejection
sampler kernels. `vllm_ascend/sample/rejection_sampler.py` launches the greedy,
random, recovered-token, expansion, and block-verification paths. V1 startup
warmup uses the same module through
`model_executor/warmup/rejection_sampler_triton_warmup.py`.

The public launch helpers remain `rejection_greedy_sample_with_triton`,
`expand_triton`, and `cal_grid_and_block_size`. Kernel signatures, random-number
inputs, output buffers, and launch configuration are unchanged by the move.

Model Runner V2 uses `worker/v2/spec_decode/rejection_sampler_utils.py` together
with [`spec_decode/resample.py`](../../spec_decode/resample.py).

Run the existing operator tests on NPU:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample.py
```
