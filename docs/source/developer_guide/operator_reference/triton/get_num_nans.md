# get_num_nans

Count NaN logits per request.

[Implementation of `get_num_nans`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/metrics/num_nans.py) · [Registration or binding in patch_triton.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/patch/worker/patch_triton.py)

## Description

For each request, count NaNs across the vocabulary. The wrapper accepts two-dimensional logits with unit vocabulary stride and returns a new INT32 tensor of shape [num_reqs]. Row starts follow the input row stride.

Each program scans one row in blocks of 8192, masks the last block, casts to FP32 and applies the CANN libdevice NaN predicate. Padding contributes zero. The Ascend patch registers the wrapper for the sampler and rejection sampler.

## Origin and differences

The upstream kernel resolves its libdevice predicate through a CUDA-specific path. The Ascend implementation imports the predicate from the CANN Triton backend.

## Test cases

The [operator tests](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_num_nans.py) compare counts exactly against a CPU PyTorch reference across FP32, BF16 and FP16, partial vocabulary tiles and multiple NaN densities. A special-values case distinguishes NaNs from infinities, signed zeros and finite extrema.

## Example

```python
from vllm_ascend.ops.triton.metrics.num_nans import get_num_nans

counts = get_num_nans(logits)
```
