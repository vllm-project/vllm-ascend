# SFA DCP fused All-to-All and LSE combine

## Purpose

`sfa_dcp_a2a_fused` post-processes SFA attention results when decode context
parallelism (DCP) is enabled. Each rank starts with a partial attention output
and its FP32 log-sum-exp (LSE). The operator returns the output slice owned by
the local DCP rank after combining contributions from every rank.

The implementation is in `vllm_ascend/ops/triton/sfa_cp.py` and has three
stages:

1. A stride-aware Triton kernel packs the partial output and LSE into one send
   buffer.
2. `torch.distributed.all_to_all_single` exchanges that buffer over the DCP
   HCCL process group.
3. A Triton kernel reconstructs the LSE values and performs a numerically
   stable weighted reduction of the partial outputs.

## Inputs and output

| Argument | Shape or type | Description |
| --- | --- | --- |
| `sfa_output` | `[tokens, heads, head_dim]` | BF16, FP16, or FP32 partial attention output. Arbitrary positive strides are supported. |
| `softmax_lse` | `[tokens, heads, 1]`, FP32 | LSE associated with each partial output row. |
| `dcp_size` | positive `int` | Number of ranks in the DCP process group. |
| `scatter_dim` | `0` or `1` | Dimension sharded by the All-to-All: tokens for `0`, heads for `1`. |
| `group_name` | `str` | Unique name of a live vLLM `GroupCoordinator`. |

The result has the same dtype and `head_dim` as `sfa_output`. Its token or head
dimension is divided by `dcp_size`, according to `scatter_dim`.

## Packed payload

The send buffer is laid out as
`[dcp_size, local_scatter_size, replicated_size, packed_dim]`. The first
`head_dim` elements contain the partial output. The remaining elements contain
the LSE representation.

- FP32 output stores the FP32 LSE directly in one element.
- BF16 and FP16 output use four elements: a signed base-2 exponent code and
  three base-256 digits containing the FP32 significand.

All four fields are integers in `[-255, 255]`, which are represented exactly by
both BF16 and FP16. Reconstructing the three digits therefore preserves the
original finite FP32 LSE, including nearby values above the FP16 finite range.
Exponent code zero marks NaN and infinity as invalid rank contributions.

## Triton launch strategy

Both Triton kernels flatten `(token, head)` into one logical row index. The
launch grid is

```python
grid_size = min(num_tokens * num_heads, get_vectorcore_num())
grid = (grid_size,)
```

Each program processes additional rows with a grid-stride loop. This limits
the program count to the physical vector-core count while still covering large
prefill and decode shapes. Each row uses one `BLOCK_D` vector, where
`BLOCK_D = next_power_of_2(head_dim)`.

The combine kernel keeps one FP32 output accumulator and scalar LSE state live.
It first finds the maximum valid LSE, then accumulates
`exp(lse - lse_max) * partial_output`. Invalid LSE rows contribute zero; when
all ranks are invalid, the output row is zero.

## Custom-op registration and graph tracing

The eager implementation resolves `group_name` through vLLM's registered
process groups and executes the HCCL collective. The `fake_impl` registered
with the operator is used only by PyTorch FakeTensor/`torch.compile` shape
propagation. It returns an empty tensor with the local output metadata and does
not execute a collective.

## Validation

The single-card nightly test covers BF16/FP16, both scatter dimensions,
head dimensions 96/128/160/256, non-contiguous inputs, invalid LSE rows, and
finite FP32 LSE values outside FP16 range. The multi-card A3 nightly test starts
a real two-rank HCCL group and invokes the registered custom operator end to
end for both scatter dimensions.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sfa_cp_a2a.py
pytest -sv tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_sfa_cp_a2a.py
```
