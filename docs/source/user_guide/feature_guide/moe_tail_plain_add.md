# MoE Tail Plain Add

When a MoE layer has both shared experts and routed experts, vLLM-Ascend merges
their outputs once per layer at the end of the MoE block. The default
implementation uses a single-element multi-tensor add:

```python
result = torch._foreach_add([shared_output], [fused_output])[0]
```

Both inputs are ordinary activation tensors, so this is a plain elementwise
add. On Ascend the `_foreach_add` form still goes through the multi-tensor
dispatch and launches a dedicated `ForeachAddList` kernel per MoE layer. That
launch has a fixed cost (~17us on Atlas 910B) that is independent of the tensor
size, and for a one-element tensor list it buys no vectorization at all.

`enable_moe_tail_plain_add` replaces the combine with `shared_output +
fused_output`. The arithmetic is identical (same dtype, same operand order,
elementwise bf16 add), so outputs are bit-identical; only the kernel that
performs the sum changes, and one op launch per MoE layer disappears.

## Usage

Enable it per instance through `additional_config`:

```bash
vllm serve <model> --additional-config '{"enable_moe_tail_plain_add": true}'
```

It is intended for decode instances, where every layer's launch overhead sits
on the token-to-token critical path. `enable_moe_tail_plain_add` composes with
the other `additional_config` switches and is disabled by default.

## Effect

For a 45-layer MoE model on Atlas 910B, the removed `ForeachAddList` kernels
account for ~0.75ms per decode step, replaced by ~0.09ms of plain `Add`
kernels, i.e. ~0.66ms saved per step. The table below is the profiled kernel
time of the two runs (same deployment, decode instance, bf16 activations):

| Kernel | Baseline | With `enable_moe_tail_plain_add` |
| --- | --- | --- |
| `ForeachAddList` | 2835 calls, 47.4ms window, 16.7us/call | not launched |
| `Add` (tail) | — | ~2.0us/call, 45 calls per step |

No accuracy change is expected: the operation, its operands and the reduction
order stay the same.
