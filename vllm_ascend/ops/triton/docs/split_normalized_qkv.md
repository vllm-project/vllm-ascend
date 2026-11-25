# split_normalized_decode_qkv

## Function

GDN verification receives convolution output with Q, K and V packed in the last dimension. The recurrent operator consumes three contiguous tensors and expects L2-normalized queries and keys. `AscendGatedDeltaNetAttention._forward_core` calls this wrapper for the speculative rows. RecoverSSM normalizes its own inputs.

Each program reads a tile of heads from one token. Query and key heads use `x * rsqrt(sum(x * x) + 1e-6)` with FP32 arithmetic. Value heads are copied. Each element has one writer. The inputs can be strided and are read only.

## Interface

| Argument | Meaning |
| --- | --- |
| `packed` | Floating-point NPU tensor of shape `[tokens, 2 * key_dim + value_dim]` |
| `key_dim`, `value_dim` | Local widths of each key and value projection after tensor parallel sharding |
| `head_k_dim`, `head_v_dim` | Width of an individual key or value head |

The projection widths must be multiples of the corresponding positive head widths. The outputs preserve the input dtype. Q and K have shape `[1, tokens, key_dim // head_k_dim, head_k_dim]`. V has shape `[1, tokens, value_dim // head_v_dim, head_v_dim]`. An empty token batch returns empty outputs.

## Origin and tests

The reference splits the packed tensor, materializes contiguous projections, and applies vLLM's `l2norm_fwd` separately to Q and K. The Ascend implementation combines these operations in one launch and specializes on head geometry and input strides. Token counts change the launch grid.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gdn_split_normalize.py
```
