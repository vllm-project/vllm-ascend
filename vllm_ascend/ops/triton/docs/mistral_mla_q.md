# Mistral MLA query preparation

## Description

- **Function**: `try_ssmax_split_rope` prepares the content and RoPE query lanes
  for Mistral Large 4 BF16 decode. Unsupported inputs return `None` so the
  attention implementation can use its ordinary path.
- **Formula**: with BF16 rounding `R`, SSMax is
  `scale = R(R(log(position + 1) * slope) + bias)`. The content output is
  `R(q_content * scale)`. The RoPE output first rounds the rotated raw query,
  then applies and rounds SSMax: `R(R(RoPE(q_rope)) * scale)`.
- **Algorithm flow**: one program handles one token/head pair. It scales the
  64 content values, rotates 32 adjacent query pairs, rounds the rotation,
  scales the rotated values, and writes the two 32-value halves separately.
- **Supported modes**: NPU BF16 decode. The existing device regression test is
  restricted to A5; execution on other Ascend devices has not been validated
  by this change. CUDA is not accepted by the production wrapper.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `query` | Input | Raw projected query, `[tokens, heads, 128]` | BF16 | Contiguous |
| `log_positions` | Input | One log position per token | Floating point | Contiguous `[tokens]` |
| `weight`, `bias` | Input | SSMax slope and intercept, one per head | Floating point | Contiguous `[heads]` |
| `cos`, `sin` | Input | Gathered RoPE tables, with the frequency half repeated | BF16 | `[tokens, 1, 1, 64]`, last stride 1 |
| Return value 0 | Output | Scaled content lane | BF16 | Contiguous `[tokens, heads, 64]` |
| Return value 1 | Output | Rotated and scaled query, in split-half layout | BF16 | Contiguous `[tokens, heads, 64]` |

## Constraints

- All inputs reside on the same NPU. `1 <= tokens <= 8` and `1 <= heads <= 64`.
  The TP8 deployment uses eight local query heads.
- This operator handles the 64-content/64-RoPE layout. NoPE layers use the
  ordinary attention path and are not rotated.
- The wrapper allocates both outputs and does not modify its inputs.
- Cosine/sine row strides and the head count specialize the JIT kernel. Token
  count changes the launch grid. There is no autotuning or new warmup policy.
- RoPE must precede SSMax for the target Attention query. The Indexer has a
  different upstream contract and continues to scale before rotation.
- Shape-compatible graph capture reuses the same wrapper. Ascend ACL Graph
  correctness and performance still require device validation.

## Origin and Differences

- **Origin**: Mistral Large 4 query preparation and the upstream
  [RoPE/SSMax implementation](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/models/mistral/amd/ops/ssmax.py).
- **Differences**: the query/key projection and key rotation remain outside
  this operator. The output RoPE lane uses Ascend's split-half layout instead
  of the upstream interleaved layout. Both layouts represent the same paired
  coordinates, but BF16 rotation and SSMax must retain their intermediate
  rounding order. Cross-device arithmetic may still differ from an Ascend
  operator; a portable-kernel test does not validate that device boundary.

## Test Cases

The existing Ascend test compares both outputs against RoPE followed by SSMax,
including the unsupported nine-token fallback:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mistral_mla_decode_fusions.py
```

Target query ordering, NoPE behavior, SSMax enabled and disabled, and
graph-padding rows are covered by `tests/ut/models/test_mistral_large4.py`.
NPU kernels, full-model accuracy, and end-to-end performance still require
Ascend validation.

## Example

```python
from vllm_ascend.ops.triton.mistral_mla_q import try_ssmax_split_rope

prepared = try_ssmax_split_rope(query, log_positions, slope, bias, cos, sin)
if prepared is not None:
    q_content, q_rope = prepared
```
