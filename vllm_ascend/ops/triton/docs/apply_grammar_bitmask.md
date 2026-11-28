# apply_grammar_bitmask

## Description

**Function.** During grammar-constrained decoding, this operator masks logits in place.
Each bit in a packed 32-bit word represents one token. A set bit preserves the
original logit. A clear bit sets the logit to negative infinity.

**Formula.** For mask row `r`, decode the request and position with
`req, pos = divmod(logits_indices[r], MASK_STRIDE)`. The position is active when
`pos < cu_num_logits[req + 1] - cu_num_logits[req]`. For an active position,
`row = cu_num_logits[req] + pos`. For each token index `v < vocab_size`:

$$
word = \left\lfloor \frac{v}{32} \right\rfloor,\qquad
bit = v \bmod 32
$$

$$
allowed_{r,v}
=
\mathbb{1}
\left[
bitmask_{r,word} \;\&\; (1 \ll bit) \neq 0
\right]
$$

$$
logits_{row,v}
=
\begin{cases}
logits_{row,v}, & allowed_{r,v}=1 \\
-\infty, & allowed_{r,v}=0
\end{cases}
$$

**Algorithm flow**

1. The caller supplies a logical grid of `(num_masks, ceil(vocab_size / BLOCK_SIZE))`.
   Each task processes one vocabulary block in one mask row.
2. The Ascend launcher distributes tasks across the available VectorCores. The
   task counts of any two Triton programs differ by at most one.
3. Each program unpacks the block's 32-bit words. Bit 0 represents the first
   token in a word and bit 31 represents the last.
4. For active positions within the vocabulary, the program loads the logits,
   selects the original value or negative infinity for each token, and writes
   the result back. It skips logits loads and stores for inactive positions and
   vocabulary padding.

**Device coverage.** The nightly business-shape test targets Ascend A2, A3, and A5.
The measurements reported for this optimization were taken on Ascend A2.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits` | Input/Output | Logits of logical shape `[num_logits, vocab_size]`, masked in place. | FP32 / FP16 / BF16 | ND |
| `logits_indices` | Input | One packed value `req * MASK_STRIDE + pos` per mask row. | INT32 | ND |
| `cu_num_logits` | Input | Cumulative logits row offsets of length `num_reqs + 1`. Adjacent offsets delimit one request's rows. | INT32 | ND |
| `MASK_STRIDE` | Attribute | Positive integer used to encode and decode the position within a request. | int | scalar |
| `bitmask` | Input | Packed grammar mask with one row per mapping entry and one word per 32 tokens. | INT32 | ND |
| `vocab_size` | Attribute | Number of vocabulary entries to process. | int | scalar |
| `BLOCK_SIZE` | Attribute | Number of vocabulary entries per logical task. The nightly tests use 8192. | int | scalar |

## Constraints

- `logits` is two-dimensional and contiguous along the vocabulary dimension
  (`logits.stride(1) == 1`). The row stride can include padding.
- `logits_indices` is one-dimensional with `bitmask.shape[0]` entries. Mask
  rows can map to logits rows in a different order. Each entry must encode
  `0 <= req < num_reqs` and `0 <= pos < MASK_STRIDE`.
- `bitmask` is two-dimensional with at least one word per 32 vocabulary entries:

$$
bitmask.shape[1]
\ge
\left\lceil
\frac{vocab\_size}{32}
\right\rceil
$$

- The last block uses `block_offset < vocab_size` to guard the vocabulary tail.
  The launcher returns immediately when the mask count or vocabulary size is zero.
- Graph-mode support has not been verified.

## Origin and Differences

The Ascend implementation replaces the vLLM structured-output kernel
`structured_outputs._apply_grammar_bitmask_kernel` through `patch_triton.py`.
It maps the upstream logical grid to the following number of Triton programs:

```python
num_programs = min(get_vectorcore_num(), total_tasks)
```

Each task processes a full vocabulary block without further subdivision by
`BLOCK_SIZE_SUB`. The launcher uses `multibuffer=False`. This version selects
between the original logit and negative infinity before storing, so the store
mask depends only on position validity and the vocabulary boundary.

## Test Cases

- The business-shape test compares BF16 logits with a CPU reference using this configuration:

```text
logits.shape         = [64, 151936]
logits.dtype         = BFLOAT16
logits_indices.shape = [64]          # packed (req_idx, position) mapping
logits_indices.dtype = INT32
cu_num_logits.shape  = [num_reqs + 1]
cu_num_logits.dtype  = INT32
bitmask.shape        = [64, 4748]
bitmask.dtype        = INT32
BLOCK_SIZE           = 8192
```

- Six dtype and vocabulary-size combinations check the output bit for bit,
  including inactive mappings, row padding, signed zeros, and NaN payloads.
- Three empty-work cases check that the call returns when the mask count,
  vocabulary size, or both are zero.

```bash
pytest -sv --noconftest \
tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_grammar_bitmask_triton.py
```

## Example

```python
import torch

from vllm.triton_utils import triton
from vllm_ascend.ops.triton.apply_grammar_bitmask import (
  _apply_grammar_bitmask_kernel,
)

device = torch.device("npu:0")

rows = 64
num_reqs = 16
logits_per_req = 4
vocab_size = 151936
block_size = 8192
mask_stride = 8  # must exceed the largest packed position
bitmask_words = (vocab_size + 31) // 32

logits = torch.randn(
  rows,
  vocab_size,
  dtype=torch.bfloat16,
  device=device,
)

# Cumulative logit rows per request.
cu_num_logits = torch.arange(
  0,
  (num_reqs + 1) * logits_per_req,
  logits_per_req,
  dtype=torch.int32,
  device=device,
)

# Each entry packs (req_idx, position) as req_idx * mask_stride + position.
req_idx = torch.arange(rows) // logits_per_req
position = torch.arange(rows) % logits_per_req
logits_indices = (req_idx * mask_stride + position).to(torch.int32).to(device)

# bit=1 means allowed. -1 means all 32 bits are 1.
bitmask = torch.full(
  (rows, bitmask_words),
  -1,
  dtype=torch.int32,
  device=device,
)

grid = (
  rows,
  triton.cdiv(vocab_size, block_size),
)

_apply_grammar_bitmask_kernel[grid](
  logits,
  logits.stride(0),
  logits_indices,
  cu_num_logits,
  bitmask,
  bitmask.stride(0),
  vocab_size,
  MASK_STRIDE=mask_stride,
  BLOCK_SIZE=block_size,
)
```
