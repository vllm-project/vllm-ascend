# VectorPagedAttention

Single-query paged attention for small decode shapes, executed entirely on the
AI vector cores.

## Why it exists

At a decode step with one query row per request, a short context and a modest
head count, the general fused attention operator is dominated by fixed
per-call cost rather than by the KV traffic it performs. Raising the declared
KV capacity from 128 to 512 on an Atlas A3 costs about 1.3 us of the ~25 us a
call takes, so most of that time is not the work.

This operator removes the fixed cost by doing only one thing. A single AI
vector core owns one `(request, head)` pair: it reads that head's slice of the
pages the sequence actually occupies, keeps the whole softmax in UB and writes
its own `head_dim` outputs. No core waits for another, there is no combine
pass, and the operator asks for no workspace of its own beyond the system
reserve. The kernel is `KERNEL_TYPE_AIV_ONLY`.

It also reads `seq_lens` from device memory, so it touches only the pages the
sequence occupies rather than a padded capacity. That matters when the caller
declares a constant capacity to keep host-side arguments stable across an
aclgraph replay.

## Interface

```python
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
attn_out = torch.ops._C_ascend.npu_vector_paged_attention(
    query,             # [batch, num_heads, head_dim] bfloat16
    key_cache,         # [num_blocks, block_size, num_kv_heads * head_dim] bfloat16
    value_cache,       # same shape as key_cache
    block_table,       # [batch, max_blocks] int32
    seq_lens,          # [batch] int32
    num_kv_heads=num_heads,
    scale=head_dim ** -0.5,
)                      # -> [batch, num_heads, head_dim] bfloat16
```

`key_cache` and `value_cache` may also be given as
`[num_blocks, block_size, num_kv_heads, head_dim]`; the adapter views them,
never copies.

## Declared domain

This is a narrow operator, not a general one. Everything below is checked in
the torch adapter, which raises a `RuntimeError` naming the violated rule, and
again in tiling. **Check the domain before an aclgraph capture**: a tiling
failure during capture is an error, not a fallback to another kernel.

| property | supported |
| --- | --- |
| dtype | bfloat16 |
| `head_dim` | 64 |
| heads | multi-head only, `num_kv_heads == num_heads`, up to 128 heads |
| query rows | one per request |
| `batch` | 1 to 32 |
| `block_size` | a power of two in [8, 128] |
| `block_size * block_table.size(1)` | at most 4096 |
| `batch * num_heads` | at most the die's AI vector core count, 48 on both supported SOCs |
| SOC | `ascend910b`, `ascend910_93` |

The `batch * num_heads` bound is the operator's shape: one task per core, no
tail loop. A caller can test it directly with
`torch_npu.npu.get_device_properties(device).vector_core_num`.

Outside this domain, keep using
`torch_npu.npu_fused_infer_attention_score`, which is what this operator is
narrower than, not a replacement for.

## Accuracy

The kernel accumulates in fp32 throughout: it casts the bfloat16 query and KV
pages up on load, applies the scale to the query rather than to every score,
and rounds once on the way out. Against an fp32 reference it is bit-exact at
the shapes in `tests/e2e/nightly/single_node/ops/singlecard_ops/test_vector_paged_attention.py`.
