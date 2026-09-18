# A5 FlashMLA C8 contract

The existing `flash_mla_with_kvcache` torch operator accepts either merged
FP16/BF16 inputs or quantized MLA inputs. C8 is selected by the query dtype;
the model backend selects it from the existing layer quantization configuration.

| Tensor | C8 shape | Type |
| --- | --- | --- |
| query | `[tokens, heads, 512]` | FP8 E4M3 |
| query_rope | `[tokens, heads, 64]` | BF16 |
| k_cache | `[pages, 128, 1, 512]` | FP8 E4M3 |
| key_rope | `[pages, 128, 1, 64]` | BF16 |
| dequant_scale_query | `[tokens, heads]` or `[tokens, heads, 1]` | FP32 |
| dequant_scale_key | one value per layer | FP32 |
| attention output | TND or NTD, value width 512 | BF16 |
| LSE | `[heads, tokens]` | FP32 |

The two cache tensors have independent first-axis strides. Their inner pages
are contiguous. A runner page stores 128 latent rows followed by 128 BF16
64-dimensional rows, using 81,920 bytes before allocator padding. The 128-token
kernel page is distinct from a larger aligned cache-manager block.

Let `sq` and `sk` be the query and KV dequantization scales. As in FIA C8 and
MLAPO mode 8, the supplied BF16 query component is already divided by `sq * sk`:

```text
logits = softmax_scale * sq * sk * (query8 @ key8.T + query_rope @ key_rope.T)
value = key8 * sk
```

The kernel handles softmax probability quantization and value dequantization.
LSE represents the scaled logits before probability quantization. Empty rows
produce zero output and negative-infinity LSE for neutral DCP merging.

`flash_mla_with_kvcache_metadata(..., is_c8=True)` schedules the C8 M64/S128
template. The default metadata remains M96/S112 for merged FP16/BF16. Metadata
is generated outside ACLGraph and updated in place for replay. The public
nonquantized ACLNN ABI is preserved; C8 uses a separate native entry point.

The backend supports DCP1 decode, DCP8 history shards, and prefill. Split paths
keep the current chunk in BF16 and combine its output/LSE once with C8 history.
They do not expand historical FP8 cache to BF16. DCP1 with MXFP8 projection
weights uses MLAPO mode 8 to write the cache directly. Replicated-Q DCP8 uses
MLAPO mode 7 to write the BF16 current chunk on every rank, then quantizes only
the persistent cache slots owned by that rank. Query quantization and scaling
of its BF16 component are fused. History exchange overlaps current-chunk
attention, and the existing output gate/projection fast paths remain enabled.

Qualification scripts under `tests/e2e/nightly/single_node/ops/` cover operator
math, independently strided cache pages, graph replay, and real DCP8 exchange.
These gates compare quantized inputs; model accuracy additionally requires
layer-specific K3 calibration scales from the model checkpoint.

## Direct DCP output

For C8 T64/H96 decode with DCP8, the backend requests `NTD_DCP` output.
The kernel writes BF16 O512 and the exact FP32 LSE bits into a 64-byte-aligned
row of 272 INT32 words. The logical output remains `[tokens, heads, 512]`;
its backing storage is `[heads, tokens, 544]` BF16. A separate ordinary LSE
tensor is also returned. The final 60 bytes of every physical row are zero.

The existing DCP exchange consumes this storage without a packing kernel.
Both merge implementations accept the 272-word pitch. If a producer cannot
provide the direct view, packing retains the same 272-word collective count
on that rank. Other paths retain the 257-word protocol.

`flash_mla_c8_wire_qualification.py` and
`flash_mla_c8_wire_graph_qualification.py` cover both layouts, FD, empty rows,
padding and changing captured metadata. Native CP causal masking is a
separate operator design and is not part of this interface.
