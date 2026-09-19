# A5 Kimi K3 non-absorbed MLA prefill

The existing MLA backend uses expanded QK192/V128 attention for BF16 prefill
with PCP1/DCP1. Decode retains absorbed FlashMLA QK576/V512, including its
existing graph, MLAPO and DCP paths. C8 retains its existing implementation.
There is no additional runtime option or service command.

Persistent KV remains compressed and first-axis strided, with a kernel block
size of 128. Prefill expands cached history in chunks of at most 16,384 tokens
and merges output/LSE online. The current chunk uses causal attention; history
chunks use noncausal attention. This avoids expanding the full cached prefix
at once. MRv1/MRv2 supply their existing exact prefill context lengths; decode
continues to use device-side lengths without a new CPU synchronization.

The A5 build includes matching `flash_attn` and `flash_attn_metadata` native
implementations and the official Torch binding. The optional `head_dim_v`
defaults to the QK dimension for existing GQA calls. The build retains GQA
D64/V64 TND and strided PA_BNBD templates and the other installed CANN Python
operators. The FlashAttn extension is compiled during the VA build.

## Measurements

Measured on A5 with CANN B060, TP8 rank 0, 12 local heads, using the actual K3
layer 3 MXFP8 projection weights and BF16 KV/gate/O weights. Values below are
single-layer forward times in milliseconds; K means 1024 tokens.

The local MLA boundary includes projection, normalization, compressed KV
write, attention, output gate and O projection. It excludes TP communication
and scheduler time. The operator comparison measures expanded native
FlashAttn against expanded FIA with prepared inputs.

| Input | FIA operator | New operator | Old absorbed local MLA | New local MLA | Local MLA saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| 40K | 20.011 | 22.576 | 67.501 | 31.655 | 35.846 |
| 32K | 13.102 | 14.544 | 44.588 | 21.200 | 23.388 |
| 4K | 0.210 | 0.288 | 1.599 | 1.274 | 0.325 |
| 16K | 2.820 | 3.477 | 12.843 | 7.009 | 5.834 |
| 7K | 0.578 | 0.770 | 3.283 | 2.212 | 1.071 |
| 512 | 0.018 | 0.025 | 0.580 | 0.613 | -0.033 |
| 128K | 239.036 | 252.420 | 698.900 | 287.457 | 411.442 |
| 128K, 99% prefix | 5.086 | 4.810 | 14.407 | 8.518 | 5.889 |

The prefix case contains 129,762 cached tokens and 1,310 current tokens. Its
operator values sum separately measured calls over the actual bounded history
chunks; they are not full-module timings. The 32K/40K operator samples vary
more than the short cases. The 512-token module regresses by approximately
33 microseconds. Operator parity with FIA is not achieved for every shape.

All eight cases passed numerical comparisons against the absorbed path and
sampled independent FP32 references. Maximum full local output RRMS between
the old and new paths was 0.424%. Additional NPU checks cover mixed
decode/prefill, random physical pages, padding, TP output-shard indexing,
GQA64 strided pages and changed-input graph replay. These are operator/module
results, not full-service or GPQA/GSM8K accuracy results. This release keeps
the measured config6 baseline; experimental tiling and strided-V candidates
are excluded.
