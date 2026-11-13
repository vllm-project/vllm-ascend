# Kimi K3.1 block-outermost KV cache

Kimi K3.1 can pack MLA, indexer and resident dense-MLA draft pages into one
physical row per scheduler block. Recurrent groups overlay the entire row,
using disjoint block IDs. A draft model is optional. Layer counts, page sizes,
dtypes and state shapes come from the actual KV cache specs.

Kimi K3.1 defaults to `BLNHC`. The existing `VLLM_KV_CACHE_LAYOUT` override
can select another block-outermost layout such as `BLHNC`. The old layer-outermost
paired-page implementation has been removed; explicit `LBNHC` selections fail
during configuration. This implementation targets Model Runner V2, packed SFA
main cache, one latent KV head, PCP=1 and DCP=1. Unsupported context-parallel
packing fails during planning.

## Ownership and storage

For one history block ID, the row contains:

```text
block b, byte address = backing + b * row_bytes
┌───────────┬───────────────┬───────────────────┬─────────────┐
│ MLA[0]    │ Indexer[0]    │ ... MLA/Indexer   │ draft pages │ ... padding
└───────────┴───────────────┴───────────────────┴─────────────┘

For a different ID owned by a recurrent group:
┌────────────┬────────────┬─────────────────┬────────────────┐
│ KDA state0 │ KDA state1 │ ... raw states  │ unused bytes   │
└────────────┴────────────┴─────────────────┴────────────────┘
```

The scheduler never assigns one live pool ID to both interpretations. Different
KDA groups also own disjoint IDs. One state remains a complete checkpoint for
one layer; packing changes where checkpoints fit, without reducing their data.

The history payload is the sum of every actual history spec's `page_size_bytes`.
Each recurrent group must fit inside that payload before stride padding. The
maximum group size is the history payload divided by one raw recurrent state
page size, rounded down, regardless of draft presence or page-size patterns.
Groups are balanced with the existing Kimi grouping helper. There is no
special branch for three or five draft layers.

Let `B` be the configured manager block size and `K=128` the kernel block size.
For history spec `s`, its kernel page has `Q[s] = page_size_bytes[s] / (B/K)`
bytes. The physical row is the largest local group payload, rounded up to the
LCM of these kernel-page sizes and the state dtypes' alignment. A PP-local row
is computed after layer projection; global specs still identify the layout on
a stage containing only recurrent or draft layers.

Descriptors use a layer's own page size as `layer_stride`, the common row size
as `block_stride`, and its position inside the row as `offset`. All descriptors
alias one allocation. KDA pages retain their raw Conv and recurrent state sizes.

## Kernel addresses

Attention kernels retain native 128-token pages. Their views span holes occupied
by other layers between successive manager blocks. For manager ID `b` and
kernel page `j` within that manager block:

```text
kernel_id[s] = b * (row_bytes / Q[s]) + j
slot[s]      = kernel_id[s] * 128 + token_offset_in_kernel_page
```

MLA, indexer and draft can share logical block ownership while requiring
different physical kernel IDs. The Ascend metadata path maps each attention
group's table and slots separately into persistent buffers; negative padding
slots remain negative. Indexer K and scales remain in each kernel page. Dense
MLA keeps the backend's split latent/RoPE planes or fused FlashMLA representation.

Copy and zero operations index manager rows, deduplicating all role aliases.
Mooncake V2 exchanges optional packed-row strides, keeps dense wire block IDs,
and converts those IDs at the final per-component address calculation. It
splits transfers at physical discontinuities even when individual kernel pages
are contiguous. PreemptOffload budgets and transfers one backing row and retains
Conv aliases for speculative resume correction.

## Block-size selection

Packing does not automatically choose a workload-specific optimal block size.
Supply `--block-size` as a multiple of 128 and of the model's prefill checkpoint
alignment. For P/D, also preserve Mooncake's requirement that the two manager
block sizes divide one another. The platform keeps the configured size instead
of growing a single MLA page to cover KDA. The planner validates actual row
capacity after all real specs are available.

The following are CPU capacity calculations for one TP1 rank, one 1,048,576-token
request, 24 MLA layers (656 bytes/token), 24 indexers (132 bytes/token), 69 KDA
layers, BF16 draft MLA (1152 bytes/token), and aligned state caching. Each row
is compared with the old B=12288 layout with the same draft and speculation
configuration. Candidates were scanned on the 128-token grid through 65536;
they are not universal defaults or P/D deployment recommendations.

| Draft MLA layers | Speculative tokens | Candidate B | KDA group widths | Old pool GiB | Packed pool GiB | Reduction GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 8192 | 23/23/23 | 20.1281 | 19.5097 | 0.6184 |
| 3 | 3 | 11776 | 35/34 | 26.1101 | 25.0219 | 1.0882 |
| 5 | 3 | 10752 | 35/34 | 28.7996 | 27.0038 | 1.7957 |

These figures describe the KV pool budget, including unused rows and stride
padding, rather than measured process HBM. They exclude model weights, graph
workspaces and metadata. Recompute from actual specs when any layer count,
dtype, tensor-parallel state shape, speculative window or transfer constraint
changes. For example, 11776 is not divisible by a prefill block size of 1536.

## Validation and upstream integration

CPU tests cover no draft, three and five draft layers, per-role mapped addresses,
raw state views, buffer reuse, row copying cycles, zeroing metadata, P/D copies
with different manager coverage, and PreemptOffload Conv-state restoration.
NPU operator execution, graph replay, full-model numerical equivalence, P/D
integration and measured HBM/latency remain hardware validation before merge.

This implementation reuses Ascend's existing planner and allocator extension
points and the supported `KVCacheGroupSpec`/tuple cache contract. It does not
copy the newer upstream `KVCacheGroup` API into unrelated models. The reference
is upstream mixed-page packing and hole-spanning kernel views. Once Ascend's
cache owners adopt the upstream standardized cache views, the specialized
planning/address adapters can be replaced by those upstream implementations.
