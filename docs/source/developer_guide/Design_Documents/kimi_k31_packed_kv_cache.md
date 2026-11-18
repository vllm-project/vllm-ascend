# Kimi K3.1 block-outermost KV cache

Kimi K3.1 can pack MLA, indexer and resident dense-MLA draft pages into one
physical row per scheduler block. Recurrent groups overlay the entire row,
using disjoint block IDs. A draft model is optional. Layer counts, page sizes,
dtypes and state shapes come from the actual KV cache specs.

Kimi K3.1 defaults to `BLNHC`. The existing `VLLM_KV_CACHE_LAYOUT` override
can select another block-outermost layout such as `BLHNC`. The old layer-outermost
paired-page implementation has been removed; explicit `LBNHC` selections fail
during configuration. This implementation targets Model Runner V2, packed SFA
main cache, C8 indexer K/scale storage and one latent KV head. PCP and DCP keep
their existing token/head ownership while using role-specific packed addresses.

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
history-coverage groups and KDA groups also own disjoint IDs. One state remains a complete checkpoint for
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

The worker selects packed allocation from the planned `is_block_outermost` specs,
without another model-name check. These specs survive PP projection, so packed
stages share one backing even when their local layers are not a hybrid mix.
Per-layer attention metadata needs only that layer's `is_block_outermost` flag.

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

## Context parallelism

History layers share a scheduler group only when their `block_size` and
`dcp_sharded` fields agree. In particular, a replicated draft and DCP-sharded
target require separate block tables, even though all their descriptors still
alias the same backing pool. Each history group keeps its own non-causal draft
ownership tag; recurrent groups follow all history groups.

The row geometry and view formula above are unchanged. For capacity planning,
sum the number of blocks required by every coverage group. A sharded history
group covers `B * DCP` global tokens per manager block, while a replicated group
covers `B`. Use the configured DCP domain, including PCP ranks when the parallel
configuration unifies them; do not divide every cache budget by PCP again.
KDA uses the actual head-sharded state specs and retains checkpoint block IDs.
The numerical capacity examples below assume PCP=DCP=1 and one history group.

```text
positions / sequence lengths
          |
          v
BlockTable resolves token owners -> dense local kernel IDs / gathered slots
          |
          v
per-role packed mapping -> physical kernel IDs / slots
          |
          +-> sharded SFA DCP context -> MLA operator
          +-> indexer scatter indices -> K/scale stores and TopK operator
          +-> PCP rank-ordered slots -> local prolog / gathered cache read
          +-> mapped PCP global table -> dense draft context gather
```

Sharded Kimi MLA/Indexer builders preserve those local addresses. They do not
expand physical packed IDs into the replicated-indexer layout. Token positions,
TopK token indices and sequence lengths retain their logical CP meaning.
Quantized SFA and TurboQuant prefill use local KV with query gathering and LSE
combination on the sharded-indexer path. Other models retain their replicated
indexer path.

Persistent mapping buffers allow two PCP prefill segments per request, a dummy
request and the PCP gathered-token extent. Local and global tables, and distinct
draft-step slot tensors, have independent buffers so one build cannot overwrite
another captured metadata view. The mapped PCP context replaces only the
role's global block table consumed by MLA; the scheduler's shared context stays
unchanged.

The scheduler's token budget is global before PCP partitioning. The slot array
passed to the mapper, however, follows the rank-major gathered layout, including
replicated requests and padding. For example, with PCP=2 and DCP>1, four decode
tokens can occupy eight slots: the second rank's duplicate entries remain in
the array even when masked to `-1`. Slot-buffer capacity therefore includes the
PCP factor; it does not increase the scheduler's token budget. Both local and
global mapping buffers currently use this conservative capacity bound.

Mooncake mixed pools use group-aware transfer metadata instead of the legacy
single-group replicated-indexer expansion. Dense wire IDs are still converted
to physical packed pages at the final copy boundary. Row copy/zero and native
KDA/RecoverSSM state strides retain their existing contracts.

## Block-size selection

When `--block-size` is omitted, startup searches capacity candidates after the
target and optional draft layers are loaded. It uses their actual cache specs,
including scale dtype, draft residency, KDA state shapes and RecoverSSM's
persistent state count. Explicit `--block-size` values are preserved and still
must satisfy native-page and prefill-checkpoint alignment.

Candidates contain whole 128-token kernel pages and satisfy checkpoint,
prefix-match-unit and configured CP-interleave alignment. The lower bound uses
the model's packed-row sizing with reference TP16; the search stops at 65536
tokens. The objective is the smallest worst-stage planner demand at the configured
maximum context, including one null row. All PP stages exchange their actual
specs and checkpoint alignments over the PP CPU group, group globally and score
their projected rows, so recurrent-only stages also select the same block size.
This is a cache-capacity objective, not a measured latency or workspace optimum.
If the search range cannot satisfy the alignments, startup asks for an explicit B.

Mooncake V2 pull (`MooncakeConnectorV2` / `MooncakePullConnector`) supports this
search independently on P and D. Its full-attention transfer maps token positions
into each peer's kernel pages: the two manager block sizes need not divide one
another. Both peers must use the same kernel page size; each CP interleave must
be a multiple of that size and divide its own manager block. Other KV connectors
require explicit B until their transfer contracts are qualified. Numerical/graph
and full-service validation remain separate from this startup capacity search.

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

## RecoverSSM integration

The capacity examples above use RecoverSSM disabled. With the merged Kimi KDA
RecoverSSM path (`--use-replayssm`), the planner consumes the actual
`MambaSpec.num_speculative_blocks=0` instead of reserving one persistent state
block per speculative token. The per-checkpoint Conv/recurrent shapes still
come from the model specs; three-token speculation still includes the Conv
working window. With five draft layers and B=10752, aligned state caching uses
98 history rows, four state rows and one null row: 25.5174 GiB of KV pool
instead of 27.0038 GiB. Request-indexed verification records are separate
allocations outside this budget, so this difference is not a process-HBM
measurement.

RecoverSSM receives the actual Conv and recurrent block strides from the packed
views. PreemptOffload saves the committed checkpoint at its accepted token
position, including exact block boundaries and partial prefill, and restores
it without applying the native speculative Conv shift again. Temporary verify
records are regenerated after resume rather than copied with persistent KV.

RecoverSSM's configuration validation, including DS state-layout rejection for Mooncake P/D, remains in
effect. Combined packed/RecoverSSM NPU execution, graph replay and P/D
equivalence require hardware validation.

## Validation and upstream integration

The feature branch is synchronized with Ascend main's main2main update #21005,
which pins vLLM `ba15685ab7f4f5fbed8630bbb7441f8837e76fbe` and retains the
`v0.33.0` release lane. Both versions provide the grouping and block-count
helpers in `kv_cache_planning`; the retired `v0.32.0` import branches are removed.
The new main keeps the resolved token-size union in `cudagraph_capture_sizes`,
so the packed metadata buffer sizing still covers split prefill/decode graphs.
Ascend's synchronized allocation and binding hooks handle main's `allocate`
argument and the changed ReplaySSM tracker ownership. Extensible KV allocation
remains unsupported on Ascend. This synchronization was checked statically;
runtime testing remains stopped, and the results below belong to the older
revision rather than the updated main2main baseline.

CPU tests cover no draft, three and five draft layers, per-role mapped addresses,
raw state views, buffer reuse, row copying cycles, zeroing metadata, P/D copies
with different manager coverage, and PreemptOffload Conv-state restoration.
On Ascend A3, 816 affected framework CPU tests passed (14 skipped) against vLLM
`4ba787ec9959e566d3b587a796dff07673324da5`. Three added real NPU tests passed:
packed main-cache store/PCP read/sparse attention, indexer K/scale store/TopK,
and native KDA Conv/recurrent updates, compared with contiguous storage.

Two-rank DCP and PCP+DCP numerical/graph tests are included. Their execution was
stopped at the user's request; no passing multi-rank or graph result is claimed.
Full-model equivalence, distributed P/D integration and measured HBM/latency
remain unverified. This evidence is operator/framework validation, not a
complete model-serving qualification.

This implementation reuses Ascend's existing planner and allocator extension
points and the supported `KVCacheGroupSpec`/tuple cache contract. It does not
copy the newer upstream `KVCacheGroup` API into unrelated models. The reference
is upstream mixed-page packing and hole-spanning kernel views. Once Ascend's
cache owners adopt the upstream standardized cache views, the specialized
planning/address adapters can be replaced by those upstream implementations.
