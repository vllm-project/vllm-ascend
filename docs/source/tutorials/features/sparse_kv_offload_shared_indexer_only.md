# Sparse KV Offload: offload_shared_indexer_only (layered-indexer models such as GLM-5.2)

## Background and motivation

Sparse KV offload places the main MLA KV cache in a host DRAM pool and, during
decode, moves the rows missed by the LIM top-k selection back into HBM hot
rows. This enlarges the schedulable KV capacity, but the miss-row HBM copyback
degrades TPOT.

For **full/shared layered-indexer** models such as GLM-5.2 (the `indexer_types`
field marks *full* layers that compute the top-k and *shared* layers that
directly reuse the top-k of the full layer in their group), this optimization
introduces `sparse_kv_offload_config.offload_shared_indexer_only` (it requires
the fused Copy-SFA path, which is the default whenever the offload is enabled,
and is mutually exclusive with `use_fused_overlap`):

- **Full (indexer-owner) layers are not offloaded**: their KV stays resident in
  the NPU paged caches, and their decode write and attention paths are
  identical to the non-offload flow (no miss copyback, no D2H), so TPOT sees no
  regression on this path. Owners with shared members still run the LIM
  operator to produce the top-k/miss plan for their group; singleton owners
  (no shared members) fall back to the standard lightning-indexer selection.
- **Shared layers stay offloaded** (miss handling follows the fused mode):
    - `miss_onboard_mode=side_stream`: right after the group owner's LIM, the
    whole group's miss copyback is issued on a **side stream**, hidden behind
    the owner's SFA + MoE compute. One `npu_lim_copy_descriptors`
    (`aclnnLimCopyDescriptors`) launch per member resolves the miss plan and
    performs the host-pool -> hot-row copy entirely in-kernel (zero Python
    side-stream pre-copy, zero engine calls); each member waits only on its own
    completion event. The member layers then invoke the fused Copy-SFA
    operator with an **all-zero miss plan**, which degenerates into pure
    direct-addressed HBM attention (no host reads, no slot-list
    materialization or indirection). When the operator is not registered in
    the installed `vllm_ascend_C` build, the path automatically falls back to
    `inline` behavior.
    - `miss_onboard_mode=inline` (default): the fused Copy-SFA operator reads
    the host pool directly inside the kernel according to the LIM miss plan —
    no slot-list materialization and no pre-copy needed.

Compared with offloading every layer, the offloaded amount drops from 78 to 57
layers (GLM-5.2: 21 full + 57 shared). The DRAM pool and the hot-row buffers
shrink accordingly, while the NPU side gains 21 layers of paged caches (the
memory planner accounts for them in the NPU budget automatically).

## Configuration

```json
{
  "sparse_kv_offload_config": {
    "enabled": true,
    "offload_shared_indexer_only": true,
    "miss_onboard_mode": "side_stream",
    "topk_buffer_size": 4096,
    "dram_size_per_dp_GB": 256
  }
}
```

- `offload_shared_indexer_only` selects which layers are offloaded (all
  layers vs. shared layers only).
- `miss_onboard_mode` selects how miss KV rows are copied:
    - `inline` (default): handled inside the fused operator; no side stream.
    - `side_stream`: side-stream group pre-copy plus direct-addressed
    attention. If `npu_lim_copy_descriptors` is not available in the installed
    `vllm_ascend_C`, the path falls back to `inline` automatically (capability
    detection, not a user switch).

No additional environment variables are introduced.

## Implementation notes

Both model runners (V1 and V2) support this optimization; the wiring is
symmetric. The fused Copy-SFA pipeline is the default attention path whenever
the offload is enabled; `use_fused_overlap` selects the alternative overlap
operator instead, and `offload_shared_indexer_only` rejects that combination
outright (see the validation in `vllm_ascend/ascend_config.py`).

- `vllm_ascend/ascend_config.py`: adds `offload_shared_indexer_only` and its
  validation (incompatible with `use_fused_overlap`).
- `vllm_ascend/worker/v1/model_runner.py` and
  `vllm_ascend/worker/v2/model_runner.py`:
    - Bind attention layer names from the forward context after model load
    (no in-tree model passes `layer_name` into the attention impl), build
    the LIM selection groups (owner -> members) and register
    `lim_indexer_owner` / `_lim_share_members`.
    - Full layers get `store_on_host=False` (their paged caches count against
    the NPU budget; allocation and reshaping take the standard branches).
    - Pass the full-resident layer list to the manager and configure the
    group topology.
- `vllm_ascend/worker/v2/attn_utils.py`: the MLA attention spec applies the
  same `store_on_host=False` carve-out for full-resident layers.
- `vllm_ascend/attention/sfa_kv_offload.py`:
    - Full layers run `exec_kv` / attention through the standard base-class
    path.
    - Only owners with shared members install the LIM selection hook; after the
    LIM they launch the group's side-stream copy.
    - Shared-layer attention waits on the group event. The decode D2H
    descriptors of new KV are batched per group into one
    `offload.sparse_copy` call per group.
- `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py`:
    - Registers only the offloaded layers (shared + MTP) and publishes the full
    layers' HBM paged-cache base addresses for direct PD pull writes.
    - `configure_lim_groups`: side stream + events + the deployment-constant
    `params` tensor (`block_size`, hot tokens, miss stride, `kv_lora_rank`,
    `qk_rope_head_dim`), all created before graph capture. The model's head
    dims are validated against the kernel's UB staging capacity so an
    unsupported MLA layout fails at startup instead of copying the wrong
    byte ranges.
    - `launch_lim_group_onboard`: rewrites the per-step `active_geometry`
    tensor (`[active_rows, block_table_width]`) and launches one
    `npu_lim_copy_descriptors` per member. `active_rows` bounds the kernel's
    request loop to the current batch, so stale miss counts left beyond the
    live request count by an earlier, larger step are never read (no explicit
    zeroing is needed), and `block_table_width` is the live block-table row
    stride, so it stays correct even when the engine reserves encoder tokens.
- `vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/`: the PD pull
  targets full layers at their rank-local HBM paged caches (every rank pulls
  the full block list, no D-TP split), and the hot-row tail D2D is skipped for
  full layers.

## Performance (GLM-5.2-w4a8c8, PD-disaggregated TP8+TP8, 64K input / 128 output, single stream, graph mode, no MTP)

Measured on the same day and environment (`vllm bench serve`, custom
longbenchv2 64K set):

| Mode | Mean TPOT (ms) | Notes |
|---|---|---|
| No offload (all KV resident in HBM, reference) | 47.08 | historical baseline |
| fused_copy_sfa (all layers offloaded, reference) | 56.07-57.10 | previous best offload mode |
| All layers offloaded | 68.39 | re-measured same day, before this work |
| shared_only (slot-list attention, intermediate state) | 59.72-59.75 | -12.6% |
| shared_only (direct addressing, intermediate state) | 55.60-55.79 | -18.5% |
| fused_copy_sfa + offload_shared_indexer_only | 54.50-54.97 | -19.6% |
| **shared_only + side_stream (direct addressing + KSC group, current best)** | **53.95-54.57** | **best: -20.5%** |

The per-member onboarding operator strides the flattened
`(active_rows, miss_stride)` space across all AIV cores (one request's misses
are spread over blocks instead of serializing on one), bounds the traversal by
the live `active_rows` (a shrunken batch never reads the previous step's
larger stale miss counts), and copies only the entries below each request's
real `miss_counts`; empty slots are skipped by a cheap count test, so the copy
volume tracks the real miss plan.

Memory (per rank, `max-model-len=65536`):

| Item | All layers offloaded | This optimization |
|---|---|---|
| Host DRAM pool | 21.95 GiB | **9.84 GiB** |
| NPU KV budget (indexer + resident layers) | 1.31 GiB | 4.43 GiB |
| Hot-row topk buffers | 4.37 GiB (78 layers) | **3.19 GiB (57 layers)** |
| Schedulable blocks | 2049 (workload-bound) | 1257 (NPU-bound) |

Compared with no offload, the HBM footprint of the main KV per block drops to
about 1/3.3 (57/78 layers moved to DRAM), so the same free device memory
sustains larger batches or longer contexts.

## Known limitations

- Requires the fused Copy-SFA path (the default); `use_fused_overlap` is not
  supported.
- `miss_onboard_mode=side_stream` requires `offload_shared_indexer_only=true`.
- Keeping full layers resident raises the NPU cost per KV block by roughly
  3.3x versus offloading everything, so the schedulable block count drops for
  the same device memory (2049 -> 1257 blocks in this test); trade
  batch/context capacity against TPOT as needed.
- A new request scheduled right after a long request finishes may
  occasionally see degraded first-token quality (pool slot reuse window). This
  is tracked as a correctness bug, not a tuning knob: one known cause was the
  onboarding operator replaying a stale miss plan after the batch shrank, which
  is fixed by bounding the launch to `active_rows`. If
  it still reproduces, please capture the request ids / pool slots and
  `lim_misses` content around the first decode step and open an issue.
