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
    attention.

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
    - Shared-layer attention waits on the group event. The decode D2H of new
    KV follows the standard per-layer `offload_new_kv` commit.
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

## Performance

### Environment

| Item | Value |
|---|---|
| Base info | two-node A3 machine, GLM-5.2-w4a8c8, graph mode, no MTP, topK=4096 |
| Bench | 64K input / 256 output, prefixhit dataset (~91% prefix cache hit) |
| D node | 1TP16DP+EP, 16 NPUs, v2 runner, util=0.95 |
| P node | 8TP2DP+EP, 16 NPUs, v2 runner |
| Scenarios | B = full offload(fused_copy_sfa); C = shared-indexer-only + side_stream (new feature) |

| case | B TPOT(ms) | C TPOT(ms) | B tput(tok/s) | C tput(tok/s) |
|---|---|---|---|---|
| bs1 | 62.4 | **58.8** | 5.8 | 6.2 |
| c4 | 65.1 | **61.1** | 47.3 | 49.5 |
| c8 | 66.7 | **62.6** | 74.5 | 78.5 |
| c16 | 68.0 | **64.3** | 106.2 | 107.7 |
| c24 | 69.5 | **65.2** | 114.3 | **117.8** |

Key Findings: C consistently outperforms B by ~6%.

### Memory (per rank, `max-model-len=65536`)

| Item | All layers offloaded | This optimization |
|---|---|---|
| Host DRAM pool | 21.95 GiB | **9.84 GiB** |
| NPU KV budget (indexer + resident layers) | 1.31 GiB | 4.43 GiB |
| Hot-row topk buffers | 4.37 GiB (78 layers) | **3.19 GiB (57 layers)** |
| Schedulable blocks | 2049 (workload-bound) | 1257 (NPU-bound) |

## Known limitations

- Requires the fused Copy-SFA path (the default); `use_fused_overlap` is not
  supported.
- `miss_onboard_mode=side_stream` requires `offload_shared_indexer_only=true`.
- Keeping full layers resident raises the NPU cost per KV block by roughly
  3.3x versus offloading everything, so the schedulable block count drops for
  the same device memory (2049 -> 1257 blocks in this test); trade
  batch/context capacity against TPOT as needed.
