# Sparse KV Offload: offload_shared_indexer_only (layered-indexer models such as GLM-5.2)

## Background and motivation

Sparse KV offload places the main MLA KV cache in a host DRAM pool. During
decode, the fused attention operator attends the top-k selection while
onboarding missed rows from the host pool **inside the attention kernel**
(host reads in the critical path). This enlarges the schedulable KV
capacity, but every offloaded layer pays the miss-row onboarding cost.

For **full/shared layered-indexer** models such as GLM-5.2 (the
`indexer_types` field marks *full* layers that compute the top-k and
*shared* layers that directly reuse the top-k of the full layer in their
group), this optimization introduces
`sparse_kv_offload_config.offload_shared_indexer_only`. It is supported on
both attention pipelines — the fused Copy-SFA path (the default whenever
the offload is enabled) and the `use_fused_overlap` overlap-operator path:

- **Full (indexer-owner) layers are not offloaded**: their KV stays resident
  in the NPU paged caches, and their decode write and attention paths are
  identical to the non-offload flow (no miss onboarding, no D2H). Owners
  with shared members still produce their group's selection — the LIM
  operator on the fused Copy-SFA path, the external LRU plan **at owner
  attention time** on the fused_overlap path (run for the group's first
  member layer, which is registered in the offload table) — for the shared
  members to reuse; singleton owners keep the standard lightning-indexer
  flow.
- **Shared layers stay offloaded** — miss handling follows
  `miss_onboard_mode`:
    - `inline` (default): the fused attention operator reads the host pool
    directly inside the kernel according to the miss plan.
    - `side_stream`: during the resident owner's forward, one pre-copy
    launch per shared member moves the group's **landed prefix misses**
    into the resident rows on a side stream, hidden behind the owner's
    SFA + MoE compute (`npu_lim_copy_descriptors` into the per-request
    hot rows on the fused Copy-SFA path, `npu_lim_copy_descriptors_linear`
    into the selection buffer on the fused_overlap path). The pre-copied
    rows are encoded as hits in the plan, so the fused operators never
    re-copy them: fused Copy-SFA members run with an all-zero miss plan
    (pure direct-addressed HBM attention) and fused_overlap members read
    the selection rows without the inline host-pool gather. The
    speculative suffix (this step's tokens whose CPU-pool writeback is
    still in flight at pre-copy time) always stays on the inline path —
    its data has landed by the time the member kernel reads it. Falls
    back to `inline` with a warning when the operators are not registered
    in the installed `vllm_ascend_C` build.
- **The MTP draft layer is device-resident** together with the full layers
  on both pipelines: the draft pass drops its pool writeback, plan reuse
  and selection-buffer flow entirely and attends its NPU paged cache
  through the standard kernel. This removes the per-step draft onboarding
  cost and keeps draft / target agreement high (measured acceptance
  74.8%, vs 73.0% with every layer offloaded).

Two plan-path optimizations keep the owner-time plan off the critical path
(fused_overlap pipeline):

- **Shared metadata batching**: `req_ids` / `stable_prefix_lens` /
  `visible_seq_lens` are identical for every group plan in a forward pass
  (only the top-k differs per group), so they are staged to the CPU once
  per pass instead of once per plan.
- **Deferred plan join**: a full layer never reads the group plan — the
  first consumer is the group's first member, whose inject joins
  `fused_plan_stream`. The plan therefore has the owner's attention + MoE
  window to complete.

Compared with offloading every layer, the offloaded amount drops from 79
to 57 layers (GLM-5.2: 22 full + 57 shared; the MTP draft layer joins the
resident set). The DRAM pool and the hot-row / selection buffers shrink
accordingly, while the NPU side gains the resident layers' paged caches
(their `store_on_host=False` specs count against the standard NPU budget
automatically).

**Measured** (GLM-5.2-w4a8c8, single-node PD, serial bs1 decode with MTP,
64k context, x30 windows): TPOT 32.84ms vs 34.44ms with every layer
offloaded (-4.6%), acceptance 74.8% vs 73.0%.

## Configuration

```json
{
  "sparse_kv_offload_config": {
    "enabled": true,
    "use_fused_overlap": true,
    "offload_shared_indexer_only": true,
    "topk_buffer_size": 4096,
    "dram_size_per_dp_GB": 256
  }
}
```

`offload_shared_indexer_only` selects which layers are offloaded (all
layers vs. shared layers only) and `miss_onboard_mode` selects how the
offloaded layers' misses are onboarded (`inline` | `side_stream`).
Omit `use_fused_overlap` to run the same layer split on the fused Copy-SFA
pipeline. No additional environment variables are introduced.

## Implementation notes

The wiring is symmetric across both pipelines; the V2 model runner carries
it.

- `vllm_ascend/ascend_config.py`: adds `offload_shared_indexer_only`
  (accepted on both pipelines).
- `vllm_ascend/worker/v2/model_runner.py`:
    - `_bind_sparse_offload_layer_groups` (after model load): binds
      attention layer names from the forward context (no in-tree model
      passes `layer_name` into the attention impl), classifies MTP layers,
      builds the shared-indexer groups from the shared
      `topk_indices_buffer` identity and registers `_lim_share_members`.
    - Full layers and the MTP draft layer get `store_on_host=False` in
      the MLA attention spec (their paged caches count against the NPU
      budget; allocation and reshaping take the standard branches).
    - Passes the full-resident layer list to the manager.
- `vllm_ascend/attention/sfa_kv_offload.py`:
    - `is_full_resident_layer`: shared-only classification (full layers
      and the MTP draft layer resident; shared layers offloaded).
    - Full layers run `exec_kv` / attention through the standard base-class
      path (with SP-padding of dummy top-k rows for sequence-parallel
      MoE), and owners with shared members run the group plan at their
      attention forward via `_run_group_plan_for_members`, right after the
      indexer filled the shared top-k buffer.
- `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_offload_manager.py`:
    - Registers only the offloaded layers and publishes the full layers'
      HBM paged-cache base addresses for direct PD pull writes.
    - `prepare_fused_overlap_external_plan` / `_run_fused_overlap_plan_async`:
      the async plan stream with per-pass shared-metadata batching and the
      `skip_save_wait` mode for owner-time plans (the owner has no
      writeback in flight, so ordering the plan after
      `current_kv_save_stream` is both unnecessary and un-capturable — an
      event wait without a matching record is rejected by ACL graph
      capture).
    - The plan's already-run guard uses the ForwardContext identity so the
      first member re-entering prepare after the owner-time plan never runs
      the planner twice (a second run would observe its own LRU updates and
      emit an all-hit plan over slots that were never gathered).
- `vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/`: the PD pull
  targets full layers at their rank-local HBM paged caches (every rank
  pulls the full contributor block list, no D-TP split), while shared
  layers keep pulling into the TP-shared host pool with the existing D-TP
  split.

## Known limitations

- Keeping full layers resident raises the NPU memory cost per KV block, so
  the schedulable block count drops for the same device memory; trade
  batch/context capacity against TPOT as needed.
