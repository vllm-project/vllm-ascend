# 310P DFlash Piecewise graph-input contract

This contract applies only when `is_310p_dflash_piecewise(vllm_config)` is
true. The ACL graph wrapper validates every retained tensor recursively in
DEBUG before replay. A mismatch is fatal; it is never converted to eager.

| Retained role | Producer / owner | Update before replay | Consumer | Lifetime and view rule | Alignment source |
| --- | --- | --- | --- | --- | --- |
| Target `input_ids` / `positions` | 310P model-runner persistent input buffers | Bounded in-place copy for the selected padded descriptor | Compiled target backbone | Same base storage, offset, shape, stride, and pointer as capture | Tensor dtype element size |
| Draft `input_ids` / `positions` | spec-decode proposer persistent buffers | AscendC expansion result copied in place | Compiled DFlash draft backbone | Descriptor-bounded slice of persistent proposer storage | Tensor dtype element size |
| DFlash hidden/context/query inputs | proposer hidden/context/query buffers | Bounded in-place copy before draft forward | DFlash projection and attention partitions | No per-request replacement after capture | Tensor dtype element size |
| Draft RoPE cos/sin | `_draft_cos` / `_draft_sin` in `ops/rotary_embedding.py` | Selected values copied in place | Draft rotary partitions | Capacity is reserved from `runner.max_num_tokens` before exact-scope profile/capture; runtime may only take bounded views | Tensor dtype element size; `npu_apply_rotary_pos_emb` consumes the persistent slice |
| Model parameters and RoPE cache | model modules | Immutable during serving | Compiled target/draft partitions | Model lifetime; no storage replacement | Tensor dtype element size |
| Attention Q/K/V/output graph edges | captured FX partition / graph pool | Produced inside the captured graph | Adjacent partition or split attention op | Capture-owned storage; external callers must not replace a retained edge | Tensor dtype element size |
| KV-cache tensors | attention layers | Slot-indexed cache update | Split attention custom op | Layer lifetime; cache base storage and physical layout are immutable | Allocator/custom-op contract; not inferred from a logical token index |
| Slot mapping, block table, sequence/query lengths | runner/proposer metadata buffers | In-place metadata update before `set_forward_context` | Split attention custom op | Persistent bounded buffers; any tensor exposed to a retained partition is recursively validated | Tensor dtype element size; custom-op-specific cache layout is validated by its producer |
| Rejection counts when absent | proposer `_zero_num_rejected_buffer_310` | Zeroed in place, then sliced to request count | Eager AscendC input builder | Exact-scope persistent buffer bounded by `runner.max_num_reqs`; AscendC outputs are copied to retained proposer buffers | Tensor dtype element size |
| Masks and attention workspaces | attention metadata builder / layer workspace | Descriptor-specific in-place update where retained | Split attention op | PIECEWISE split op remains an explicit eager boundary; any tensor crossing back into a retained partition is covered by recursive validation | Consuming attention implementation; no guessed stronger alignment |
| Token-selection indices | proposer / sampler buffers | Bounded in-place update | Draft sampling and verification | Eager sampling boundary unless present in a retained partition; retained instances are recursively validated | Tensor dtype element size |

The recursive contract records tensor path, `data_ptr`, base storage pointer,
storage offset and accessible byte bounds, dtype, shape, stride, contiguity,
device, and natural alignment. Host-visible metadata only is used: collection
must not call `Tensor.item()` or introduce an NPU synchronization.

Allocating conversions inside the eager AscendC builder are not graph-retained
inputs. Its outputs must be copied into the persistent proposer buffers listed
above before the compiled DFlash forward is entered. A future custom-op change
that retains one of those temporary inputs must first declare its operation-
specific alignment and ownership here and add an address-lifetime regression.
