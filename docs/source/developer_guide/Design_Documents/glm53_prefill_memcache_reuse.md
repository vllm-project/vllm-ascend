# GLM-5.3-Flash Prefill Memcache reuse: experimental implementation

This implementation adds shared main-cache allocation and recurrent-state
continuation to `AscendStoreConnector` with the Memcache GVA backend. It is a
Prefill implementation for hardware validation, not a validated P/D deployment.
The sparse Decode and `SfaRemoteD2HConnector` instructions in the
[offloading guide](../../user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md)
do not apply to GLM-5.3-Flash.

## Supported configuration for initial validation

- Full GLM-5.3-Flash BF16 model; eager execution and no speculative decoding.
- `mamba_cache_mode=align`, no speculative state blocks, prefix caching enabled.
- PP, DP, PCP and DCP are all 1. TP may shard KDA state across ranks.
- `prefix_match_unit` is unset or equals the aligned KDA block size.
- One saving `AscendStoreConnector`, using Memcache and layerwise transfer.
- `VLLM_ASCEND_LW_DEFER_LAST_SAVE=0` (the default).

MultiConnector/P2P handoff and sparse Decode offload are deliberately rejected
for this first shared-buffer implementation. In particular, retaining a private
tail on Prefill does not implement transferring that tail to a Decode worker.
Do not advertise the producer as ready for an existing P2P deployment.

Prepare Memcache using the Prefill-node instructions in the offloading guide.
For an isolated Prefill validation run, add these options to the model's normal
launch command (select TP according to the full BF16 model's capacity):

```bash
--enforce-eager \
--enable-prefix-caching \
--mamba-cache-mode align \
--kv-transfer-config '{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "use_layerwise": true,
    "layerwise_num_shared_buffers": 3,
    "layerwise_independent_layers": [0]
  }
}'
```

Use requests with `max_tokens=1` for Prefill measurements. Colocated generation,
graph execution and MTP throughput are outside this validation target.

## Allocation and lifecycle

The logical groups and global block IDs remain unchanged:

| Group | Cache entries in the 45-layer model | Residency |
| --- | --- | --- |
| G0 | 11 main MLA + 11 compressed indexer | Main rotates; indexer stays resident |
| G1 | 11 private KeyPool tails | Resident, never a shared prefix object |
| G2 | KDA layers 0, 4, ..., 44 | Layer 0 independent; remaining layers rotate |
| G3 | KDA layers 1, 5, ..., 41 | Rotate |
| G4 | KDA layers 2, 6, ..., 42 | Rotate |

With three shared buffers and independent layer 0, main allocations change from
12 to 7. The original 11 indexer/tail backing allocations are retained whole.
Descriptors keep `layer_stride=0`, their padded page stride and the same global
block capacity. For fixed capacity N, main page bytes P and small page bytes S,
allocation changes from `N * (12P + 11S)` to `N * (7P + 11S)`. This is raw-cache
accounting; allocator alignment, workspaces and weights are additional.

The existing layerwise sender waits for the layer's NPU completion event before
copying its component slices. A following user of the same backing waits for
that source copy to finish and for the producer's NPU event, even if it has no
GET. It then restores its own data before entering the layer. Object publication
remains after all component/layer writes complete. Source-buffer release and
object publication are separate operations.

The final layer drains pending saves. This also protects step-begin zeroing of
newly allocated global IDs on the next step. Live groups use disjoint IDs;
existing KDA initialization and align pre-copy run after the layer load hook.
With no speculative state, align postprocessing has no state relocation to do:
the running state is already at `ceil(end_token / block_size) - 1`.

## Recurrent endpoints, partial chunks and prefix hits

KDA writes one running conv/SSM state at the current endpoint. A non-null old
state ID is not proof that its bytes still belong to the current layer after
backing reuse. Accordingly, a layer publishes a full KDA checkpoint only when
the computed endpoint is aligned, and only for that endpoint's state block.
Historical immutable checkpoints remain in Memcache without being rewritten.

An unaligned endpoint uses the existing request-scoped partial key and GVA copy
path. Both conv and SSM are copied with their real component lengths and padded
device strides. The object has full component capacity; it is not a shortened
token tensor. A new request generation gets a UUID namespace, so a restarted
request ID cannot read a previous generation's partial state. Full prefix keys
retain the existing group/hash/rank format.

On continuation, KDA restores only the exact preceding endpoint. The existing
per-layer align pre-copy then moves that state into the next running block, if
necessary. MLA restores its required history and partial page. Indexer state
shares G0's logical endpoint; the G1 tail stays resident between chunks. On a
new request or preemption, restore a legal complete checkpoint and replay the
suffix to rebuild the private tail.

Prefix probing is capped at a block-aligned position before the final token.
This checks whether the actual replay checkpoint exists, rather than accepting
a full hit and trimming it to a possibly absent earlier KDA state. The existing
Mamba scheduler is responsible for producing aligned chunk checkpoints.

Local prefix metadata is retained, but shared-layer bytes can have been
overwritten. A local hit therefore requires complete external restore coverage.
Missing external coverage, missing state IDs, failed allocation or failed
transfer stop execution instead of feeding incomplete state to the model.
Automatic recovery after Memcache eviction is not implemented in this version.

## Validation and remaining work

The CPU tests use real Torch views over aliased backing storage and actual
`ctypes.memmove` copies through the existing GVA sender/receiver threads. They
cover the 45-layer allocation, unchanged groups, unaligned/aligned chunk
continuation, endpoint selection, snapshot generations and failure guards.
NPU events and the backend service are mocked; these tests do not establish
NPU or Memcache-service correctness.

Before merge, run on A3 against an identical no-reuse baseline (omit
`layerwise_num_shared_buffers`): cold requests, complete and partial prefix
hits, consecutive requests, forced preemption, and small concurrent batches.
Include short, 32K and 128K total contexts, unaligned chunk boundaries, every
TP rank's conv/SSM payload, transfer failures and an evicted external checkpoint.
Compare generated tokens and device cache allocation at the same global block
capacity. Record software versions, commands, allocation bytes and results.
P2P integration needs separate component-stride and endpoint validation before
the MultiConnector restriction can be removed.
