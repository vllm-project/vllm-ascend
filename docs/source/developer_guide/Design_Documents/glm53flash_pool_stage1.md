# GLM-5.3-Flash pool and PD integration

This document describes the experimental `LayerwisePoolConnector` and the
shared-format ordinary Ascend Store integration. Functional validation covers
GLM-5.3-Flash BF16, TP8/EP8, PP=PCP=DCP=1, MRV2 eager and text requests.
P/D use matching MTP settings: disabled on both sides or one speculative token
on both sides. It is not a production-readiness or dataset-accuracy claim.

## Configuration contract

Both ordinary pooling and PD must use the same compatible target schema,
model weights/epoch (`cache_prefix`) and `store_tp_size`. PD uses:

```json
{
  "kv_connector": "LayerwisePoolConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "store_tp_size": 8,
    "cache_prefix": "your-model-weight-epoch",
    "use_layerwise": true,
    "load_async": false,
    "save_decode_cache": true
  }
}
```

Use `kv_consumer` for D. Ordinary prefix pooling uses `AscendStoreConnector`
with the same extra configuration and the appropriate role. `use_layerwise`
may differ between endpoints without changing object identity.

With graph execution enabled, the shared format performs bulk I/O at the
runner's step boundaries. D restores the PD cache and copies the loaded Mamba
state before graph replay. Decode writeback fences the completed step before
copying and publishing committed cache. An eager P can continue to write
layer by layer into the same pool.

The validated serving configuration used `--tensor-parallel-size 8`,
`--enable-expert-parallel`, `--enforce-eager`, `--enable-prefix-caching`,
`--mamba-cache-mode align`, `--max-model-len 1024` and
`--max-num-batched-tokens 640`. MRV2 was selected with
`VLLM_USE_V2_MODEL_RUNNER=1`. The effective logical block size was 640 even
when the CLI block-size input was 128; inspect actual cache specs rather than
assuming the CLI value is the final layout. MTP=1 used
`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`.

MemCache capacity, transport endpoints and device assignments are deployment
configuration, not connector defaults. Configure them for the actual cluster.

## Real cache binding

The adapter derives ranges from registered tensors and specs. It does not
change Flash attention, KDA, model forward or speculative algorithms.

| Component | Observed local layout at TP8 | Sharing behavior |
| --- | --- | --- |
| MLA latent KV | BF16 kernel pages `(B, 128, 1, 512)`, multiple kernel pages per logical block; empty rope component | Replicated content; rank zero writes and every rank reads |
| Compressed indexer | BF16 `(B, 32, 1, 128)` with compression ratio 4 | Copy complete compressed rows, not an invalid partial row |
| KDA conv | BF16 `(B, 3, 3072)` without MTP, `(B, 4, 3072)` with MTP=1 | Bind physical page stride; restore committed checkpoint |
| KDA recurrent | FP32 `(B, 8, 128, 128)` | Component offset changes with conv width; use actual views |
| Circular indexer tail | FP32 `(B, 4, 256)` or `(B, 5, 256)` with MTP=1 | PD-only request state; copy content without rotating or copying page padding |

Observed KDA page size was 655360 bytes, with recurrent offsets 18432 or
24576 bytes depending on MTP. Circular page stride was 40960 bytes. These are
validation observations, not hardcoded model dimensions in the generic adapter.
The native byte tests retained these shapes/strides and reduced only the block
count; model tests separately exercised real cache allocation.

## Request flow

For an 800-token prompt, the stateful path uses a P boundary of 799 tokens:

```mermaid
sequenceDiagram
    participant Proxy as Original CPD proxy
    participant PS as pd_pool/PoolPDScheduler
    participant PW as pd_pool/PoolPDWorker
    participant Pool as shared_transfer + shared_io / MemCache
    participant D as D connector / MRV2
    Proxy->>PS: P request, do_remote_decode
    PS->>PS: on_new_request: capture original digest, truncate to N-1
    PS->>Pool: Ordinary prefix lookup and full-block plans
    PW->>Pool: save_layer or finish: ready full blocks and PD tail/state
    Note over PW,Pool: 640 full tokens + 159-token tail; exact state boundary 799
    Pool-->>PS: Saves complete; checkpoint completion events
    PS-->>Proxy: request_finished: compatible CPD handoff
    Proxy->>D: Same prompt and forwarded kv_transfer_params
    D->>Pool: PoolPDWorker.start_load: restore full receive plan
    Pool-->>D: All target components restored
    D->>D: Recompute final prompt token; initialize local draft; decode
```

Full blocks are shared with ordinary pooling. Ordinary pooling does not write
or look up partial tails. PD final tails are scheduled once the final prompt
chunk reaches the handoff boundary; this is distinct from ordinary full-block
save scheduling. Objects remain in the pool after handoff.

Layerwise writes enqueue `StoreLayerTask` on the existing
`KVCacheStoreLayerSendingThread`. The forward thread records a fresh ready
event and returns; the sender waits for readiness and copies full-block and
PD tail/state ranges. `wait_for_save` drains the queue before publishing,
returning checkpoint completion events or allowing the next step to overwrite
live state. Abort and shutdown also drain before releasing leases or buffers.
This enables inter-layer overlap without enabling deferred cross-step writes
or shared-HBM reuse. It does not change MTP checkpoint selection or commit timing.
Post-change remote tests cover blocked copies, drain/abort and asynchronous
failure; earlier model acceptance does not establish this change's NPU overlap
or throughput, which still needs a new hardware run.

If D later crosses a committed full-block boundary, `save_decode_cache` can
publish that checkpoint. A later P request looks up the same objects, restores
the common attention/state boundary, and skips the corresponding prefix work.
Ordinary-prefix circular buffers use aligned replay; arbitrary PD boundaries
also require their exact circular snapshot.

## MTP and ownership

P does not ask D to execute a draft model on its behalf. D performs its own
MTP work. The compatibility field checks equal speculative-token counts
because this Flash configuration changes target conv and circular storage as
well as adding draft caches. Different MTP configurations are rejected.

`shared_draft_names` and `target_group_specs` exclude draft owners without
dropping target owners that share a grouped spec. A worker-generated manifest
keeps scheduler group selection consistent with actual registration.

One metadata-only change outside PD/pool is necessary:
`worker/v2/attn_utils.py::get_kv_cache_spec` propagates an owner's
`is_draft_layer` flag after Ascend spec conversion/padding, when the spec
supports `is_draft`. Otherwise reconstructed auxiliary specs can lose ownership.
No model code, `build_layer_cache_bundles`, allocator algorithm or MTP
computation is changed. Runtimes without explicit ownership are rejected by
shared-format MTP validation instead of guessing draft layers.

The runner retains responsibility for checkpoint source/destination selection
and state copying. The connector uses the MRV2 post-load/deferred state-copy
extension points. Scheduler save ranges exclude unaccepted or uncomputed
tokens; completion events release pinned target checkpoints without pinning
speculative scratch blocks.

## Validation evidence

Hardware validation used two eight-device Ascend950PR endpoints. The paired
runtime was vLLM `5df63a7203cdb9328d824a27b682e77c7b9928be` and Ascend
`23e0e3764293f0c85ebf96c1383fbbf431b8fbb3`, with the connector delta applied.
The development baseline was Ascend `3141813e89fd162514bc12ed10966041771bb412`.
The runtime also needed upstream StoreLayout/SSM helper and draft-spec metadata
compatibility backports. This is not evidence that an unmodified release wheel
has all required interfaces; dependency integration must be resolved before merge.

| Stage | Evidence |
| --- | --- |
| 1A native baseline | Native model output and actual per-rank cache specs, shape/stride/dtype and target/draft ownership captured before pool acceptance |
| 1B object recovery | Native MemCache MLA/indexer/tail byte tests; eight KDA/ring cases covering MTP=0/1 and all bulk/layerwise combinations, fresh writer/reader processes, exact content hashes and untouched padding/other blocks |
| 1C ordinary sharing | Four bulk/layerwise combinations in both directions for MTP=0 and MTP=1; external 640-token hit and all eight rank GET records, output compared with native baseline |
| 1D PD | Original CPD proxy, 639/640/641 boundaries, chunked/short/repeated prompts, D writeback followed by P hit, request-level missing-object failure and subsequent healthy requests |
| 1E MTP/lifecycle | Source continuation/checkpoint equality, accepted/rejected draft counts, cancellation, 8 concurrent 128-token outputs with 57 observed preemptions, drained queues and local session/reference counters |

Focused remote tests passed: 229 connector/layout/scheduler tests and six
draft-ownership tests. Final deployment checks compared 10 sequential HTTP
responses and four post-restoration responses with native references: all
output tokens and their logprobs matched. Byte tests do not substitute for
these model and HTTP tests. Local reference counters do not prove native
backend internal lease accounting.

## Limitations retained in this draft

- No heterogeneous TP/CP/PP model acceptance, graph-mode, quantized-model,
  multimodal-input, dataset-accuracy or throughput claim.
- Concurrent output token IDs matched the sequential/native controls, but six
  logprob arrays differed, with maximum absolute difference about 0.01264.
  Native and cold-recompute controls did not establish the cause. This is an
  open observation, not a proved rounding explanation or bitwise precision claim.
- Ordinary synchronous eviction between lookup and lease acquisition can fail
  the execution step; request-level isolation remains follow-up work.
- Hybrid load-failure recompute remains prohibited by the existing Ascend
  platform guard. PD missing-object tests cover request failure and healthy
  subsequent requests, not automatic recovery of the failed request.
- Performance, prolonged concurrency, concurrent writers and eviction/lease
  fault injection require further production validation.

See [shared layout and lifecycle](ascend_store_shared_layout.md) for the
generic object format, interfaces and reuse boundary.
