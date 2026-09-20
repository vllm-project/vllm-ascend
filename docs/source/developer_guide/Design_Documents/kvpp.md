# KVPP Design

## 1. Background

MLA/SFA models can replicate historical KV caches across tensor parallel (TP) ranks, limiting capacity for long contexts and concurrent requests. KVPP distributes persistent caches by layer within each TP group and broadcasts a layer's cache when computation needs it.

Model execution retains its TP/EP/PP configuration. The scheduler keeps the complete logical cache specification and block numbering.

## 2. Design

KVPP combines layer ownership, two reusable scratch buffers, and prefetching one layer ahead:

```text
TP group within one PP stage
    Owner rank: persistent main KV + indexer KV for its layers
         |
         +-- Full-layer broadcast --> Other ranks: scratch buffer
                                          |
                                          +-- Wait --> Compute current layer
                                                       Prefetch next layer
```

### 2.1 Layer Ownership

Each pipeline parallel (PP) stage sorts its local target layers by layer index and assigns contiguous, balanced partitions to owner ranks in its TP group. A layer's main KV, indexer KV, and quantization components form one bundle with a single owner. MTP caches remain independently allocated on each rank that requires them and are excluded from target-layer partitioning.

Each rank allocates persistent storage for its owned layers and reuses two scratch buffers for other layers. Target-layer execution ordinals select alternating scratch buffers, allowing the current layer and the next prefetch to use separate storage.

### 2.2 Physical Layout and Memory Budget

Components within a bundle are contiguous, and each component contains all logical blocks for that layer. Layers do not need one globally contiguous allocation; only each broadcast bundle must be contiguous. The layout follows the actual cache specifications, including main KV, indexer data, and scales.

The physical budget on each rank is:

```text
Physical bytes per block
  = Sum of owned target-bundle bytes per block
  + Local MTP-cache bytes per block
  + 2 * Largest target-bundle bytes per block

Available blocks = floor(Available KV memory / Physical bytes per block)
```

The worker converts the resulting block capacity into a logical memory budget for the planner. Physical allocation uses the final block count, preserving logical cache management while allocating storage according to the KVPP layout.

### 2.3 Full-Layer Broadcast and Prefetch

A forward pass with previously computed tokens in any actual request starts by prefetching the first layer. Each attention layer waits for its broadcast after projections and before its first KV-cache access or write, then starts prefetching the next layer.

Each broadcast stays within the current PP stage's KVPP group. The owner broadcasts the complete bundle once, including all components and all allocated logical blocks. Other ranks receive it into the corresponding scratch buffer. After computation, the owner retains the updated cache for subsequent forward passes.

Prefetching uses a separate transfer stream. A ready event establishes data dependencies before broadcast, and a completion event ensures the device transfer has finished before the wait returns. The next layer's broadcast can overlap with current-layer computation; the overlap depends on communication and compute costs. Forward passes without history, dummy runs, and profiling paths do not broadcast historical caches.

## 3. Support and Constraints

| Area | Scope |
| --- | --- |
| Models and execution | Non-hybrid MLA/SFA models in eager mode; Model Runner V1 and V2 |
| Supported combinations | TP, EP, PP, chunked prefill, prefix caching, asynchronous scheduling, fixed-step MTP |
| Cache layouts | Allocated from actual specifications, including LI-C8 and SFA-C8 |
| Not supported | Graph execution, PCP, DCP, disaggregated prefill/decode, variable-step MTP |

KVPP trades communication for persistent cache capacity. Each rank still needs two scratch buffers sized for the largest target layer, plus its independent MTP caches. Memory savings therefore depend on layer count, TP size, and per-layer cache sizes. Broadcast traffic grows with the allocated block count; payloads are not filtered by request or active block.

## 4. Usage

Enable KVPP through the model launch arguments. The KVPP group size follows the TP size:

```bash
vllm serve <model-path> \
  --tensor-parallel-size 2 \
  --enforce-eager \
  --additional-config '{"enable_kvpp": true}'
```

`enable_kvpp` defaults to `false`. With PP enabled, each stage assigns owners and broadcasts within its own TP group.

## P-side layerwise offload

The Memcache shared-buffer opt-in uses the same full-layer HCCL broadcast transport. It adds a load-completion dependency in the prefetch executor and gives the owner three reusable buffers, while keeping the two peer receive buffers. `KVPPPhysicalCachePlan.buffer_slots()` drives both physical allocation and the connector's reuse predecessors so their lifetimes cannot diverge.

For layer L, attention queues H2D through L+2, and the KVPP hook queues the broadcast of L+1. H2D for the first two layers starts without an attention gate. The model runner starts KVPP only after `start_load_kv` has reset the previous forward's completion events. The compute and broadcast paths share those events until the next forward; neither clears a completion the other still needs.

All ranks enqueue layers in global order, including empty non-owner transfer tasks. Empty tasks still fence previous compute and D2H users of a peer buffer. Only owner tasks copy host KV into HBM. Full-object Memcache publication retains its existing single writer, and read leases are released at the end of the forward rather than at the last owned H2D layer.

The physical budget accounts for three owner and two peer buffers directly; the generic layerwise logical-memory multiplier is skipped. The generic layerwise tensor-merging pass is also skipped because the KVPP allocator consumes the complete logical cache specification.

For replicated Memcache offload caches, a TP CPU barrier at the next forward boundary prevents a non-writer rank from acquiring a read lease before the preceding writer has finished publishing the full object. Layerwise overlap within a forward is unchanged.
