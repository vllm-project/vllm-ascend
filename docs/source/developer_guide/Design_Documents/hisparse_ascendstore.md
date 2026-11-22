# HiSparse prefix reuse with AscendStore in colocated serving

## Motivation

An agent often sends the same long system prompt and conversation history on successive requests. If the prefix has left the worker's local cache, the next request has to run Prefill over that history again. HiSparse helps the *running* request fit more KV by keeping historical main KV in local DDR and fetching selected rows for Decode. AscendStore provides a way to reuse KV *across* requests. [Ascend HiSparse RFC](https://github.com/vllm-project/vllm-ascend/issues/19293)

This design combines these features on a single Ascend A5 node that runs both Prefill and Decode. A GLM-5.3 request can restore a previously saved prefix, compute its new suffix, and continue decoding with the existing HiSparse attention path. [HiSparse foundation](https://github.com/vllm-project/vllm-ascend/pull/20926) · [Integration PR](https://github.com/vllm-project/vllm-ascend/pull/21924)

## Prefix lifecycle

### A prefix shared by two requests

```text
Request 1  [ shared prefix ] [ first question ]
             Prefill produces KV
             AscendStore saves complete prefix blocks to the external pool (PUT)

Request 2  [ shared prefix ] [ next question ]
             Local prefix miss, external pool hit
             AscendStore restores matching blocks to DDR and NPU (GET)
                             │
                             ▼
             Prefill computes the remaining tokens
             HiSparse serves Decode with Resident, Hot and Host Source KV
```

AscendStore moves **whole prefix blocks** between the worker and the pool. Inside each Decode step, HiSparse selects **individual KV rows** and fetches missing rows from local DDR. A pool hit restores the state needed for that selection.

### What goes into a pool entry?

| HiSparse group | Contents and location | AscendStore action |
| --- | --- | --- |
| Host Source | Main KV history in local DDR | Save and restore the main KV for each complete prefix block. |
| Indexer | Index keys on the NPU used to select historical tokens | Save and restore the matching keys, including their scales for C8. |
| Resident | Main KV pages currently on the NPU | Keep local. HiSparse manages their residency. |
| Hot | NPU slots holding selected rows fetched from DDR | Keep local. HiSparse fills them as Decode needs them. |

The integration registers Host Source and Indexer with their own block counts. Indexer, Resident and Hot retain their shared NPU allocation. For a 128-token C8 Indexer page, the store transfers 16,384 bytes of keys and 512 bytes of scales. The device's page stride locates the next page, so alignment padding can stay on the device.

### How the two connectors work together

1. **Save a prefix.** HiSparse mirrors new main KV into DDR. AscendStore waits for those writes, including pending MTP draft rows, before its sender reads the block. The worker keeps the source slots valid until the queued PUTs finish. Eager Host mirroring is required even for Prefill-only saves because a one-token Prefill chunk can take the Decode path.
2. **Restore a prefix.** AscendStore loads Host Source and Indexer asynchronously into their respective destinations. The scheduler makes the imported prefix available after every tensor-parallel rank succeeds.
3. **Resume after a failed GET.** The destinations stay reserved until all ranks finish. The default `kv_load_failure_policy=fail` fails the affected request. With `recompute`, the worker computes the missing prefix locally and the scheduler skips further external lookups for that request.

### Resident admission on a 128K partial hit

Consider a 131,072-token prompt with a 65,536-token external hit and 128 tokens per block. The hit covers 512 blocks. The remaining 512 blocks will be computed over successive Prefill chunks. Requiring all 512 suffix blocks to fit in Resident at admission can stall a request that the worker has enough memory to process one chunk at a time.

For external-import admission, the manager starts from the larger of the computed prefix page count and the number of pages already assigned to the request. It looks ahead by at most the Resident spec's `max_admission_blocks_per_request`. Core accounts separately for a writable import-tail page. Startup sizing uses the same window and leaves room for that one extra page, bounded by the maximum context. The cap applies to the admission estimate. Each running step still allocates the pages its chunk needs. [Resident spec and manager](https://github.com/Esir98/vllm-ascend/blob/8e91a8436508df40b0b57e9aab959b5089739a38/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/hisparse_admission.py)

### External pool key isolation

The combined path uses `hisparse-ascendstore-v1:<digest>` as its pool key namespace. The version identifies the external key format. The digest includes target and optional draft checkpoint identities and revisions, dummy-weight mode, effective model configuration (including RoPE), and transferable page geometry (including dtype, block size, stride and scale layout). Two workers can share a prefix only when the namespace and the prefix key match. Ordinary AscendStore retains its own key format.

For a persistent pool, pin the Hub commit if the resolved revision is unavailable. For local weights, use a new versioned directory when replacing a checkpoint in place. The key records the path and configuration and does not hash the weight files. [Key construction](https://github.com/Esir98/vllm-ascend/blob/8e91a8436508df40b0b57e9aab959b5089739a38/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/hisparse.py)

## Configuration

Use Model Runner V2 on one node with colocated Prefill and Decode, whole-block transfers and `prefill_context_parallel_size=1`. Set `hisparse_config.eager_host_mirror=true` in the attention configuration and `ock.mmc.local_service.protocol = host_tcp` in the MemCache configuration. The following JSON is the value of `--kv-transfer-config`. The 4 GiB Host Source budget is an example **per TP rank**. Size it for the target workload. [HiSparse settings](../../user_guide/feature_guide/hisparse.md) · [MemCache setup](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md)

```json
{
  "kv_connector": "MultiConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "HiSparseConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"host_pool_gib": 4}
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"backend": "memcache", "load_async": true}
      }
    ]
  }
}
```

## Design trade-offs

| Choice | Benefit | Cost or alternative |
| --- | --- | --- |
| Compose the connectors with `MultiConnector` | HiSparse owns residency and refill. AscendStore owns prefix lookup and transfer. Each remains usable independently. | The parent must order draft mirroring before the store's save hook. Building pooling into HiSparse would simplify that handoff but couple its worker lifecycle to an external store. |
| Reserve headroom and take an aligned view | Core allocates the shared NPU slab once with up to 2 MiB minus one byte of extra space. The runner binds Indexer, Resident and Hot to a 2 MiB-aligned slice of the same storage, with no KV copy. | Ascend must coordinate reservation before allocation and view binding afterwards. Native alignment support in the upstream allocator could replace both steps. |
| Require eager Host mirroring and fence each save | The sender waits for the Host-write event before the first PUT. Even a one-token Prefill chunk or MTP draft write is visible in DDR when the pool reads it. | Eager mirroring adds copy work. Draining queued PUTs can stall the worker, but removing the drain would let Core recycle slots while the sender still reads them. |

## Validation

The implementation was exercised with GLM-5.3 on one A5 TP8 node. Two comparisons illustrate the scope.

| Case | What was exercised | Observed result |
| --- | --- | --- |
| BF16, 128K | Cold request, 65,536-token partial hit and 131,072-token full hit | All completed with matching output tokens and selected-token logprobs. |
| C8, MTP=3, 16,384-token input | HiSparse alone with a local hit, the combination with a local hit, and the combination with a forced MemCache hit | Each reused 16,256 tokens. The external replay had zero local hits, issued 2,032 GET keys and computed the remaining 128 tokens. All 32 output token IDs, selected logprobs and top-5 logprob maps matched across the three runs. |

The forced replay kept the pool warm and cleared only the local prefix cache. This checks that the restored KV actually served the request. The implementation PR records the tested revisions, byte-level replay checks and four-request performance measurements. [Implementation and results](https://github.com/vllm-project/vllm-ascend/pull/21924) · [Matched-hit accuracy comparison](https://github.com/vllm-project/vllm-ascend/pull/21924#issuecomment-7064234306)

## Future work

- A public Core hook for pending draft Host writes and per-slot send-completion tracking could narrow the queue-wide wait while preserving source lifetime. Native allocator alignment could remove the Ascend-specific slab reservation.
- Enabling Mooncake requires an SDK guarantee that a timed-out PUT has stopped writing before its destination can be reused, followed by an A5 test of unrelated keys after timeout. The current combination accepts MemCache `host_tcp` only. [Mooncake issue](https://github.com/kvcache-ai/Mooncake/issues/4491)
- P/D disaggregation and layerwise transfer need separate page ownership and completion rules for their respective handoffs.
