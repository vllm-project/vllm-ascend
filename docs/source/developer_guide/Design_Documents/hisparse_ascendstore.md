# RFC: Reuse HiSparse prefixes with AscendStore in colocated serving

## Motivation

An agent often sends the same long system prompt and conversation history on successive requests. If the prefix has left the worker's local cache, the next request has to run Prefill over that history again. HiSparse helps the *running* request fit more KV by keeping historical main KV in local DDR and fetching selected rows for Decode. AscendStore provides a way to reuse KV *across* requests. [Ascend HiSparse RFC](https://github.com/vllm-project/vllm-ascend/issues/19293)

We propose combining these features on a single Ascend A5 node that runs both Prefill and Decode. A GLM-5.3 request can restore a previously saved prefix, compute its new suffix, and continue decoding with the existing HiSparse attention path. This builds on the [HiSparse foundation](https://github.com/vllm-project/vllm-ascend/pull/20926).

## Proposed change

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

The integration registers Host Source and Indexer with their own block counts. Indexer, Resident and Hot retain their shared NPU allocation. For a 128-token C8 Indexer page, the store transfers 16,384 bytes of keys and 512 bytes of scales. The device's page stride locates the next page, so alignment padding can stay on the device. Pool keys include the model identities and KV layout to keep incompatible prefixes separate.

### How the two connectors work together

1. **Save a prefix.** HiSparse mirrors new main KV into DDR. AscendStore waits for those writes, including pending MTP draft rows, before its sender reads the block. The worker keeps the source slots valid until the queued PUTs finish. Eager Host mirroring is required even for Prefill-only saves because a one-token Prefill chunk can take the Decode path.
2. **Restore a prefix.** AscendStore loads Host Source and Indexer asynchronously into their respective destinations. The scheduler makes the imported prefix available after every tensor-parallel rank succeeds. For a partial hit, the Resident reservation covers the active Prefill window, allowing the worker to process a long remaining suffix in chunks.
3. **Resume after a failed GET.** The destinations stay reserved until all ranks finish. The default `kv_load_failure_policy=fail` fails the affected request. With `recompute`, the worker computes the missing prefix locally and the scheduler skips further external lookups for that request.

### Initial scope

Use Model Runner V2 and a `MultiConnector` containing one `HiSparseConnector` and one `AscendStoreConnector`, all with `kv_role=kv_both`. The first version covers single-node, whole-block transfers with MemCache `host_tcp`, `load_async=true`, `eager_host_mirror=true` and `prefill_context_parallel_size=1`. P/D disaggregation and layerwise transfer are separate use cases. Mooncake is disabled for this combination pending a fix and A5 validation for writes that outlive a timeout. [Backend issue](https://github.com/kvcache-ai/Mooncake/issues/4491)

## Validation

The implementation was exercised with GLM-5.3 on one A5 TP8 node. Two comparisons illustrate the scope.

| Case | What was exercised | Observed result |
| --- | --- | --- |
| BF16, 128K | Cold request, 65,536-token partial hit and 131,072-token full hit | All completed with matching output tokens and selected-token logprobs. |
| C8, MTP=3, 16,384-token input | HiSparse alone with a local hit, the combination with a local hit, and the combination with a forced MemCache hit | Each reused 16,256 tokens. The external replay had zero local hits, issued 2,032 GET keys and computed the remaining 128 tokens. All 32 output token IDs, selected logprobs and top-5 logprob maps matched across the three runs. |

The forced replay kept the pool warm and cleared only the local prefix cache. This checks that the restored KV actually served the request. The implementation PR records the tested revisions, byte-level replay checks and four-request performance measurements. [Implementation and results](https://github.com/vllm-project/vllm-ascend/pull/21924) · [Matched-hit accuracy comparison](https://github.com/vllm-project/vllm-ascend/pull/21924#issuecomment-7064234306)

## Feedback period

One week from the start of formal review. Feedback is welcome on the page groups and connector responsibilities proposed for whole-block prefix reuse.
