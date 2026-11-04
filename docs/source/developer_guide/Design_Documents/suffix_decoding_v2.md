# Suffix decoding adaptation for Model Runner V2

Draft placeholder for [RFC #20832](https://github.com/vllm-project/vllm-ascend/issues/20832).
This change records the adaptation scope; runtime implementation is pending.

## Integration scope

1. Adapt the device-resident suffix speculator from
   [vLLM #60339](https://github.com/vllm-project/vllm/pull/60339), including
   own-history matching and an optional shared corpus of completed responses.
2. Wire Ascend MRV2 dispatch, dense token history and the upstream request hooks:
   `add_request`, `on_requests_finished` and `reset_state`. Ingest responses before
   releasing slots and clear corpus/statistics after warmup.
3. Port the kernels to NPU and preserve adaptive-verification statistics,
   draft-slot accounting and deterministic TP corpus-ingestion order. Exercise
   short histories, overwritten corpus entries, ring wraparound and slot reuse.
4. Qualify async execution, acceptance and performance on the verified upstream
   revision. Preserve documented corpus-sharing and seeded-reproducibility
   semantics, and distinguish MRV2 options from V1 Arctic options.

## Existing implementation reference

[Ascend Draft PR #19394](https://github.com/vllm-project/vllm-ascend/pull/19394)
contains an earlier NPU port. Reconcile it with the merged upstream protocol,
including request/seed handling, and resolve its reported adaptive observe-kernel
compilation gap before claiming end-to-end support.
