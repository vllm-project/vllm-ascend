# Speculative Decoding Upstream Alignment

## Scope

Issue [#10661](https://github.com/vllm-project/vllm-ascend/issues/10661)
asks vLLM Ascend to reduce divergence from upstream speculative decoding.
This change centralizes family selection for the current MRV2 factory and
the 310P MTP router. It is an incremental alignment, not a claim that all
speculative decoding implementations are now identical.

The refresh targets the current mainline Model Runner V2. Model Runner V1 has
been removed from that baseline. Its proposers, slot-mapping helper and
readiness fallback are not restored by this change. `VLLM_USE_V2_MODEL_RUNNER=0`
is not a supported rollback; the mainline runner validation remains unchanged.

## Ownership

| Boundary | Owner | Ascend responsibility |
| --- | --- | --- |
| Family selection | `vllm_ascend.worker.spec_decode_family` | Classify draft architecture and method consistently for MRV2 and 310P. |
| Generic sampling and rejection | Upstream vLLM | Reuse upstream orchestration rather than duplicate it in the selector. |
| N-gram and hidden-state extraction | Upstream MRV2 speculators | Preserve current request-state forwarding and mainline integration. |
| Model-specific speculators | Existing Ascend MRV2 adapters | Retain NPU attention metadata, operators, LM-head TP alignment and graph handling. |
| 310P MTP | Existing Ascend 310P adapter | Keep CPU input preparation and route through the common family decision. |
| KV layout and graph buffers | Existing MRV2 implementations | Preserve PCP/DCP layout, padding and graph-safe buffers; no removed MRV1 helper is introduced. |

The selector preserves LiLiCorr, DFlash2, multi-module MTP, Gemma4 and Step3.5
routing as well as ordinary EAGLE, MTP and draft-model routing. Multi-KV-group
MTP continues using the existing generic Ascend MTP speculator.

## Patch Lifecycle

The existing `vllm_ascend/patch/worker/patch_spec_decode_310.py` adapter remains
registered through the mainline worker patch mechanism. This change only
replaces its duplicated MTP decision with the common selector and preserves
the request-state argument passed to the parent factory. The adapter can be
removed when upstream provides the corresponding 310P implementation or an
equivalent platform factory extension. No new patch registration is added.

## Regression Gates

- Family tests cover method selection, special-family precedence, unknown
  methods, and architecture attributes on draft and nested model configs.
- MRV2 dispatch tests retain mainline n-gram request-state coverage and cover
  model-specific factories, including LiLiCorr and multi-module MTP.
- 310P tests cover both MTP routing and delegation to the parent factory.
- Reuse mainline model-backed E2E under `tests/e2e/pull_request/one_card/`
  and `two_card/model_runner_v2/` for n-gram, hidden-state extraction, MTP,
  EAGLE3, DFlash and DSpark. Do not introduce a second set of stale golden
  outputs or a parallel nightly configuration for the same families.

Historical PR tests and hardware runs do not validate this refreshed commit.
Focused CPU tests, current-pin NPU execution, family-specific model parity,
graph capture/replay and performance comparisons remain separate evidence.
Missing draft checkpoints or unavailable hardware must be reported explicitly.
