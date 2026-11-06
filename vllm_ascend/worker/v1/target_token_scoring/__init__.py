"""Target Token Scoring: compact scorer for fixed-candidate-set scoring.

When every request in a wave only needs scores for a small, fixed set of
candidate token ids (rerankers / relevance scoring), vLLM's default path still
runs the full-vocab LM Head (``[B, H] @ [V, H]^T -> [B, V]``) and then gathers
the K candidate columns in the sampler. ``logprob_token_ids`` is a
*sampler/output* semantic, not a sparse-LM-Head contract: seeing K results in
the response does not imply the device computed only K columns.

This package adds an opt-in fast path that, on an eligible wave, replaces the
full-vocab projection with ``index_select`` of the K candidate weight rows
(``[K, H]``) -> a compact ``[B, K]`` linear, then argmax->remap->target-set
log-softmax, producing a ``SamplerOutput`` directly. Ineligible waves fall back
to the native path unchanged.

This is the Ascend-local counterpart of vllm-project/vllm#56782. It exists
because ``NPUModelRunner`` overrides ``execute_model`` / ``sample_tokens`` /
``_sample``, so the upstream hooks in ``GPUModelRunner`` do not fire on NPU.
When #56782 lands upstream, ``NPUModelRunner`` should call the upstream API
directly and this vendored package should be removed.

Only the dense, single-rank, unquantized LM Head case is supported. Everything
else (TP vocab sharding, quantized/packed LM Head, custom output layers,
structured output, speculative decoding) falls back to native rather than risk
silently wrong semantics.
"""

from .admission import (
    AdmissionDecision,
    TargetTokenScoringState,
    evaluate_wave_admission,
)
from .projector import project_target_token_logits
from .compact_sampler import compact_sample

__all__ = [
    "TargetTokenScoringState",
    "AdmissionDecision",
    "evaluate_wave_admission",
    "project_target_token_logits",
    "compact_sample",
]
