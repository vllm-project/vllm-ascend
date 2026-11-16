import vllm.model_executor.layers.mamba.ops.causal_conv1d as _cc1d
import vllm.third_party.flash_linear_attention.ops as fla_ops
import vllm.third_party.flash_linear_attention.ops.layernorm_guard as fla_layernorm_guard
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.layernorm_guard import LayerNormFn


def _next_power_of_2(n):
    # vLLM #58651's packed_qk_rope kernel evaluates
    # triton.next_power_of_2(rotary_dim) with a tl.constexpr argument.
    # The host helper only understands plain ints, so unwrap first.
    if isinstance(n, tl.constexpr):
        n = n.value
    return next_power_of_2(n)


# Newer Triton tracks JIT kernel dependencies and rejects plain Python
# callables. Keep the host helper callable from constexpr code as well.
if HAS_TRITON and hasattr(triton, "constexpr_function"):
    triton.next_power_of_2 = triton.constexpr_function(next_power_of_2)
else:
    triton.next_power_of_2 = _next_power_of_2


fla_layernorm_guard.LayerNormFn = LayerNormFn
fla_ops.chunk_gated_delta_rule = chunk_gated_delta_rule

# GLM-5.3-Flash (and Kimi KDA) import fused_recurrent_kda / chunk_kda_with_fused_gate
# from this FLA module. Swap them for the NPU Triton implementations before the
# model is constructed. Missing on older vLLM trees that predate KDA.
try:
    import vllm.third_party.flash_linear_attention.ops.kda as fla_kda

    from vllm_ascend.ops.triton.kda.kda import (
        chunk_kda as _npu_chunk_kda,
    )
    from vllm_ascend.ops.triton.kda.kda import (
        chunk_kda_with_fused_gate as _npu_chunk_kda_with_fused_gate,
    )
    from vllm_ascend.ops.triton.kda.kda import (
        fused_kda_gate as _npu_fused_kda_gate,
    )
    from vllm_ascend.ops.triton.kda.kda import (
        fused_recurrent_kda as _npu_fused_recurrent_kda,
    )

    fla_kda.fused_recurrent_kda = _npu_fused_recurrent_kda
    fla_kda.chunk_kda = _npu_chunk_kda
    fla_kda.chunk_kda_with_fused_gate = _npu_chunk_kda_with_fused_gate
    fla_kda.fused_kda_gate = _npu_fused_kda_gate
except ImportError:
    pass

# npu_kda_causal_conv1d_triton: the PyTorch fallback calls .item() per
# request, which aborts ACL graph capture. The upstream NPU Triton kernel has
# no host sync and accepts the spec-decode kwargs the fallback had to drop.
try:
    from vllm_ascend.ops.triton.mamba.causal_conv1d import (  # type: ignore[attr-defined]
        causal_conv1d_update_npu as _cc1d_update_npu,
    )

    _cc1d.causal_conv1d_update = _cc1d_update_npu
    logger.debug("Bound the NPU Triton causal_conv1d_update for the KDA layers.")
except Exception as _cc1d_err:
    logger.warning(
        "NPU Triton causal_conv1d_update is unavailable (%s); falling back to the"
        " PyTorch implementation, which syncs per request and therefore stalls ACL"
        " graph capture at decode-FULL.",
        _cc1d_err,
    )


# ---------------------------------------------------------------------------
# V2 Model Runner kernel rebinds. These used to live in a separate
# ``patch_triton_v2.py`` module; they were merged here so that every Ascend
# Triton kernel replacement lives in one file.
#
# The targets below live under ``vllm.v1.worker.gpu.*`` and are only exercised
# by the V2 Model Runner. They stay in this module so that every Ascend Triton
# kernel replacement lives in one place. The section must run *after* the shared
# model/kernel rebinds above: importing these v2 modules earlier would let a
# from-import capture an unpatched symbol before we rebind it.
# ---------------------------------------------------------------------------
import vllm.v1.worker.gpu.spec_decode.speculator as base_speculator  # noqa: E402
from vllm.v1.sample.ops import topk_topp_sampler  # noqa: E402
from vllm.v1.worker import mamba_utils  # noqa: E402
from vllm.v1.worker.gpu import structured_outputs  # noqa: E402
from vllm.v1.worker.gpu.sample import (  # noqa: E402
    bad_words,
    gumbel,
    logprob,
    output,
    penalties,
    prompt_logprob,
    sampler,
    states,
    thinking_budget,
    trace_replay,
)
from vllm.v1.worker.gpu.spec_decode import rejection_sampler, rejection_sampler_utils  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspark_speculator  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.eagle import speculator  # noqa: E402

from vllm_ascend.ops.triton.v2.apply_grammar_bitmask import _apply_grammar_bitmask_kernel  # noqa: E402
from vllm_ascend.ops.triton.v2.mamba.precopy import precopy_mamba_align_fused_kernel  # noqa: E402
from vllm_ascend.ops.triton.v2.metrics.num_nans import get_num_nans  # noqa: E402
from vllm_ascend.ops.triton.v2.sample.categorical_sample import categorical_sample  # noqa: E402
from vllm_ascend.ops.triton.v2.sample.fill_logprob_token_idx import _fill_logprob_token_ids_kernel  # noqa: E402
from vllm_ascend.ops.triton.v2.sample.sampling_mask import sampling_mask_from_logits_npu  # noqa: E402
from vllm_ascend.ops.triton.v2.sample.thinking_budget import (  # noqa: E402
    _load_effective_token_ascend,
    _update_committed_marker_cache_kernel_ascend,
)
from vllm_ascend.worker.v2.sample.apply_top_k_top_p import apply_top_k_top_p_npu  # noqa: E402
from vllm_ascend.worker.v2.sample.bad_words import apply_bad_words  # noqa: E402
from vllm_ascend.worker.v2.sample.gumbel import apply_temperature  # noqa: E402
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs, compute_topk_logprobs  # noqa: E402
from vllm_ascend.worker.v2.sample.min_p import apply_min_p  # noqa: E402
from vllm_ascend.worker.v2.sample.penalties import apply_penalties, bincount  # noqa: E402
from vllm_ascend.worker.v2.sample.trace_replay import apply_trace_tokens  # noqa: E402
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (  # noqa: E402
    rejection_sample as npu_rejection_sample,
)

# triton ops that need to be filed in ops/triton
penalties.apply_penalties = apply_penalties
# because sampler.py and speculator.py are imported before this patch, they must be overridden
prompt_logprob.compute_topk_logprobs = compute_topk_logprobs
sampler.compute_topk_logprobs = compute_topk_logprobs
rejection_sampler.compute_topk_logprobs = compute_topk_logprobs
states.apply_min_p = apply_min_p
penalties.bincount = bincount
bad_words.apply_bad_words = apply_bad_words
gumbel.apply_temperature = apply_temperature
states.apply_temperature = apply_temperature
logprob.compute_token_logprobs = compute_token_logprobs
rejection_sampler_utils.rejection_sample = npu_rejection_sample
rejection_sampler.rejection_sample = npu_rejection_sample
# triton ops that filed in ops/triton
gumbel.gumbel_sample = categorical_sample
speculator.gumbel_sample = categorical_sample
base_speculator.gumbel_sample = categorical_sample
dspark_speculator.gumbel_sample = categorical_sample
sampler.gumbel_sample = categorical_sample
topk_topp_sampler.apply_top_k_top_p_triton = apply_top_k_top_p_npu
structured_outputs._apply_grammar_bitmask_kernel = _apply_grammar_bitmask_kernel
mamba_utils.precopy_mamba_align_fused_kernel = precopy_mamba_align_fused_kernel
# This patch may be revisited or reverted once the compiler and Triton Ascend toolkit
# support the upstream implementation of fill_logprob_token_ids_kernel.
# For now, use the Ascend-specific implementation.
logprob._fill_logprob_token_ids_kernel = _fill_logprob_token_ids_kernel
# This patch may be revisited or reverted once the compiler and Triton Ascend toolkit
# support the upstream implementation of get_num_nans.
# For now, use the Ascend-specific implementation.
sampler.get_num_nans = get_num_nans
rejection_sampler.get_num_nans = get_num_nans
# Avoid excessive UB allocation for strided vocab-dimension loads.
output.SamplingMaskTensors.from_logits = classmethod(sampling_mask_from_logits_npu)
# TODO: Remove after the new Q4 Triton-Ascend release is available.
thinking_budget._load_effective_token = _load_effective_token_ascend
# TODO: Remove after Triton-Ascend 3.6.0 is the minimum supported version.
thinking_budget._update_committed_marker_cache_kernel = _update_committed_marker_cache_kernel_ascend
trace_replay.apply_trace_tokens = apply_trace_tokens
