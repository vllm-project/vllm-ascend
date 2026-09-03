import vllm.v1.worker.gpu.spec_decode.speculator as base_speculator
from vllm.v1.sample.ops import topk_topp_sampler
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu import structured_outputs
from vllm.v1.worker.gpu.sample import bad_words, gumbel, logprob, penalties, prompt_logprob, sampler, states
from vllm.v1.worker.gpu.spec_decode import rejection_sampler, rejection_sampler_utils
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.eagle import speculator

from vllm_ascend.ops.triton.v2.apply_grammar_bitmask import _apply_grammar_bitmask_kernel
from vllm_ascend.ops.triton.v2.mamba.precopy import precopy_mamba_align_fused_kernel
from vllm_ascend.ops.triton.v2.metrics.num_nans import get_num_nans
from vllm_ascend.ops.triton.v2.sample.fill_logprob_token_idx import _fill_logprob_token_ids_kernel
from vllm_ascend.worker.v2.sample.apply_top_k_top_p import apply_top_k_top_p_npu
from vllm_ascend.worker.v2.sample.bad_words import apply_bad_words
from vllm_ascend.worker.v2.sample.gumbel import apply_temperature, gumbel_sample
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs, compute_topk_logprobs
from vllm_ascend.worker.v2.sample.min_p import apply_min_p
from vllm_ascend.worker.v2.sample.penalties import apply_penalties, bincount
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import _prepare_dflash_inputs_kernel_ascend
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
    rejection_sample as npu_rejection_sample,
)

# triton ops that need to be filed in ops/triton
penalties.apply_penalties = apply_penalties
# because sampler.py and speculator.py are imported before this patch, they must be overridden
sampler.gumbel_sample = gumbel_sample
prompt_logprob.compute_topk_logprobs = compute_topk_logprobs
sampler.compute_topk_logprobs = compute_topk_logprobs
rejection_sampler.compute_topk_logprobs = compute_topk_logprobs
states.apply_min_p = apply_min_p
penalties.bincount = bincount
speculator.gumbel_sample = gumbel_sample
base_speculator.gumbel_sample = gumbel_sample
bad_words.apply_bad_words = apply_bad_words
gumbel.gumbel_sample = gumbel_sample
gumbel.apply_temperature = apply_temperature
states.apply_temperature = apply_temperature
logprob.compute_token_logprobs = compute_token_logprobs
rejection_sampler_utils.rejection_sample = npu_rejection_sample
rejection_sampler.rejection_sample = npu_rejection_sample
dflash_speculator._prepare_dflash_inputs_kernel = _prepare_dflash_inputs_kernel_ascend
# triton ops that filed in ops/triton
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
