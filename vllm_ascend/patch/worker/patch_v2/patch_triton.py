from vllm.v1.sample.ops import topk_topp_sampler
from vllm.v1.worker.gpu import input_batch, model_runner, structured_outputs
from vllm.v1.worker.gpu.sample import bad_words, gumbel, logprob, penalties, prompt_logprob, sampler, states
from vllm.v1.worker.gpu.spec_decode import rejection_sampler, rejection_sampler_utils
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.eagle import speculator

from vllm_ascend.sample.sampler import apply_top_k_top_p as npu_apply_top_k_top_p
from vllm_ascend.worker.v2.input_batch import post_update
from vllm_ascend.worker.v2.sample.bad_words import apply_bad_words
from vllm_ascend.worker.v2.sample.gumbel import apply_temperature, gumbel_sample
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs, compute_topk_logprobs
from vllm_ascend.worker.v2.sample.min_p import apply_min_p
from vllm_ascend.worker.v2.sample.penalties import apply_penalties, bincount
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import _prepare_dflash_inputs_kernel_ascend
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import (
    rejection_sample as npu_rejection_sample,
)
from vllm_ascend.worker.v2.structured_outputs import _apply_grammar_bitmask_kernel

penalties.apply_penalties = apply_penalties
# because sampler.py and speculator.py are imported before this patch, they must be overridden
sampler.gumbel_sample = gumbel_sample
input_batch.post_update = post_update
prompt_logprob.compute_topk_logprobs = compute_topk_logprobs
sampler.compute_topk_logprobs = compute_topk_logprobs
rejection_sampler.compute_topk_logprobs = compute_topk_logprobs
states.apply_min_p = apply_min_p
penalties.bincount = bincount
speculator.gumbel_sample = gumbel_sample
model_runner.post_update = post_update
bad_words.apply_bad_words = apply_bad_words
gumbel.apply_temperature = apply_temperature
states.apply_temperature = apply_temperature
logprob.compute_token_logprobs = compute_token_logprobs
structured_outputs._apply_grammar_bitmask_kernel = _apply_grammar_bitmask_kernel
rejection_sampler_utils.rejection_sample = npu_rejection_sample
rejection_sampler.rejection_sample = npu_rejection_sample
dflash_speculator._prepare_dflash_inputs_kernel = _prepare_dflash_inputs_kernel_ascend

# Upstream routes to apply_top_k_top_p_triton when HAS_TRITON and batch >= 8,
# which is broken on Ascend.
# TODO: remove the following patches when apply_top_k_top_p_triton is fixed on Ascend.
topk_topp_sampler.apply_top_k_top_p = npu_apply_top_k_top_p
states.apply_top_k_top_p = npu_apply_top_k_top_p
sampler.apply_top_k_top_p = npu_apply_top_k_top_p
