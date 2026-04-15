from vllm.v1.worker.gpu import input_batch, model_runner
from vllm.v1.worker.gpu.sample import bad_words, penalties, prompt_logprob, sampler, state
from vllm.v1.worker.gpu.spec_decode import rejection_sampler
from vllm.v1.worker.gpu.spec_decode.eagle import speculator

from vllm_ascend.worker.v2.input_batch import post_update
from vllm_ascend.worker.v2.sample.bad_words import apply_bad_words
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.sample.logprob import compute_topk_logprobs
from vllm_ascend.worker.v2.sample.min_p import apply_min_p
from vllm_ascend.worker.v2.sample.penalties import apply_penalties, bincount

penalties.apply_penalties = apply_penalties
# because sampler.py and speculator.py are imported before this patch, they must be overridden
sampler.gumbel_sample = gumbel_sample
input_batch.post_update = post_update
prompt_logprob.compute_topk_logprobs = compute_topk_logprobs
sampler.compute_topk_logprobs = compute_topk_logprobs
rejection_sampler.compute_topk_logprobs = compute_topk_logprobs
state.apply_min_p = apply_min_p
penalties.bincount = bincount
speculator.gumbel_sample = gumbel_sample
model_runner.post_update = post_update
bad_words.apply_bad_words = apply_bad_words
