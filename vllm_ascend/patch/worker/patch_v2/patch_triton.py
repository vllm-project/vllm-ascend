from vllm.v1.worker.gpu import input_batch, sample

from vllm_ascend.worker.v2.input_batch import post_update
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs
from vllm_ascend.worker.v2.sample.penalties import apply_penalties

sample.logprob = compute_token_logprobs
sample.penalties.apply_penalties = apply_penalties
sample.gumbel.gumbel_sample = gumbel_sample
input_batch.post_update = post_update
