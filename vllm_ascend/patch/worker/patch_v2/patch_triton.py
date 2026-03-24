import vllm

from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs
from vllm_ascend.worker.v2.sample.penalties import apply_penalties

vllm.v1.worker.gpu.sample.logprob = compute_token_logprobs
vllm.v1.worker.gpu.sample.penalties.apply_penalties = apply_penalties
vllm.v1.worker.gpu.sample.gumbel.gumbel_sample = gumbel_sample
