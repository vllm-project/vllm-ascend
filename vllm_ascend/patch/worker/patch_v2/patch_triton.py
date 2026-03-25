from vllm.v1.worker.gpu import input_batch
from vllm.v1.worker.gpu.sample import gumbel, logprob, penalties

from vllm_ascend.worker.v2.input_batch import post_update
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs
from vllm_ascend.worker.v2.sample.penalties import apply_penalties

logprob.compute_token_logprobs = compute_token_logprobs
penalties.apply_penalties = apply_penalties
gumbel.gumbel_sample = gumbel_sample
input_batch.post_update = post_update
