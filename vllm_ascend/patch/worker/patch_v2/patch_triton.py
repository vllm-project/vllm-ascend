from vllm.v1.worker.gpu import input_batch
from vllm.v1.worker.gpu.sample import gumbel, logprob, penalties, sampler
from vllm.v1.worker.gpu.sample import states

from vllm_ascend.worker.v2.input_batch import post_update
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.sample.logprob import compute_token_logprobs
from vllm_ascend.worker.v2.sample.penalties import apply_penalties
from vllm_ascend.worker.v2.sample.gumbel import apply_temperature

logprob.compute_token_logprobs = compute_token_logprobs
penalties.apply_penalties = apply_penalties
gumbel.gumbel_sample = gumbel_sample
# because sampler.py is imported before this patch, it must be overridden
sampler.gumbel_sample = gumbel_sample
input_batch.post_update = post_update
gumbel.apply_temperature = apply_temperature
states.apply_temperature = apply_temperature
