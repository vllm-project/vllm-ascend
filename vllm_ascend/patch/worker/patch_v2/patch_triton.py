from vllm.v1.worker.gpu import input_batch
from vllm.v1.worker.gpu.sample import gumbel, logprob, penalties, sampler, min_p

from vllm_ascend.worker.v2.input_batch import post_update
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.sample.logprob import compute_topk_logprobs
from vllm_ascend.worker.v2.sample.penalties import apply_penalties
from vllm_ascend.worker.v2.sample.min_p import apply_min_p

logprob.compute_topk_logprobs = compute_topk_logprobs
penalties.apply_penalties = apply_penalties
gumbel.gumbel_sample = gumbel_sample
# because sampler.py is imported before this patch, it must be overridden
sampler.gumbel_sample = gumbel_sample
input_batch.post_update = post_update
min_p.apply_min_p = apply_min_p
