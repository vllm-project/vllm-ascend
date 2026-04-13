from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.sample import penalties, sampler

from vllm_ascend.worker.v2.input_batch import post_update
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.sample.penalties import apply_penalties

penalties.apply_penalties = apply_penalties
# because sampler.py is imported before this patch, it must be overridden
sampler.gumbel_sample = gumbel_sample
model_runner.post_update = post_update
