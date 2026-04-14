from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.sample import penalties, sampler
from vllm.v1.worker.gpu.spec_decode.eagle import speculator

from vllm_ascend.worker.v2.input_batch import post_update
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample
from vllm_ascend.worker.v2.sample.penalties import apply_penalties

penalties.apply_penalties = apply_penalties
# because sampler.py and speculator.py are imported before this patch, they must be overridden
sampler.gumbel_sample = gumbel_sample
speculator.gumbel_sample = gumbel_sample
model_runner.post_update = post_update
