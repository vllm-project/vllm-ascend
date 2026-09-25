import vllm.v1.worker.gpu.model_runner
import vllm.v1.worker.utils

from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace

# Ascend layer caches can be tuples/lists of K/V or Mamba state tensors, while
# the upstream helper assumes one tensor per entry. Swap in the Ascend
# implementation, which unpacks nested entries and deduplicates shared views.
# The v2 model runner imported the upstream helper by name, so its module
# namespace must be patched in addition to the source module.
vllm.v1.worker.utils.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace
vllm.v1.worker.gpu.model_runner.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace
