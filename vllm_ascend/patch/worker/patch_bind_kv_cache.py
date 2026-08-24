import vllm.v1.worker.utils as utils

from vllm_ascend.worker.utils import bind_kv_cache

utils.bind_kv_cache = bind_kv_cache
