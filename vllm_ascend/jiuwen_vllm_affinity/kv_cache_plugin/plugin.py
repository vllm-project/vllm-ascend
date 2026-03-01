import logging

from vllm_ascend.jiuwen_vllm_affinity.kv_cache_plugin.patcher import apply_patches


def register():
    logging.info("register jiuwen_vllm_affinity kv_cache_plugin")
    apply_patches()
