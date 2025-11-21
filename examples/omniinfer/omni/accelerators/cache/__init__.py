from vllm.v1.core import kv_cache_manager as orig_manager
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.sched import scheduler
from vllm.v1.engine import core as engine_core
from vllm.v1.worker import block_table, gpu_input_batch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

ENABLED = False


from . import kv_cache_interface as itfc
from .kv_cache_interface import (OmniAttentionSpec, OmniMultiGroupBlockTable,
                                 get_kv_cache_config_omni_type,
                                 get_omni_hybrid_kv_cache_spec)
from .kv_cache_manager import OmniKVCacheBlocks, OmniKVCacheManager
from .nsa_kv_manager import NSAHostDeviceKVCacheManager, get_nsa_kv_cache_spec
from .pd import OmniBiGroupDataDistManager
from .utils import compute_omni_attn_metadata, to_bool_or_raise


def check_omni_attn_cmd_arg(additional_config: dict) -> bool:
    if additional_config is None or "enable_omni_attn" not in additional_config:
        return False
    return to_bool_or_raise(additional_config["enable_omni_attn"])


def apply_omni_attn_patch(enable=False, is_kv_consumer=True, config=None):
    if not enable:
        return

    global ENABLED
    ENABLED = True

    if config is not None:
        # update hyperparameters from command line args
        if "sink" in config:
            sink_val = config["sink"]
            if not isinstance(sink_val, int):
                raise ValueError(f"sink should be int, but is given {sink_val}")
            itfc.SINK = sink_val
        if "recent" in config:
            recent_val = config["recent"]
            if not isinstance(recent_val, int):
                raise ValueError(f"recent should be int, but is given {recent_val}")
            itfc.RECENT = recent_val
        if "beta" in config:
            beta_val = config["beta"]
            if not isinstance(beta_val, float) or beta_val <= 0 or beta_val >= 1:
                raise ValueError(f"beta should be float in (0,1), but is given {beta_val}")
            itfc.BETA = beta_val
        if "pattern" in config:
            pattern = config["pattern"]
            if not isinstance(pattern, list) or any(pi not in [0,1] for pi in pattern):
                raise ValueError(f"pattern should be a list of 0s and 1s, but is given {pattern}")
            itfc.PATTERN = pattern

    # kv_cache_utils.get_kv_cache_config = get_kv_cache_config_omni_type
    # engine_core.get_kv_cache_config = get_kv_cache_config_omni_type
    # GPUModelRunner.get_kv_cache_spec = get_nsa_kv_cache_spec

    if is_kv_consumer:
        block_table.MultiGroupBlockTable = OmniMultiGroupBlockTable
        gpu_input_batch.MultiGroupBlockTable = OmniMultiGroupBlockTable
    orig_manager.KVCacheBlocks = OmniKVCacheBlocks
    scheduler.KVCacheBlocks = OmniKVCacheBlocks
    orig_manager.KVCacheManager = NSAHostDeviceKVCacheManager
    scheduler.KVCacheManager = NSAHostDeviceKVCacheManager


__all__ = [
    "apply_omni_attn_patch",
    "check_omni_attn_cmd_arg",
]
