import os
from functools import lru_cache
from typing import TYPE_CHECKING

from vllm.logger import logger

from vllm_ascend import envs as envs_ascend
from vllm_ascend.config.vllm_ascend import get_ascend_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_IS_MOE_MODEL = None
_IS_VL_MODEL = None
_ENABLE_SP = None
_HAS_ROPE = None


def matmul_allreduce_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE


def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    global _ENABLE_SP
    if _ENABLE_SP is None:
        if vllm_config is None:
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
        _ENABLE_SP = (
            vllm_config.compilation_config.pass_config.enable_sp
            or envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1
            # Flash comm 1 should be enabled by env VLLM_ASCEND_ENABLE_FLASHCOMM1
            # We retain the env VLLM_ASCEND_ENABLE_FLASHCOMM here for backward compatibility.
            or bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", '0'))))

        if not _ENABLE_SP and enable_shared_expert_dp:
            _ENABLE_SP = True
            logger.info(
                "shared_expert_dp requires enable_sp = True. has set enable_sp to True"
            )

        if not _ENABLE_SP:
            return _ENABLE_SP

        assert vllm_config.parallel_config.tensor_parallel_size > 1, \
            "Flash Comm v1 (Sequence Parallelism) is only supported when tp_size > 1."

        assert (
            not is_moe_model(vllm_config)
            or vllm_config.parallel_config.enable_expert_parallel
        ), "Flash Comm v1 (Sequence Parallelism) requires enable_expert_parallel=True for MoE models."

    return _ENABLE_SP


def shared_expert_dp_enabled() -> bool:
    return get_ascend_config().enable_shared_expert_dp or enable_sp()


def prefill_context_parallel_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL


def _is_contain_expert(config: "VllmConfig"):
    if isinstance(config, dict):
        for k, v in config.items():
            if "expert" in str(k):
                return True
            if _is_contain_expert(v):
                return True
    return False


def is_moe_model(vllm_config: "VllmConfig"):
    """Checks if the model is a MoE model by config"""
    global _IS_MOE_MODEL
    if _IS_MOE_MODEL is None:
        model_configs = vllm_config.model_config.hf_text_config.to_dict()
        _IS_MOE_MODEL = _is_contain_expert(model_configs)
    return _IS_MOE_MODEL


def is_vl_model(vllm_config: "VllmConfig"):
    """Checks if the model is a VL model by config"""
    global _IS_VL_MODEL
    if _IS_VL_MODEL is None and vllm_config and vllm_config.model_config:
        hf_config = vllm_config.model_config.hf_config.to_dict()
        if "thinker_config" in hf_config:
            # Qwen-Omni-thinker models
            _IS_VL_MODEL = True
        else:
            _IS_VL_MODEL = "vision_config" in hf_config
    return _IS_VL_MODEL


def has_rope(vllm_config: "VllmConfig"):
    """Checks if the model uses rope."""
    global _HAS_ROPE
    if _HAS_ROPE is None and vllm_config and vllm_config.model_config:
        hf_config = vllm_config.model_config.hf_text_config.to_dict()
        _HAS_ROPE = "rope_parameters" in hf_config
    return _HAS_ROPE


def flashcomm2_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE > 0


def o_shard_enable() -> bool:
    layer_sharding = get_ascend_config().layer_sharding
    if layer_sharding is None:
        return False
    return "o_proj" in layer_sharding


def get_flashcomm2_config_and_validate(ascend_config, vllm_config):
    flashcomm2_oproj_tp_size = envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE
    global_tp_size = vllm_config.parallel_config.tensor_parallel_size

    if not flashcomm2_enable():
        return 0

    logger.info(
        f"Enable FLASHCOMM2 with flashcomm2_oproj_tensor_parallel_size = {flashcomm2_oproj_tp_size}"
    )

    layer_sharding = ascend_config.layer_sharding or []
    if layer_sharding:
        if layer_sharding == ["o_proj"]:
            logger.info_once(
                "Enable FLASHCOMM2 with o_proj layer sharding for reduced memory consumption."
            )
        else:
            raise ValueError(
                "FLASHCOMM2 only supports 'o_proj' as the sole layer sharding configuration! "
                f"Found invalid layer_sharding: {layer_sharding}")
    if not envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1:
        logger.warning_once(
            "It is recommended to enable FLASHCOMM1 simultaneously when starting FLASHCOMM2 for optimal performance."
        )
    if ascend_config.finegrained_tp_config.oproj_tensor_parallel_size > 0:
        raise AssertionError(
            "flashcomm2_oproj_tensor_parallel_size cannot be enabled simultaneously with oproj_tensor_parallel_size"
        )
    if global_tp_size <= flashcomm2_oproj_tp_size:
        raise AssertionError(
            f"flashcomm2_oproj_tensor_parallel_size ({flashcomm2_oproj_tp_size}) cannot exceed global tensor parallel size ({global_tp_size})"
        )
    if global_tp_size % flashcomm2_oproj_tp_size != 0:
        raise AssertionError(
            f"Global tensor parallel size ({global_tp_size}) must be divisible by flashcomm2_oproj_tensor_parallel_size ({flashcomm2_oproj_tp_size})"
        )
    if vllm_config.kv_transfer_config is None:
        logger.warning_once(
            "It is recommended to enable FLASHCOMM2 in P-scenario deployments, enable it in hybrid deployment may lead to decode performance degradation."
        )
    if vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_consumer:
        raise AssertionError(
            "FLASHCOMM2 primarily targets P-scenario deployments, with additional support for hybrid deployment scenarios. It is not applicable in D-scenario environments."
        )

    return flashcomm2_oproj_tp_size


def refresh_block_size(vllm_config):
    """
    Refresh the block size in cache config.
    """
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config

    if not cache_config:
        return

    if cache_config.block_size is None:
        cache_config.block_size = 128

    if not scheduler_config or not model_config:
        return

    # TODO(MengqingCao): Remove the model_type check, after resolving the hidden error in get_kv_cache_groups.
    if not model_config.hf_text_config.model_type == "qwen3_next" and cache_config.block_size != 128:
        if cache_config.enable_prefix_caching or scheduler_config.enable_chunked_prefill:
            logger.info(
                "Block size is set to 128 if prefix cache or chunked prefill is enabled."
            )
            cache_config.block_size = 128


def check_kv_extra_config(vllm_config):

    def _check(name: str, config: dict):
        tp_key = "tp_size"
        dp_key = "dp_size"
        if tp_key in config:
            config_tp = config[tp_key]
            vllm_tp = vllm_config.parallel_config.tensor_parallel_size
            if config_tp != vllm_tp:
                raise ValueError(
                    f"KV transfer '{name}' config has a conflicting tensor parallel size. "
                    f"Expected {vllm_tp}, but got {config_tp}.")
        if dp_key in config:
            config_dp = config[dp_key]
            vllm_dp = vllm_config.parallel_config.data_parallel_size
            if config_dp != vllm_dp:
                raise ValueError(
                    f"KV transfer '{name}' config has a conflicting data parallel size. "
                    f"Expected {vllm_dp}, but got {config_dp}.")

    if vllm_config.kv_transfer_config.is_kv_producer:
        _check(
            "prefill",
            vllm_config.kv_transfer_config.get_from_extra_config(
                "prefill", {}))
    if vllm_config.kv_transfer_config.is_kv_consumer:
        _check(
            "decode",
            vllm_config.kv_transfer_config.get_from_extra_config("decode", {}))


@lru_cache(maxsize=1)
def enable_dsa_cp() -> bool:
    from vllm.config import get_current_vllm_config
    vllm_config = get_current_vllm_config()
    is_ds_v32 = hasattr(
        vllm_config.model_config, "hf_text_config") and hasattr(
            vllm_config.model_config.hf_text_config, "index_topk")
    if is_ds_v32 and enable_sp():
        return True
    return False


@lru_cache(maxsize=1)
def enable_dsa_cp_with_layer_shard() -> bool:
    if not enable_dsa_cp():
        return False
    from vllm.config import get_current_vllm_config
    vllm_config = get_current_vllm_config()
    is_prefill_instance = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_producer
    return is_prefill_instance
