import os
from typing import Any

from vllm.logger import logger

from vllm_ascend import envs as envs_ascend

_IS_MOE_MODEL = None
_IS_VL_MODEL = None
_ENABLE_SP = None


def _is_contain_expert(config: Any):
    if isinstance(config, dict):
        for k, v in config.items():
            if "expert" in str(k):
                return True
            if _is_contain_expert(v):
                return True
    return False

def is_moe_model(vllm_config: Any):
    """Checks if the model is a MoE model by config"""
    global _IS_MOE_MODEL
    if _IS_MOE_MODEL is None:
        model_configs = vllm_config.model_config.hf_text_config.to_dict()
        _IS_MOE_MODEL = _is_contain_expert(model_configs)
    return _IS_MOE_MODEL


def is_vl_model(vllm_config: Any):
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
