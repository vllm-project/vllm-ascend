import math
import os
from functools import lru_cache
from typing import Any

from vllm.logger import logger

from vllm_ascend import envs as envs_ascend
from vllm_ascend.config.vllm_ascend import get_ascend_config
from vllm_ascend.utils import VllmConfig, _is_default_capture_sizes, update_cudagraph_capture_sizes, \
    get_max_hidden_layers

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


_HAS_ROPE = None


def update_default_aclgraph_sizes(vllm_config: VllmConfig) -> None:
    """
    Update ACL graph default capture sizes, so that new sizes
    are more friendly to ascend ops && hardware.
    """

    if vllm_config.model_config is None or \
        vllm_config.model_config.enforce_eager or \
        not _is_default_capture_sizes(vllm_config):
        return

    # modify the default capture_sizes for Qwen3-MoE models on dp settings.
    # this is mainly because performance of _npu_paged_attention might degrades
    # on special shapes.
    # TODO(Angazenn): we will remove this once _npu_paged_attention is fully
    # replaced by npu_fused_infer_attention_score which does not contain such bugs.
    if vllm_config.model_config and vllm_config.model_config.hf_text_config.model_type == "qwen3_moe" \
        and vllm_config.parallel_config.tensor_parallel_size == 1 \
        and vllm_config.parallel_config.data_parallel_size > 1 :

        max_capture_size = vllm_config.compilation_config.max_cudagraph_capture_size
        new_cudagraph_capture_sizes = [1, 2, 5, 10, 15, 20] + [
            i for i in range(24, max_capture_size + 1, 8)
        ]
        update_cudagraph_capture_sizes(vllm_config,
                                       new_cudagraph_capture_sizes)


def update_aclgraph_sizes(vllm_config: VllmConfig) -> None:
    """Update ACL graph capture sizes based on hardware limitations"""
    # NOTE: Currently, we can only capture 1800 graphs at most,
    # due to the limitation of ACL graph. This number is bounded by
    # the number of streams, which is 2048, we save 248 streams
    # as a buffer.
    # Maximum number of graphs that can be captured by ACL Graph
    # TODO: Find out whether we need to solve allreduce function
    MAX_CAPTURE_SIZE = 1800

    # enable pcp or dcp will add new communication and consume additional approximately less than 100 streams
    CP_ADDITIONAL_STREAM_NUM = 100

    # Store original configuration and temporarily clear it
    compilation_config = vllm_config.compilation_config
    original_sizes, compilation_config.cudagraph_capture_sizes = \
        compilation_config.cudagraph_capture_sizes, None

    # Calculate parallel configuration factor
    if not vllm_config.model_config:
        logger.warning(
            "Got empty model config. This typically occurs when an empty vllm_config is "
            "initialized (e.g., in unit tests), where config updates are intentionally skipped."
        )

        return
    hf_config = vllm_config.model_config.hf_text_config
    if hasattr(hf_config, 'num_hidden_layers'):
        num_hidden_layers = hf_config.num_hidden_layers
    else:
        num_hidden_layers = get_max_hidden_layers(hf_config)
    parallel_config = vllm_config.parallel_config

    # Calculate maximum supported batch sizes considering model architecture
    resources_per_graph = num_hidden_layers + 1
    # For suffix decoding, use the suffix path when no draft_model_config is provided.
    if (spec := vllm_config.speculative_config) and \
    (draft := spec.draft_model_config):
        resources_per_graph += draft.hf_config.num_hidden_layers + 1

    # TODO: Find out whether we need to take into account the pp_size
    num_comm_groups = sum(size > 1 for size in [
        parallel_config.data_parallel_size,
        parallel_config.tensor_parallel_size,
    ])

    if os.getenv("HCCL_OP_EXPANSION_MODE") == 'AIV':
        # TODO: Find out whether we need to take into account the pp_size
        parallel_factor = 1 + num_comm_groups + int(
            parallel_config.enable_expert_parallel) + int(
                vllm_config.additional_config.get(
                    "multistream_overlap_shared_expert", False))
        if is_moe_model(vllm_config):
            parallel_factor += (parallel_config.data_parallel_size > 1)
        else:
            # When AIV mode is enabled, the allreduce operator of the dense
            # layer model will occupy additional streams, which are buffered here.
            MAX_CAPTURE_SIZE = MAX_CAPTURE_SIZE - parallel_factor * resources_per_graph

        # Calculate maximum supported batch sizes considering model architecture on the A2 Hardware Device
        # Assume the following case:
        # MAX_CAPTURE_SIZE = 1920, num_hidden_layers = 48, data_parallel_size is 1, tensor_parallel_size is 4,
        # According to the formula, max_num_batch_sizes = math.floor(1920 / (48 + 1) / 2) = 19
        max_num_batch_sizes = math.floor(MAX_CAPTURE_SIZE /
                                         resources_per_graph / parallel_factor)
        logger.info(
            "Calculated maximum supported batch sizes for ACL graph: %s",
            max_num_batch_sizes)
    else:
        # enable pcp or dcp will add new communication and consume additional approximately less than 100 streams
        if parallel_config.prefill_context_parallel_size > 1:
            MAX_CAPTURE_SIZE = MAX_CAPTURE_SIZE - CP_ADDITIONAL_STREAM_NUM
        if parallel_config.decode_context_parallel_size > 1:
            MAX_CAPTURE_SIZE = MAX_CAPTURE_SIZE - CP_ADDITIONAL_STREAM_NUM

        # The above describes an empirical formula applicable to the A2 hardware.
        # Under this configuration, HCCL employs the FFTS+ method for execution unfolding,
        # which adds only 1 concurrent stream without consuming collective communication execution unfolding streams.
        # On A3 hardware, HCCL defaults to the AICPU method.
        # This approach may additionally allocate up to rank_size (max 16) - 1 streams per collective communication domain on the device (worst case).
        # Using the default collective communication unfolding method on A3 will lead to a significant reduction in the maximum supported sizes.
        # Therefore, the calculation formula has been modified as follows:
        # Assume the following case:
        # MAX_CAPTURE_SIZE = 1920, num_hidden_layers = 48, data_parallel_size is 1, tensor_parallel_size is 4,
        # According to the formula, max_num_batch_sizes = math.floor((1920 - 1 * 40) / (48 + 1) / (1 + 1 * 2)) = 12
        max_num_batch_sizes = math.floor(
            (MAX_CAPTURE_SIZE - num_comm_groups * 40) / resources_per_graph /
            (1 + num_comm_groups * 2))
        logger.info(
            "Calculated maximum supported batch sizes for ACL graph: %s",
            max_num_batch_sizes)
        logger.warning(
            "Currently, communication is performed using FFTS+ method, which reduces "
            "the number of available streams and, as a result, limits the range of runtime "
            "shapes that can be handled. To both improve communication performance and "
            "increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV."
        )

    # If original sizes exceed maximum, sample a representative subset
    if max_num_batch_sizes < len(original_sizes):
        # Sample uniformly from original sizes
        step = (len(original_sizes) - 1) / (max_num_batch_sizes - 1)
        indices = [round(i * step) for i in range(max_num_batch_sizes)]

        # Ensure first and last elements are preserved
        indices[0], indices[-1] = 0, len(original_sizes) - 1

        sampled_sizes = [original_sizes[i] for i in indices]
        update_cudagraph_capture_sizes(vllm_config, sampled_sizes)

        logger.info(
            "Adjusted ACL graph batch sizes for %s model (layers: %d): %d → %d sizes",
            vllm_config.model_config.architectures[0],
            num_hidden_layers,
            len(original_sizes),
            len(compilation_config.
                cudagraph_capture_sizes  # type: ignore[arg-type]
                ))
    else:
        # No adjustment needed
        compilation_config.cudagraph_capture_sizes = original_sizes
        logger.info(
            "No adjustment needed for ACL graph batch sizes: %s model (layers: %d) with %d sizes",
            vllm_config.model_config.architectures[0], num_hidden_layers,
            len(original_sizes))


def lmhead_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.lmhead_tensor_parallel_size > 0


def embedding_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.embedding_tensor_parallel_size > 0


def oproj_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.oproj_tensor_parallel_size > 0


def mlp_tp_enable() -> bool:
    return get_ascend_config(
    ).finegrained_tp_config.mlp_tensor_parallel_size > 0


def matmul_allreduce_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE


def shared_expert_dp_enabled() -> bool:
    return get_ascend_config().enable_shared_expert_dp or enable_sp()


def prefill_context_parallel_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL


def has_rope(vllm_config: VllmConfig):
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
