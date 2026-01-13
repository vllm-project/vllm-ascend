import math
import os
from typing import List, TYPE_CHECKING

from vllm_ascend.config.utils import is_moe_model
from vllm.logger import logger
if TYPE_CHECKING:
    from vllm.config import VllmConfig

def get_max_hidden_layers(hf_config) -> int:
    cfg_dict = hf_config.to_dict()
    layer_counts = []

    def _rec_find(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == "num_hidden_layers" and isinstance(v, int):
                    layer_counts.append(v)
                else:
                    _rec_find(v)

    _rec_find(cfg_dict)
    if not layer_counts:
        raise ValueError("Not found num_hidden_layers in model config.")
    return max(layer_counts)

# Update cudagraph capture sizes for vllm config
def update_cudagraph_capture_sizes(vllm_config: "VllmConfig",
                                   cudagraph_capture_sizes: List[int]):

    valid_max_size = (cudagraph_capture_sizes[-1]
                      if cudagraph_capture_sizes else 0)
    if (vllm_config.compilation_config.max_cudagraph_capture_size is not None
            and vllm_config.compilation_config.max_cudagraph_capture_size
            != valid_max_size):
        if vllm_config.compilation_config.cudagraph_capture_sizes is not None:
            raise ValueError(
                "customized max_cudagraph_capture_size"
                f"(={vllm_config.compilation_config.max_cudagraph_capture_size}) "
                "should be consistent with the max value of "
                f"cudagraph_capture_sizes(={valid_max_size})")
        logger.warning(
            "Truncating max_cudagraph_capture_size to %d",
            valid_max_size,
        )

    vllm_config.compilation_config.max_cudagraph_capture_size = valid_max_size

    if vllm_config.compilation_config.cudagraph_capture_sizes is not None and len(
            cudagraph_capture_sizes) < len(
                vllm_config.compilation_config.cudagraph_capture_sizes):
        logger.warning(
            ("cudagraph_capture_sizes specified in compilation_config"
             " %s is overridden by config %s"),
            vllm_config.compilation_config.cudagraph_capture_sizes,
            cudagraph_capture_sizes,
        )
    vllm_config.compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes
    vllm_config.compilation_config.post_init_cudagraph_sizes()


def _is_default_capture_sizes(vllm_config: "VllmConfig") -> bool:
    """
    Check whether it is vLLM default capture sizes.
    """

    max_cudagraph_capture_size = \
        vllm_config.compilation_config.max_cudagraph_capture_size
    cudagraph_capture_sizes = [
        i for i in [1, 2, 4] if i <= max_cudagraph_capture_size
    ]
    if max_cudagraph_capture_size >= 8:
        # Step size 8 for small batch sizes, up to 256(not included)
        cudagraph_capture_sizes += list(
            range(8, min(max_cudagraph_capture_size + 1, 256), 8))
    if max_cudagraph_capture_size >= 256:
        # Step size 16 for larger batch sizes
        cudagraph_capture_sizes += list(
            range(256, max_cudagraph_capture_size + 1, 16))
    # in newer version, vLLM use ascending order of cudagraph_capture_sizes.
    target_cudagraph_capture_sizes = sorted(cudagraph_capture_sizes)
    if target_cudagraph_capture_sizes == \
            vllm_config.compilation_config.cudagraph_capture_sizes:
        return True

    return False


def update_default_aclgraph_sizes(vllm_config: "VllmConfig") -> None:
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


def update_aclgraph_sizes(vllm_config: "VllmConfig") -> None:
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
