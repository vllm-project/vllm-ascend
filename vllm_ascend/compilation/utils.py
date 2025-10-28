#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
import math
import os
from typing import List

from vllm.logger import logger

from vllm_ascend.torchair.utils import (check_torchair_cache_exist,
                                        delete_torchair_cache_file)
from vllm_ascend.utils import (enable_sp, get_max_hidden_layers, is_310p,
                               is_moe_model, vllm_version_is)


def update_compilation_config(vllm_config, ascend_config, enforce_eager):
    from vllm.config import CompilationMode
    from vllm.config.compilation import CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    if enforce_eager:
        logger.info("Compilation disabled, using eager mode by default")
        compilation_config.mode = CompilationMode.NONE
    compilation_config.cudagraph_num_of_warmups = 1
    if compilation_config.mode not in [
            CompilationMode.NONE, CompilationMode.VLLM_COMPILE
    ]:
        logger.warning(
            "NPU does not support %s compilation mode. Setting CUDAGraphMode to NONE",
            compilation_config.mode)
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    # set CUDAGraphMode to None when torchair is enabled, no mather what compilation_config.level is.
    if ascend_config.torchair_graph_config.enabled:
        logger.info(
            "Torchair compilation enabled on NPU. Setting CUDAGraphMode to NONE"
        )
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        # Note: We delete the torchair cache folder here to prevent runtime issues caused by dimension
        # mismatches or configuration inconsistencies when users reuse cached computation graphs. Though
        # this will increase graph compilation duration, it significantly enhances robustness and decreases
        # graph launching time during inference.
        if check_torchair_cache_exist(
        ) and not ascend_config.torchair_graph_config.use_cached_kv_cache_bytes:
            logger.warning(
                "Torchair cache folder is deleted here to prevent runtime issues caused by dimension "
                "mismatches or configuration inconsistencies when users reuse cached computation graphs. "
                "In order to decrease torchair graph compilation time, users can enable both use_cached_graph "
                "and use_cached_kv_cache_bytes in torchair_graph_config.")
            delete_torchair_cache_file()

    # TODO delete graph size update here when compilation_config.pass_config.enable_sequence_parallelism
    # is supported by vllm-ascend.
    if vllm_config.parallel_config.tensor_parallel_size > 1 and not vllm_config.model_config.enforce_eager and \
            enable_sp(vllm_config):
        original_sizes = compilation_config.cudagraph_capture_sizes
        sp_aclgraph_sizes = \
            vllm_config.update_sizes_for_sequence_parallelism(original_sizes)
        assert sp_aclgraph_sizes, (
            f"cudagraph_capture_sizes {original_sizes} does not contain"
            f"values that are multiples of tp_size "
            f"{vllm_config.parallel_config.tensor_parallel_size}")
        if len(sp_aclgraph_sizes) != len(original_sizes):
            compilation_config.cudagraph_capture_sizes = sp_aclgraph_sizes
            _update_cudagraph_capture_sizes(vllm_config, sp_aclgraph_sizes)

    # TODO: Full graph is fully supported later, and the default value will be set to full graph.
    if compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE:
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    if compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        compilation_config.mode = CompilationMode.NONE
    elif compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE:
        logger.info(
            "PIECEWISE compilation enabled on NPU. use_inductor not supported - "
            "using only ACL Graph mode")
        assert compilation_config.mode == CompilationMode.VLLM_COMPILE, \
            "When enabling VLLM_COMPILE aclgraph, please make sure compilation_config.mode == CompilationMode.VLLM_COMPILE and compilation_config.cudagraph_mode == CUDAGraphMode.VLLM_COMPILE"
        compilation_config.set_splitting_ops_for_v1()
        compilation_config.use_inductor = False
        compilation_config.splitting_ops.extend(["vllm::mla_forward"])
        _update_aclgraph_sizes(vllm_config)
    elif compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
        logger.info(
            "FULL_DECODE_ONLY compilation enabled on NPU. use_inductor not supported - "
            "using only ACL Graph mode")
        compilation_config.use_inductor = False
        warning_message = """\033[91m
        **********************************************************************************
        * WARNING: You have enabled the *full graph* feature.
        * This is an early experimental stage and may involve various unknown issues.
        * A known problem is that capturing too many batch sizes can lead to OOM
        * (Out of Memory) errors or inference hangs. If you encounter such issues,
        * consider reducing `gpu_memory_utilization` or manually specifying a smaller
        * batch size for graph capture.
        * For more details, please refer to:
        * https://docs.vllm.ai/en/stable/configuration/conserving_memory.html#reduce-cuda-graphs
        **********************************************************************************\033[0m
        """
        logger.warning(warning_message)
    else:
        logger.info(
            "%s cudagraph_mode is not support on NPU. falling back to NONE",
            compilation_config.cudagraph_mode)
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        compilation_config.mode = CompilationMode.NONE

    # TODO: Remove this check when ACL Graph supports ASCEND_LAUNCH_BLOCKING=1
    # Then, we will have to discuss the error handling strategy and user experience
    if compilation_config.cudagraph_mode != CUDAGraphMode.NONE and \
        os.environ.get("ASCEND_LAUNCH_BLOCKING", "0") == "1":
        raise ValueError(
            "ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1. "
            "Please unset ASCEND_LAUNCH_BLOCKING or set it to 0. If you "
            "need ASCEND_LAUNCH_BLOCKING for debugging, consider other methods — "
            "for example, check the plog files (default: $HOME/ascend/log/debug) "
            "for more information about runtime errors.")
    # Activate custom ops for v1, except on 310P
    if not is_310p():
        compilation_config.custom_ops = ["all"]


def update_compilation_config_v0_11_0(vllm_config, ascend_config,
                                      enforce_eager):
    from vllm.config import CompilationLevel
    from vllm.config.compilation import CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    if enforce_eager:
        logger.info("Compilation disabled, using eager mode by default")
        compilation_config.level = CompilationLevel.NO_COMPILATION
    compilation_config.cudagraph_num_of_warmups = 1

    if compilation_config.level not in [
            CompilationLevel.NO_COMPILATION, CompilationLevel.PIECEWISE
    ]:
        logger.warning(
            "NPU does not support %s compilation level. Setting CUDAGraphMode to NONE",
            compilation_config.level)
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    # set CUDAGraphMode to None when torchair is enabled, no mather what compilation_config.level is.
    if ascend_config.torchair_graph_config.enabled:
        logger.info(
            "Torchair compilation enabled on NPU. Setting CUDAGraphMode to NONE"
        )
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        # Note: We delete the torchair cache folder here to prevent runtime issues caused by dimension
        # mismatches or configuration inconsistencies when users reuse cached computation graphs. Though
        # this will increase graph compilation duration, it significantly enhances robustness and decreases
        # graph launching time during inference.
        if check_torchair_cache_exist(
        ) and not ascend_config.torchair_graph_config.use_cached_kv_cache_bytes:
            logger.warning(
                "Torchair cache folder is deleted here to prevent runtime issues caused by dimension "
                "mismatches or configuration inconsistencies when users reuse cached computation graphs. "
                "In order to decrease torchair graph compilation time, users can enable both use_cached_graph "
                "and use_cached_kv_cache_bytes in torchair_graph_config.")
            delete_torchair_cache_file()

    # TODO delete graph size update here when compilation_config.pass_config.enable_sequence_parallelism
    # is supported by vllm-ascend.
    if vllm_config.parallel_config.tensor_parallel_size > 1 and not vllm_config.model_config.enforce_eager and \
            enable_sp(vllm_config):
        original_sizes = compilation_config.cudagraph_capture_sizes
        sp_aclgraph_sizes = \
            vllm_config.update_sizes_for_sequence_parallelism(original_sizes)
        assert sp_aclgraph_sizes, (
            f"cudagraph_capture_sizes {original_sizes} does not contain"
            f"values that are multiples of tp_size "
            f"{vllm_config.parallel_config.tensor_parallel_size}")
        if len(sp_aclgraph_sizes) != len(original_sizes):
            compilation_config.cudagraph_capture_sizes = sp_aclgraph_sizes
            compilation_config.init_with_cudagraph_sizes(sp_aclgraph_sizes)

    # TODO: Full graph is fully supported later, and the default value will be set to full graph.
    if compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE:
        compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

    if compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        compilation_config.level = CompilationLevel.NO_COMPILATION
    elif compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE:
        logger.info(
            "PIECEWISE compilation enabled on NPU. use_inductor not supported - "
            "using only ACL Graph mode")
        assert compilation_config.level == CompilationLevel.PIECEWISE, \
            "When enabling piecewise aclgraph, please make sure compilation_config.level == CompilationLevel.PIECEWISE and compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE"
        compilation_config.set_splitting_ops_for_v1()
        compilation_config.use_inductor = False
        compilation_config.splitting_ops.extend(
            ["vllm.unified_ascend_attention_with_output", "vllm.mla_forward"])
        _update_aclgraph_sizes(vllm_config)
    elif compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY:
        logger.info(
            "FULL_DECODE_ONLY compilation enabled on NPU. use_inductor not supported - "
            "using only ACL Graph mode")
        compilation_config.use_inductor = False
        warning_message = """\033[91m
        **********************************************************************************
        * WARNING: You have enabled the *full graph* feature.
        * This is an early experimental stage and may involve various unknown issues.
        * A known problem is that capturing too many batch sizes can lead to OOM
        * (Out of Memory) errors or inference hangs. If you encounter such issues,
        * consider reducing `gpu_memory_utilization` or manually specifying a smaller
        * batch size for graph capture.
        * For more details, please refer to:
        * https://docs.vllm.ai/en/stable/configuration/conserving_memory.html#reduce-cuda-graphs
        **********************************************************************************\033[0m
        """
        logger.warning(warning_message)
    else:
        logger.info(
            "%s cudagraph_mode is not support on NPU. falling back to NONE",
            compilation_config.cudagraph_mode)
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        compilation_config.level = CompilationLevel.NO_COMPILATION

    # TODO: Remove this check when ACL Graph supports ASCEND_LAUNCH_BLOCKING=1
    # Then, we will have to discuss the error handling strategy and user experience
    if compilation_config.cudagraph_mode != CUDAGraphMode.NONE and \
        os.environ.get("ASCEND_LAUNCH_BLOCKING", "0") == "1":
        raise ValueError(
            "ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1. "
            "Please unset ASCEND_LAUNCH_BLOCKING or set it to 0. If you "
            "need ASCEND_LAUNCH_BLOCKING for debugging, consider other methods — "
            "for example, check the plog files (default: $HOME/ascend/log/debug) "
            "for more information about runtime errors.")
    # Activate custom ops for v1, except on 310P
    if not is_310p():
        compilation_config.custom_ops = ["all"]


def _update_aclgraph_sizes(vllm_config) -> None:
    """Update ACL graph capture sizes based on hardware limitations"""
    # NOTE: Currently, we can only capture 1800 graphs at most,
    # due to the limitation of ACL graph. This number is bounded by
    # the number of streams, which is 2048, we save 248 streams
    # as a buffer.
    # Maximum number of graphs that can be captured by ACL Graph
    # TODO: Find out whether we need to solve allreduce function
    MAX_CAPTURE_SIZE = 1800

    # Store original configuration and temporarily clear it
    compilation_config = vllm_config.compilation_config
    original_sizes, compilation_config.cudagraph_capture_sizes = \
        compilation_config.cudagraph_capture_sizes, None

    # Calculate parallel configuration factor
    hf_config = vllm_config.model_config.hf_config
    if hasattr(hf_config, 'num_hidden_layers'):
        num_hidden_layers = hf_config.num_hidden_layers
    else:
        num_hidden_layers = get_max_hidden_layers(hf_config)
    parallel_config = vllm_config.parallel_config

    # Calculate maximum supported batch sizes considering model architecture
    resources_per_graph = num_hidden_layers + 1
    if vllm_config.speculative_config is not None:
        draft_model_hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        resources_per_graph += draft_model_hf_config.num_hidden_layers + 1

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
        if vllm_version_is("0.11.0"):
            compilation_config.init_with_cudagraph_sizes(sampled_sizes)
        else:
            _update_cudagraph_capture_sizes(vllm_config, sampled_sizes)

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

    # default or defined cudagraph_capture_sizes may not consider num_speculative_tokens>1 scenario
    # the maximum size cudagraph_capture_sizes[0] should be greater or equal than
    # (num_speculative_tokens+1)*max_num_seqs, otherwise draft model will run in eager mode
    if vllm_config.speculative_config is not None and \
        vllm_config.speculative_config.num_speculative_tokens > 1:
        num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        original_sizes, compilation_config.cudagraph_capture_sizes = \
            compilation_config.cudagraph_capture_sizes, None
        assert len(original_sizes) > 0
        if original_sizes[0] < (num_speculative_tokens + 1) * max_num_seqs:
            enlarged_sizes = [(num_speculative_tokens + 1) * size
                              for size in original_sizes]
            if vllm_version_is("0.11.0"):
                compilation_config.init_with_cudagraph_sizes(enlarged_sizes)
            else:
                _update_cudagraph_capture_sizes(vllm_config, enlarged_sizes)
            logger.info(
                "Adjusted ACL graphs: %s → %s for speculative decoding",
                original_sizes, enlarged_sizes)
        else:
            compilation_config.cudagraph_capture_sizes = original_sizes


# Update cudagraph capture sizes for vllm config
def _update_cudagraph_capture_sizes(vllm_config,
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
