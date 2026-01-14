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

"""
Utilities module for vLLM Ascend.

This module provides utility functions organized by functionality:
- tensor_utils: Tensor format and shape operations
- device_utils: Device type and management
- stream_utils: Stream management
- communication_utils: HCCL and communication
- graph_utils: ACL/CUDA graph configuration
- model_utils: Model type detection
- profiler: Performance profiling
- debug_utils: Debug and printing utilities
- config_utils: Configuration management
- parallel_config: Parallelism flags and configuration
- custom_ops: Custom operation registration
"""

# Import all utilities for backward compatibility
from vllm_ascend.utils.tensor_utils import (
    ACL_FORMAT_FRACTAL_ND,
    ACL_FORMAT_FRACTAL_NZ,
    _custom_pad,
    _custom_reshape,
    _custom_transpose,
    _round_up,
    aligned_16,
    dispose_layer,
    dispose_tensor,
    maybe_trans_nz,
    nd_to_nz_2d,
    nd_to_nz_spec,
)

from vllm_ascend.utils.device_utils import (
    AscendDeviceType,
    check_ascend_device_type,
    get_ascend_device_type,
)

from vllm_ascend.utils.stream_utils import (
    _CURRENT_STREAM,
    _GLOBAL_STREAM,
    _PREFETCH_STREAM,
    _SHARED_EXPERTS_CALCULATION_STREAM,
    _CP_CHUNKEDPREFILL_COMM_STREAM,
    _WEIGHT_PREFETCH_METHOD,
    cp_chunkedprefill_comm_stream,
    current_stream,
    global_stream,
    npu_stream_switch,
    prefetch_stream,
    set_weight_prefetch_method,
    shared_experts_calculation_stream,
    get_weight_prefetch_method,
)

from vllm_ascend.utils.communication_utils import (
    calculate_dp_buffer_size,
    find_hccl_library,
    get_default_buffer_config,
    get_hccl_config_for_pg_options,
    is_hierarchical_communication_enabled,
)

from vllm_ascend.utils.parallel_config import (
    _DEFAULT_BUFFER_SIZE,
    _MIN_DP_BUFFER_SIZE,
    create_hccl_pg_options,
    embedding_tp_enable,
    enable_dsa_cp,
    enable_dsa_cp_with_layer_shard,
    enable_sp,
    flashcomm2_enable,
    get_flashcomm2_config_and_validate,
    get_flashcomm2_reorgnized_batch_ids,
    lmhead_tp_enable,
    matmul_allreduce_enable,
    mlp_tp_enable,
    o_shard_enable,
    oproj_tp_enable,
    prefill_context_parallel_enable,
    shared_expert_dp_enabled,
)

from vllm_ascend.utils.graph_utils import (
    _is_default_capture_sizes,
    update_aclgraph_sizes,
    update_cudagraph_capture_sizes,
    update_default_aclgraph_sizes,
)

from vllm_ascend.utils.model_utils import (
    _HAS_LAYER_IDX,
    _HAS_ROPE,
    _IS_DRAFTER_MOE_MODEL,
    _IS_MOE_MODEL,
    _IS_VL_MODEL,
    _is_contain_expert,
    get_max_hidden_layers,
    has_layer_idx,
    has_rope,
    is_drafter_moe_model,
    is_moe_model,
    is_vl_model,
    speculative_enable_dispatch_gmm_combine_decode,
)

from vllm_ascend.utils.profiler import ProfileExecuteDuration

from vllm_ascend.utils.debug_utils import acl_graph_print

from vllm_ascend.utils.config_utils import (
    check_kv_extra_config,
    refresh_block_size,
    singleton,
    vllm_version_is,
)

from vllm_ascend.utils.custom_ops import (
    REGISTERED_ASCEND_OPS,
    _ASCEND_CUSTOMOP_IS_REIGISTERED,
    _CUSTOM_OP_ENABLED,
    enable_custom_op,
    register_ascend_customop,
)

from vllm_ascend.utils.weak_ref_utils import (
    weak_ref_tensor,
    weak_ref_tensors,
)

from vllm_ascend.utils.version_utils import (
    adapt_patch,
)

__all__ = [
    # Tensor utilities
    "ACL_FORMAT_FRACTAL_ND",
    "ACL_FORMAT_FRACTAL_NZ",
    "_custom_pad",
    "_custom_reshape",
    "_custom_transpose",
    "_round_up",
    "aligned_16",
    "dispose_layer",
    "dispose_tensor",
    "maybe_trans_nz",
    "nd_to_nz_2d",
    "nd_to_nz_spec",
    # Device utilities
    "AscendDeviceType",
    "check_ascend_device_type",
    "get_ascend_device_type",
    # Stream utilities
    "_CURRENT_STREAM",
    "_GLOBAL_STREAM",
    "_PREFETCH_STREAM",
    "_SHARED_EXPERTS_CALCULATION_STREAM",
    "_CP_CHUNKEDPREFILL_COMM_STREAM",
    "_WEIGHT_PREFETCH_METHOD",
    "cp_chunkedprefill_comm_stream",
    "current_stream",
    "global_stream",
    "npu_stream_switch",
    "prefetch_stream",
    "set_weight_prefetch_method",
    "shared_experts_calculation_stream",
    "get_weight_prefetch_method",
    # Communication utilities
    "calculate_dp_buffer_size",
    "find_hccl_library",
    "get_default_buffer_config",
    "get_hccl_config_for_pg_options",
    "is_hierarchical_communication_enabled",
    # Parallel config
    "_DEFAULT_BUFFER_SIZE",
    "_MIN_DP_BUFFER_SIZE",
    "create_hccl_pg_options",
    "embedding_tp_enable",
    "enable_dsa_cp",
    "enable_dsa_cp_with_layer_shard",
    "enable_sp",
    "flashcomm2_enable",
    "get_flashcomm2_config_and_validate",
    "get_flashcomm2_reorgnized_batch_ids",
    "lmhead_tp_enable",
    "matmul_allreduce_enable",
    "mlp_tp_enable",
    "o_shard_enable",
    "oproj_tp_enable",
    "prefill_context_parallel_enable",
    "shared_expert_dp_enabled",
    # Graph utilities
    "_is_default_capture_sizes",
    "update_aclgraph_sizes",
    "update_cudagraph_capture_sizes",
    "update_default_aclgraph_sizes",
    # Model utilities
    "_HAS_LAYER_IDX",
    "_HAS_ROPE",
    "_IS_DRAFTER_MOE_MODEL",
    "_IS_MOE_MODEL",
    "_IS_VL_MODEL",
    "_is_contain_expert",
    "get_max_hidden_layers",
    "has_layer_idx",
    "has_rope",
    "is_drafter_moe_model",
    "is_moe_model",
    "is_vl_model",
    "speculative_enable_dispatch_gmm_combine_decode",
    # Profiler
    "ProfileExecuteDuration",
    # Debug utilities
    "acl_graph_print",
    # Config utilities
    "check_kv_extra_config",
    "refresh_block_size",
    "singleton",
    "vllm_version_is",
    # Custom ops
    "REGISTERED_ASCEND_OPS",
    "_ASCEND_CUSTOMOP_IS_REIGISTERED",
    "_CUSTOM_OP_ENABLED",
    "enable_custom_op",
    "register_ascend_customop",
    # Weak ref
    "weak_ref_tensor",
    "weak_ref_tensors",
    # Version utils
    "adapt_patch",
]

# Re-export constants from parent utils module
import vllm_ascend.envs as envs_ascend

COMPILATION_PASS_KEY = "graph_fusion_manager"
ASCEND_QUANTIZATION_METHOD = "ascend"
COMPRESSED_TENSORS_METHOD = "compressed-tensors"
SOC_VERSION_INFERENCE_SERIES = ["Ascend310P3"]

__all__.extend([
    "COMPILATION_PASS_KEY",
    "ASCEND_QUANTIZATION_METHOD",
    "COMPRESSED_TENSORS_METHOD",
    "SOC_VERSION_INFERENCE_SERIES",
])
