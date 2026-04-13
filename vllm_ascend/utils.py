#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from vllm-project/vllm/vllm/worker/worker.py
#

from __future__ import annotations

import functools
import math
import os
from contextlib import nullcontext
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import regex as re
import torch
import torch_npu  # noqa: F401
from packaging.version import InvalidVersion, Version
from vllm.logger import logger
from vllm.sequence import IntermediateTensors

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import WeightPrefetchConfig, get_ascend_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

COMPILATION_PASS_KEY = "graph_fusion_manager"
ASCEND_QUANTIZATION_METHOD = "ascend"
COMPRESSED_TENSORS_METHOD = "compressed-tensors"
SOC_VERSION_INFERENCE_SERIES = ["Ascend310P3"]
REGISTERED_ASCEND_OPS = {}

ACL_FORMAT_FRACTAL_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29

_CUSTOM_OP_ENABLED = None
_DEVICE_PRINT_OP_REGISTERED = False
_CURRENT_STREAM = None
_PREFETCH_STREAM = None
_WEIGHT_PREFETCH_METHOD = None
_GLOBAL_STREAM = None
_SHARED_EXPERTS_CALCULATION_STREAM = None
_CP_CHUNKEDPREFILL_COMM_STREAM = None
_ASCEND_CUSTOMOP_IS_REIGISTERED = False
_DEFAULT_BUFFER_SIZE = 200
_MIN_DP_BUFFER_SIZE = 50
_DYNAMIC_EPLB_BUFFER_SIZE = 100
_IS_MOE_MODEL = None
_IS_DRAFTER_MOE_MODEL = None
_IS_VL_MODEL = None
_ENABLE_SP = None
_HAS_LAYER_IDX = None
_HAS_ROPE = None


def is_310p():
    return get_ascend_device_type() == AscendDeviceType._310P


def _mark_op_side_effectful(op: Any) -> None:
    torch.fx.node.has_side_effect(op)
    default_overload = getattr(op, "default", None)
    if default_overload is not None:
        torch.fx.node.has_side_effect(default_overload)


def _ensure_device_print_registered() -> None:
    global _DEVICE_PRINT_OP_REGISTERED

    if _DEVICE_PRINT_OP_REGISTERED:
        return

    if not enable_custom_op():
        raise RuntimeError(
            "device_print requires _C_ascend.device_print ops to be available "
            "when custom ops are enabled in the current Ascend build."
        )

    try:
        # Mark device_print ops side-effectful so FX/Inductor does not DCE or reorder these debug callbacks.
        _mark_op_side_effectful(torch.ops._C_ascend.device_print)
        _mark_op_side_effectful(torch.ops._C_ascend.device_print_tensor)
        _DEVICE_PRINT_OP_REGISTERED = True
    except AttributeError as exc:
        raise RuntimeError(
            "device_print requires _C_ascend.device_print ops to be available "
            "when custom ops are enabled in the current Ascend build."
        ) from exc


def device_print(
    value: torch.Tensor | int | float | bool | str | torch.dtype | torch.device | torch.Size,
) -> None:
    """Print one value from a device callback.

    This helper is intended for debugging. To stay replay-safe under
    ``torch.npu.graph`` capture/replay, the underlying callback payloads are
    retained instead of being reclaimed after the first host callback runs.
    Avoid using it in hot paths or long-running high-frequency loops, otherwise
    there may be memory issues due to too many retained payloads.

    Supported usage:

        >>> from vllm_ascend.utils import device_print
        >>> device_print(x)
        >>> device_print("already formatted text")
        >>> device_print(7)

    Unsupported usage:

        >>> device_print("x =", x)
        >>> device_print("This is ", x, "and this is ", y)

    If you need device-time tensor values, pass the tensor itself. If you need
    text, pass one final string that is already formatted.

    Tensor values are copied to host on the current stream before the callback
    prints them, so printing remains ordered with respect to the surrounding
    device work.

    DO NOT FORMAT A DEVICE TENSOR INTO A STRING YOURSELF AND THEN PRINT, for example:

        >>> device_print(f"x = {x}")
        >>> device_print("x = " + str(x))
    """
    _ensure_device_print_registered()

    if isinstance(value, torch.Tensor):
        torch.ops._C_ascend.device_print_tensor(value)
    elif isinstance(value, (str, int, float, bool, torch.dtype, torch.device, torch.Size)):
        torch.ops._C_ascend.device_print(str(value))
    else:
        raise TypeError(
            f"Unsupported device_print value type: {type(value)!r}. "
            "Use exactly one argument: device_print(tensor), device_print('formatted text')."
        )


def _should_trans_nz(weight: torch.Tensor) -> bool:
    # FP32 cannot use NZ.
    if weight.dtype == torch.float32:
        return False

    # 310P always converts to NZ.
    if is_310p():
        return True

    # NZ is disabled on non-310P.
    if not envs_ascend.VLLM_ASCEND_ENABLE_NZ:
        return False

    # BF16/FP16 convert only when enable_nz == 2.
    if weight.dtype in {torch.bfloat16, torch.float16}:
        return envs_ascend.VLLM_ASCEND_ENABLE_NZ == 2

    # Quantized or other supported dtypes convert by default.
    return True


# NZ conversion policy:
# - 310P: always convert supported weights to FRACTAL_NZ
# - non-310P: follow VLLM_ASCEND_ENABLE_NZ
# - FP32: never convert
def maybe_trans_nz(weight: torch.Tensor) -> torch.Tensor:
    if not _should_trans_nz(weight):
        return weight
    return torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)


def _round_up(x: int, align: int):
    # round up x to align, for example, if align is 16, x will be rounded up to 16, 32, 48, etc.
    # input: 15, 16 -> output: 16
    # input: 17, 16 -> output: 32
    # input: 30, 16 -> output: 32
    # input: 33, 16 -> output: 48
    # ...
    return (x + align - 1) // align * align


def _custom_pad(x, pad_dims):
    # pad the input tensor to the shape of pad_dims
    # input: (13, 30), pad_dims: [0, 2, 0, 3]
    # output: (16, 32)
    return torch.nn.functional.pad(x, pad_dims)


def _custom_reshape(x, target_shape):
    # reshape the input tensor to the shape of target_shape
    # input: (16, 32), target_shape: [1, 16, 2, 16]
    # output: (1, 16, 2, 16)
    return x.reshape(target_shape)


def _custom_transpose(x, dim1, dim2):
    # transpose the input tensor
    # input: (1, 16, 2, 16), dim1: 1, dim2: 2
    # output: (1, 2, 16, 16)
    return x.transpose(dim1, dim2)


def nd_to_nz_2d(in_tensor: torch.Tensor) -> torch.Tensor:
    # in_tensor: (13, 30)
    aux_dims = [1, 0, 0, 16]
    # aux_dims[1]: 16
    aux_dims[1] = _round_up(in_tensor.size(0), 16)
    # aux_dims[2]: 2
    aux_dims[2] = _round_up(in_tensor.size(1), 16) // 16

    # after: aux_dims: [1, 16, 2, 16]

    pad_dims = [0, 0, 0, 0]
    # pad_dims[1]: 2
    pad_dims[1] = _round_up(in_tensor.size(1), 16) - in_tensor.size(1)
    # pad_dims[3]: 3
    pad_dims[3] = _round_up(in_tensor.size(0), 16) - in_tensor.size(0)

    # after: pad_dims: [0, 2, 0, 3]

    # return: (1, 2, 16, 16)
    return _custom_transpose(_custom_reshape(_custom_pad(in_tensor, pad_dims), aux_dims), 1, 2).contiguous()


def nd_to_nz_spec(mask_tensor: torch.Tensor) -> torch.Tensor:
    num_tokens = mask_tensor.shape[0]
    max_seq_len = mask_tensor.shape[1]

    tokens_pad = (num_tokens + 15) // 16 * 16
    max_seq_len_pad = (max_seq_len + 15) // 16 * 16

    mask_tensor_pad = torch.zeros((1, tokens_pad, max_seq_len_pad), dtype=mask_tensor.dtype, device=mask_tensor.device)
    mask_tensor_pad[0][:num_tokens, :max_seq_len] = mask_tensor
    mask = mask_tensor_pad.reshape((1, tokens_pad, max_seq_len_pad // 16, 16)).permute(0, 2, 1, 3)
    return mask


def aligned_16(tensor: torch.Tensor):
    """Aligned tensor for 310P"""

    # Get the size of the current 0th dimension
    n = tensor.size(0)

    # Calculate the aligned size
    n_aligned = ((n + 15) // 16) * 16

    # If already aligned, return the original tensor
    if n == n_aligned:
        return tensor

    # Create a new tensor with shape (n_aligned, H, W) and fill it with zeros
    new_tensor = torch.zeros(n_aligned, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)

    # Copy the original tensor to the first N positions of the new tensor
    new_tensor[:n] = tensor

    return new_tensor


def enable_custom_op():
    """
    Enable lazy init for vllm_ascend_C to avoid early initialization of CANN's RTS component.
    Ensure that ASCEND_RT_VISIBLE_DEVICES can be dynamically modified before torch.npu.set_device().
    """
    import vllm.envs as envs

    global _CUSTOM_OP_ENABLED

    if _CUSTOM_OP_ENABLED is not None:
        return _CUSTOM_OP_ENABLED

    # There are some customed operators which aren't implemented
    # with batch invariant in vllm-ascend, we need to disable them.
    # FIXME(linfeng): Currently custom op compilation and execution are partially available
    # in ASCEND950 chip, we temporarily disable all custom ops. Please refer to
    # https://github.com/vllm-project/vllm-ascend/issues/7157 for latest update about custom op.
    if envs.VLLM_BATCH_INVARIANT or get_ascend_device_type() == AscendDeviceType.A5:
        _CUSTOM_OP_ENABLED = False
        return _CUSTOM_OP_ENABLED

    try:
        # isort: off
        # register custom ops into torch_library here
        import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401

        # register the meta implementation for custom kernel if necessary
        import vllm_ascend.meta_registration  # type: ignore  # noqa: F401

        # isort: on
        _CUSTOM_OP_ENABLED = True
    except ImportError:
        _CUSTOM_OP_ENABLED = False
        logger.warning("Warning: Failed to register custom ops, all custom ops will be disabled")
    return _CUSTOM_OP_ENABLED


def find_hccl_library() -> str:
    """
    We either use the library file specified by the `HCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libhccl.so` can be
    found by `ctypes` automatically.
    """
    so_file = envs_ascend.HCCL_SO_PATH

    # manually load the hccl library
    if so_file:
        logger.info("Found hccl from environment variable HCCL_SO_PATH=%s", so_file)
    else:
        if torch.version.cann is not None:
            so_file = "libhccl.so"
        else:
            raise ValueError("HCCL only supports Ascend NPU backends.")
        logger.info("Found hccl from library %s", so_file)
    return so_file


def current_stream() -> torch.npu.Stream:
    """
    replace `torch.npu.current_stream()` with `vllm.utils.current_stream()`.
    it turns out that `torch.npu.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.npu.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.npu.current_stream()`.

    """
    global _CURRENT_STREAM
    if _CURRENT_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _CURRENT_STREAM = torch.npu.current_stream()
    return _CURRENT_STREAM


def prefetch_stream() -> torch.npu.Stream:
    global _PREFETCH_STREAM
    if _PREFETCH_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _PREFETCH_STREAM = torch_npu.npu.Stream()
    return _PREFETCH_STREAM


def set_weight_prefetch_method(weight_prefetch_config: WeightPrefetchConfig):
    global _WEIGHT_PREFETCH_METHOD
    if _WEIGHT_PREFETCH_METHOD is None:
        from vllm_ascend.ops.weight_prefetch import WeightPrefetchMethod

        _WEIGHT_PREFETCH_METHOD = WeightPrefetchMethod(weight_prefetch_config)
    return _WEIGHT_PREFETCH_METHOD


def get_weight_prefetch_method():
    return _WEIGHT_PREFETCH_METHOD


def global_stream() -> torch.npu.Stream:
    global _GLOBAL_STREAM
    if _GLOBAL_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _GLOBAL_STREAM = torch_npu.npu.Stream()
    return _GLOBAL_STREAM


def shared_experts_calculation_stream() -> torch.npu.Stream:
    global _SHARED_EXPERTS_CALCULATION_STREAM
    if _SHARED_EXPERTS_CALCULATION_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _SHARED_EXPERTS_CALCULATION_STREAM = torch_npu.npu.Stream()
    return _SHARED_EXPERTS_CALCULATION_STREAM


def cp_chunkedprefill_comm_stream() -> torch.npu.Stream:
    global _CP_CHUNKEDPREFILL_COMM_STREAM
    if _CP_CHUNKEDPREFILL_COMM_STREAM is None:
        _CP_CHUNKEDPREFILL_COMM_STREAM = torch_npu.npu.Stream()
    return _CP_CHUNKEDPREFILL_COMM_STREAM


def adapt_patch(is_global_patch: bool = False):
    if is_global_patch:
        from vllm_ascend.patch import platform  # noqa: F401
    else:
        from vllm_ascend.patch import worker  # noqa: F401


@functools.cache
def vllm_version_is(target_vllm_version: str):
    if envs_ascend.VLLM_VERSION is not None:
        vllm_version = envs_ascend.VLLM_VERSION
    else:
        import vllm

        vllm_version = vllm.__version__
    try:
        return Version(vllm_version) == Version(target_vllm_version)
    except InvalidVersion:
        raise ValueError(
            f"Invalid vllm version {vllm_version} found. A dev version of vllm "
            "is installed probably. Set the environment variable VLLM_VERSION "
            "to control it by hand. And please make sure the value follows the "
            "format of x.y.z."
        )


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
def update_cudagraph_capture_sizes(vllm_config: VllmConfig, cudagraph_capture_sizes: list[int]):
    valid_max_size = cudagraph_capture_sizes[-1] if cudagraph_capture_sizes else 0
    if (
        vllm_config.compilation_config.max_cudagraph_capture_size is not None
        and vllm_config.compilation_config.max_cudagraph_capture_size != valid_max_size
    ):
        if vllm_config.compilation_config.cudagraph_capture_sizes is not None:
            raise ValueError(
                "customized max_cudagraph_capture_size"
                f"(={vllm_config.compilation_config.max_cudagraph_capture_size}) "
                "should be consistent with the max value of "
                f"cudagraph_capture_sizes(={valid_max_size})"
            )
        logger.warning(
            "Truncating max_cudagraph_capture_size to %d",
            valid_max_size,
        )

    vllm_config.compilation_config.max_cudagraph_capture_size = valid_max_size

    if vllm_config.compilation_config.cudagraph_capture_sizes is not None and len(cudagraph_capture_sizes) < len(
        vllm_config.compilation_config.cudagraph_capture_sizes
    ):
        logger.warning(
            ("cudagraph_capture_sizes specified in compilation_config %s is overridden by config %s"),
            vllm_config.compilation_config.cudagraph_capture_sizes,
            cudagraph_capture_sizes,
        )
    vllm_config.compilation_config.cudagraph_capture_sizes = cudagraph_capture_sizes
    vllm_config.compilation_config.post_init_cudagraph_sizes()


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
    original_sizes, compilation_config.cudagraph_capture_sizes = compilation_config.cudagraph_capture_sizes, None

    # Calculate parallel configuration factor
    if not vllm_config.model_config:
        logger.warning(
            "Got empty model config. This typically occurs when an empty vllm_config is "
            "initialized (e.g., in unit tests), where config updates are intentionally skipped."
        )

        return
    hf_config = vllm_config.model_config.hf_text_config
    if hasattr(hf_config, "num_hidden_layers"):
        num_hidden_layers = hf_config.num_hidden_layers
    else:
        num_hidden_layers = get_max_hidden_layers(hf_config)
    parallel_config = vllm_config.parallel_config

    # Calculate maximum supported batch sizes considering model architecture
    resources_per_graph = num_hidden_layers + 1
    # For suffix decoding, use the suffix path when no draft_model_config is provided.
    if (spec := vllm_config.speculative_config) and (draft := spec.draft_model_config):
        # Use get_total_num_hidden_layers() to correctly handle MTP models,
        # which store layer count in num_nextn_predict_layers or
        # mtp_num_hidden_layers (for Qwen3.5) instead of num_hidden_layers.
        resources_per_graph += draft.get_total_num_hidden_layers() + 1

    # TODO: Find out whether we need to take into account the pp_size
    num_comm_groups = sum(
        size > 1
        for size in [
            parallel_config.data_parallel_size,
            parallel_config.tensor_parallel_size,
        ]
    )

    if os.getenv("HCCL_OP_EXPANSION_MODE") == "AIV":
        # TODO: Find out whether we need to take into account the pp_size
        parallel_factor = (
            1
            + num_comm_groups
            + int(parallel_config.enable_expert_parallel)
            + int(vllm_config.additional_config.get("multistream_overlap_shared_expert", False))
        )
        if is_moe_model(vllm_config):
            parallel_factor += parallel_config.data_parallel_size > 1
        else:
            # When AIV mode is enabled, the allreduce operator of the dense
            # layer model will occupy additional streams, which are buffered here.
            MAX_CAPTURE_SIZE = MAX_CAPTURE_SIZE - parallel_factor * resources_per_graph

        # Calculate maximum supported batch sizes considering model architecture on the A2 Hardware Device
        # Assume the following case:
        # MAX_CAPTURE_SIZE = 1920, num_hidden_layers = 48, data_parallel_size is 1, tensor_parallel_size is 4,
        # According to the formula, max_num_batch_sizes = math.floor(1920 / (48 + 1) / 2) = 19
        max_num_batch_sizes = math.floor(MAX_CAPTURE_SIZE / resources_per_graph / parallel_factor)
        logger.info("Calculated maximum supported batch sizes for ACL graph: %s", max_num_batch_sizes)
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
        # This approach may additionally allocate up to rank_size (max 16) - 1 streams per collective communication
        # domain on the device (worst case).
        # Using the default collective communication unfolding method on A3 will lead to a significant reduction
        # in the maximum supported sizes.
        # Therefore, the calculation formula has been modified as follows:
        # Assume the following case:
        # MAX_CAPTURE_SIZE = 1920, num_hidden_layers = 48, data_parallel_size is 1, tensor_parallel_size is 4,
        # According to the formula, max_num_batch_sizes = math.floor((1920 - 1 * 40) / (48 + 1) / (1 + 1 * 2)) = 12
        max_num_batch_sizes = math.floor(
            (MAX_CAPTURE_SIZE - num_comm_groups * 40) / resources_per_graph / (1 + num_comm_groups * 2)
        )
        logger.info("Calculated maximum supported batch sizes for ACL graph: %s", max_num_batch_sizes)
        logger.warning(
            "Currently, communication is performed using FFTS+ method, which reduces "
            "the number of available streams and, as a result, limits the range of runtime "
            "shapes that can be handled. To both improve communication performance and "
            "increase the number of supported shapes, set HCCL_OP_EXPANSION_MODE=AIV."
        )

    arch_name = vllm_config.model_config.architecture

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
            arch_name,
            num_hidden_layers,
            len(original_sizes),
            len(
                compilation_config.cudagraph_capture_sizes  # type: ignore[arg-type]
            ),
        )
    else:
        # No adjustment needed
        compilation_config.cudagraph_capture_sizes = original_sizes
        logger.info(
            "No adjustment needed for ACL graph batch sizes: %s model (layers: %d) with %d sizes",
            arch_name,
            num_hidden_layers,
            len(original_sizes),
        )


# TODO(wxy): Move to ops module
def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))


def register_ascend_customop(vllm_config: VllmConfig | None = None):
    """Register Ascend CustomOP

    NOTE: if the register branch requires model type, please use `vllm.config.get_current_vllm_config`,
    and ensure this will execute after model config is initilazed.
    """
    global _ASCEND_CUSTOMOP_IS_REIGISTERED
    if _ASCEND_CUSTOMOP_IS_REIGISTERED:
        return
    from vllm.model_executor.custom_op import CustomOp

    from vllm_ascend.ops.activation import AscendQuickGELU, AscendSiluAndMul
    from vllm_ascend.ops.conv import AscendConv3dLayer
    from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE, AscendSharedFusedMoE
    from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
    from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm, AscendRMSNormGated
    from vllm_ascend.ops.linear import (
        AscendColumnParallelLinear,
        AscendMergedColumnParallelLinear,
        AscendQKVParallelLinear,
        AscendReplicatedLinear,
        AscendRowParallelLinear,
    )
    from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention
    from vllm_ascend.ops.mm_encoder_attention import AscendMMEncoderAttention
    from vllm_ascend.ops.qwen2_decoder import AscendCustomQwen2Decoder
    from vllm_ascend.ops.rel_pos_attention import AscendRelPosAttention
    from vllm_ascend.ops.rotary_embedding import (
        AscendApplyRotaryEmb,
        AscendDeepseekScalingRotaryEmbedding,
        AscendMRotaryEmbedding,
        AscendRotaryEmbedding,
        AscendYaRNRotaryEmbedding,
    )
    from vllm_ascend.ops.vocab_parallel_embedding import (
        AscendLogitsProcessor,
        AscendParallelLMHead,
        AscendVocabParallelEmbedding,
    )

    global REGISTERED_ASCEND_OPS
    REGISTERED_ASCEND_OPS = {
        "QuickGELU": AscendQuickGELU,
        "SiluAndMul": AscendSiluAndMul,
        "RotaryEmbedding": AscendRotaryEmbedding,
        "MRotaryEmbedding": AscendMRotaryEmbedding,
        "ColumnParallelLinear": AscendColumnParallelLinear,
        "RowParallelLinear": AscendRowParallelLinear,
        "YaRNScalingRotaryEmbedding": AscendYaRNRotaryEmbedding,
        "MergedColumnParallelLinear": AscendMergedColumnParallelLinear,
        "QKVParallelLinear": AscendQKVParallelLinear,
        "ReplicatedLinear": AscendReplicatedLinear,
        "DeepseekScalingRotaryEmbedding": AscendDeepseekScalingRotaryEmbedding,
        "VocabParallelEmbedding": AscendVocabParallelEmbedding,
        "ParallelLMHead": AscendParallelLMHead,
        "LogitsProcessor": AscendLogitsProcessor,
        "RMSNorm": AscendRMSNorm,
        "GemmaRMSNorm": AscendGemmaRMSNorm,
        "FusedMoE": AscendFusedMoE,
        "SharedFusedMoE": AscendSharedFusedMoE,
        "MultiHeadLatentAttentionWrapper": AscendMultiHeadLatentAttention,
        "MMEncoderAttention": AscendMMEncoderAttention,
        "ApplyRotaryEmb": AscendApplyRotaryEmb,
        "RMSNormGated": AscendRMSNormGated,
        "Conv3dLayer": AscendConv3dLayer,
        "RelPosAttention": AscendRelPosAttention,
        "CustomQwen2Decoder": AscendCustomQwen2Decoder,
        "GatedDeltaNetAttention": AscendGatedDeltaNetAttention,
    }

    # 310P: override selected ops with 310P implementations (keep minimal changes outside _310p)
    if is_310p():
        from vllm_ascend._310p.fused_moe.fused_moe import AscendFusedMoE310, AscendSharedFusedMoE310
        from vllm_ascend._310p.ops.activation import AscendSiluAndMul310
        from vllm_ascend._310p.ops.fla.gdn_310 import AscendGatedDeltaNetAttention310
        from vllm_ascend._310p.ops.layernorm import (
            AscendGemmaRMSNorm310,
            AscendRMSNorm310,
            AscendRMSNormGated310,
        )
        from vllm_ascend._310p.ops.mm_encoder_attention import AscendMMEncoderAttention310
        from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310
        from vllm_ascend._310p.ops.vocab_parallel_embedding import (
            AscendParallelLMHead310,
            AscendVocabParallelEmbedding310,
        )

        REGISTERED_ASCEND_OPS.update(
            {
                "SiluAndMul": AscendSiluAndMul310,
                "RotaryEmbedding": AscendRotaryEmbedding310,
                "RMSNorm": AscendRMSNorm310,
                "GemmaRMSNorm": AscendGemmaRMSNorm310,
                "RMSNormGated": AscendRMSNormGated310,
                "FusedMoE": AscendFusedMoE310,
                "SharedFusedMoE": AscendSharedFusedMoE310,
                "ParallelLMHead": AscendParallelLMHead310,
                "VocabParallelEmbedding": AscendVocabParallelEmbedding310,
                "MMEncoderAttention": AscendMMEncoderAttention310,
                "GatedDeltaNetAttention": AscendGatedDeltaNetAttention310,
            }
        )

        REGISTERED_ASCEND_OPS.pop("MRotaryEmbedding", None)

    for name, op_cls in REGISTERED_ASCEND_OPS.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)

    # NOTE: Keep this at last to ensure all custom actions are registered
    _ASCEND_CUSTOMOP_IS_REIGISTERED = True


class AscendDeviceType(Enum):
    A2 = 0
    A3 = 1
    _310P = 2
    A5 = 3


_ascend_device_type = None


def _init_ascend_device_type():
    global _ascend_device_type
    from vllm_ascend import _build_info  # type: ignore

    _ascend_device_type = AscendDeviceType[_build_info.__device_type__]


def check_ascend_device_type():
    global _ascend_device_type
    if _ascend_device_type is None:
        _init_ascend_device_type()

    soc_version = torch_npu.npu.get_soc_version()
    if 220 <= soc_version <= 225:
        cur_device_type = AscendDeviceType.A2
    elif 250 <= soc_version <= 255:
        cur_device_type = AscendDeviceType.A3
    elif 200 <= soc_version <= 205:
        cur_device_type = AscendDeviceType._310P
    elif soc_version == 260:
        cur_device_type = AscendDeviceType.A5
    else:
        raise RuntimeError(f"Can not support soc_version: {soc_version}.")

    assert _ascend_device_type == cur_device_type, (
        f"Current device type: {cur_device_type} does not match the installed version's device type: "
        f"{_ascend_device_type}, please check your installation package."
    )


def get_ascend_device_type():
    global _ascend_device_type
    if _ascend_device_type is None:
        _init_ascend_device_type()
    return _ascend_device_type


def lmhead_tp_enable() -> bool:
    return get_ascend_config().finegrained_tp_config.lmhead_tensor_parallel_size > 0


def embedding_tp_enable() -> bool:
    return get_ascend_config().finegrained_tp_config.embedding_tensor_parallel_size > 0


def oproj_tp_enable() -> bool:
    return get_ascend_config().finegrained_tp_config.oproj_tensor_parallel_size > 0


def mlp_tp_enable() -> bool:
    return get_ascend_config().finegrained_tp_config.mlp_tensor_parallel_size > 0


def matmul_allreduce_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE


def enable_sp_by_pass():
    return get_ascend_config().enable_sp_by_pass


def enable_sp(vllm_config=None, enable_shared_expert_dp: bool = False) -> bool:
    global _ENABLE_SP
    if _ENABLE_SP is None:
        if vllm_config is None:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        _ENABLE_SP = (
            envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1
            # Flash comm 1 should be enabled by env VLLM_ASCEND_ENABLE_FLASHCOMM1
            # We retain the env VLLM_ASCEND_ENABLE_FLASHCOMM here for backward compatibility.
            or bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", "0")))
        )

        if not _ENABLE_SP and enable_shared_expert_dp:
            _ENABLE_SP = True
            logger.info("shared_expert_dp requires enable_sp = True. has set enable_sp to True")

    return _ENABLE_SP


# TODO remove it after vllm has this func
def shared_expert_dp_enabled() -> bool:
    return get_ascend_config().enable_shared_expert_dp or enable_sp() or enable_sp_by_pass()


def prefill_context_parallel_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL


def is_moe_model(vllm_config: VllmConfig):
    """Checks if the model is a MoE model by config"""
    global _IS_MOE_MODEL
    if _IS_MOE_MODEL is None:
        model_configs = vllm_config.model_config.hf_text_config.to_dict()
        _IS_MOE_MODEL = _is_contain_expert(model_configs)
    return _IS_MOE_MODEL


def is_drafter_moe_model(vllm_config: VllmConfig):
    """Checks if the drafter model is a MoE model by config"""
    global _IS_DRAFTER_MOE_MODEL
    if _IS_DRAFTER_MOE_MODEL is None:
        model_configs = vllm_config.speculative_config.draft_model_config.hf_text_config.to_dict()
        _IS_DRAFTER_MOE_MODEL = _is_contain_expert(model_configs)
    return _IS_DRAFTER_MOE_MODEL


def speculative_enable_dispatch_gmm_combine_decode(vllm_config: VllmConfig) -> bool:
    """When draft contains MOE Arch and non-w8a8, disable dispatch_gmm_combine_decode."""
    if vllm_config.speculative_config is None:
        return True
    speculative_method = getattr(vllm_config.speculative_config, "method", None)
    if speculative_method in [None, "ngram", "suffix"]:
        return True
    if speculative_method in ["eagle", "eagle3"]:
        if is_drafter_moe_model(vllm_config):
            draft_model_config = vllm_config.speculative_config.draft_model_config
            hf_text_config = draft_model_config.hf_text_config
            quant_type = getattr(hf_text_config, "moe_quantize", None)
            if quant_type is None:
                quant_type = getattr(hf_text_config, "quantize", None)
            return quant_type == "w8a8_dynamic"
        else:
            return True
    if speculative_method == "mtp":
        mtp_quant_type = getattr(vllm_config.model_config.hf_text_config, "mtp_quantize", None)
        return mtp_quant_type == "w8a8_dynamic"
    return False


def _is_contain_expert(config: Any):
    if isinstance(config, dict):
        for k, v in config.items():
            if "expert" in str(k):
                return True
            if _is_contain_expert(v):
                return True
    return False


def is_vl_model(vllm_config: VllmConfig = None):
    """Checks if the model is a VL model by config.

    Uses the same criterion as vllm itself (model_config.py): a model is
    multimodal when its top-level hf_config differs from its hf_text_config
    (i.e. there is a separate vision sub-config).  The legacy key-name checks
    are kept as fallbacks for configs that override get_text_config() to return
    self (rare but possible).
    """
    global _IS_VL_MODEL
    if vllm_config is None:
        from vllm.config import get_current_vllm_config_or_none

        vllm_config = get_current_vllm_config_or_none()
    if _IS_VL_MODEL is None and vllm_config and vllm_config.model_config:
        model_config = vllm_config.model_config
        # Primary: vllm's own VL detection — hf_config is the top-level
        # (multimodal) config; hf_text_config is the language-model sub-config.
        # They are the same object for pure-text models.
        if model_config.hf_config is not model_config.hf_text_config:
            _IS_VL_MODEL = True
        else:
            # Fallback: check well-known config keys
            hf_config = model_config.hf_config.to_dict()
            if "thinker_config" in hf_config or "vision_config" in hf_config:
                _IS_VL_MODEL = True
            else:
                _IS_VL_MODEL = False
    return _IS_VL_MODEL


def has_rope(vllm_config: VllmConfig):
    """Checks if the model uses rope."""
    global _HAS_ROPE
    if _HAS_ROPE is None and vllm_config and vllm_config.model_config:
        hf_config = vllm_config.model_config.hf_text_config.to_dict()
        _HAS_ROPE = "rope_parameters" in hf_config
    return _HAS_ROPE


def weak_ref_tensor(tensor: Any) -> Any:
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    """
    if isinstance(tensor, torch.Tensor):
        return torch_npu._C._weak_ref_tensor(tensor)
    else:
        return tensor


def weak_ref_tensors(
    tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
) -> torch.Tensor | list[Any] | tuple[Any] | Any:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.

    This function should be used in the following scenario:
    When a tensor is created during graph capture, and it's held by a method
    that's not part of the graph, we don't really need to store it, but we
    **do need** its buffer pointer. If we don't handle this, it cannot
    be garbage collected, leading to a memory leak. To avoid this,
    we should create a weak reference to the tensor.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    # For IntermediateTensors used in pipeline parallelism
    if isinstance(tensors, IntermediateTensors):
        ret = IntermediateTensors({key: weak_ref_tensor(val) for key, val in tensors.tensors.items()})
        return ret
    raise ValueError("Invalid type for tensors")


def npu_stream_switch(target_stream: torch.npu.Stream, *, enabled: bool = True):
    """
    Switch to the target stream if enabled is True.
    Otherwise, do nothing.
    """
    if not enabled:
        return nullcontext()
    assert target_stream is not None
    return torch.npu.stream(target_stream)


def create_hccl_pg_options(group_name: str):
    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    hccl_config = get_hccl_config_for_pg_options(group_name)
    if hccl_config is not None:
        options.hccl_config = hccl_config
    return options


def get_hccl_config_for_pg_options(group_name: str) -> dict | None:
    """
    Get HCCL process group options for the given communication group name.

    Args:
        group_name: Name of the communication group

    Returns:
        HCCL pg_options or None for mc2 group
    """
    # FIXME: Current mc2 operators only perform communication space partitioning
    # based on HCCL_BUFFSIZE configuration. Using pg_options with mc2 group would
    # result in memory misalignment problems.
    if group_name and "mc2" in group_name:
        return None
    hccl_config_map = {
        "dp": {"hccl_buffer_size": calculate_dp_buffer_size()},
        "dynamic_eplb": {"hccl_buffer_size": _DYNAMIC_EPLB_BUFFER_SIZE},
    }
    return hccl_config_map.get(group_name, get_default_buffer_config())


def get_default_buffer_config() -> dict:
    return {"hccl_buffer_size": _DEFAULT_BUFFER_SIZE}


def calculate_dp_buffer_size() -> int:
    """
    formula of dp buffer size:
    dp_size + 1 (flags: with_prefill)
    """
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    dp_size = vllm_config.parallel_config.data_parallel_size
    int32_size = torch.iinfo(torch.int32).bits // 8
    dp_buffer_size = math.ceil((dp_size + 1) * int32_size / (1024 * 1024))
    return max(dp_buffer_size, _MIN_DP_BUFFER_SIZE)


# Currently, when in A2, setting the environment variables HCCL_INTRA_PCIE_ENABLE=1
# and HCCL_INTRA_ROCE_ENABLE=0 can reduce cross-machine communication traffic and
# significantly improve communication performance of MC2 ops dispatch/combine.
def is_hierarchical_communication_enabled():
    return (
        os.getenv("HCCL_INTRA_ROCE_ENABLE", "") == "0" and os.getenv("HCCL_INTRA_PCIE_ENABLE", "") == "1"
    ) or get_ascend_config().enable_mc2_hierarchy_comm


def should_skip_allreduce_across_dp_group(vllm_config, is_draft_model: bool = False) -> bool:
    """Decide whether to skip the all-reduce across the DP group.

    Skipping is applicable for all dense models and for moe models only on ranks
    that act as KV consumers. We skip the DP all-reduce when either:
    - Both the prefill and decode communication methods are MC2 (or FUSED_MC2), or
    - Decode requires MC2 and ascend_config.recompute_scheduler_enable is True.

    Skipping means each rank may have a different number of tokens, so MC2 needs
    a non-zero global_bs and must NOT receive mc2_mask.

    Returns False when hierarchy comm is enabled because hierarchy requires
    global_bs=0 (uniform tokens), which is incompatible with skipping allreduce.
    """
    if is_hierarchical_communication_enabled():
        return False

    # For dense models, since we don't actually need dp communication, we simply skip it.
    # This usually happens when main model is moe while eagle draft model is dense.
    is_context_moe_model = is_drafter_moe_model(vllm_config) if is_draft_model else is_moe_model(vllm_config)
    if not is_context_moe_model:
        return True

    # Only applicable to MoE models on KV consumer ranks.
    is_kv_consumer = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_consumer
    if not is_kv_consumer:
        return False

    from vllm_ascend.ascend_forward_context import select_moe_comm_method
    from vllm_ascend.ops.fused_moe.moe_comm_method import MoECommType

    def needs_mc2(n: int) -> bool:
        return select_moe_comm_method(n, vllm_config) in {MoECommType.MC2, MoECommType.FUSED_MC2}

    compilation_config = vllm_config.compilation_config
    scheduler_config = vllm_config.scheduler_config
    speculative_config = vllm_config.speculative_config
    uniform_decode_query_len = 1 if not speculative_config else 1 + speculative_config.num_speculative_tokens
    decode_max_num_seqs = getattr(scheduler_config, "decode_max_num_seqs", 0)
    max_num_reqs = max(scheduler_config.max_num_seqs, decode_max_num_seqs)

    # Determine whether decode must use MC2. Use max cudagraph capture size
    # if available, otherwise use the maximal uniform decode token count.
    if compilation_config.cudagraph_capture_sizes:
        potential_max_tokens = compilation_config.max_cudagraph_capture_size
    else:
        potential_max_tokens = min(max_num_reqs * uniform_decode_query_len, 512)

    decode_must_use_mc2 = needs_mc2(potential_max_tokens)
    # For prefill, use the scheduler's max_num_batched_tokens for a single batch.
    prefill_must_use_mc2 = needs_mc2(scheduler_config.max_num_batched_tokens)
    # Skip all-reduce if decode requires MC2 and either prefill also
    # requires MC2 or recompute-based scheduler is enabled.
    return decode_must_use_mc2 and (prefill_must_use_mc2 or get_ascend_config().recompute_scheduler_enable)


def has_layer_idx(model_instance: torch.nn.Module) -> bool:
    if model_instance is None:
        return False

    global _HAS_LAYER_IDX
    if _HAS_LAYER_IDX is None:
        _HAS_LAYER_IDX = hasattr(model_instance, "model") and hasattr(model_instance.model, "start_layer")
    return _HAS_LAYER_IDX


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

    logger.info(f"Enable FLASHCOMM2 with flashcomm2_oproj_tensor_parallel_size = {flashcomm2_oproj_tp_size}")

    layer_sharding = ascend_config.layer_sharding or []
    if layer_sharding:
        if layer_sharding == ["o_proj"]:
            logger.info_once("Enable FLASHCOMM2 with o_proj layer sharding for reduced memory consumption.")
        else:
            raise ValueError(
                "FLASHCOMM2 only supports 'o_proj' as the sole layer sharding configuration! "
                f"Found invalid layer_sharding: {layer_sharding}"
            )
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
            f"flashcomm2_oproj_tensor_parallel_size ({flashcomm2_oproj_tp_size}) cannot exceed "
            f"global tensor parallel size ({global_tp_size})"
        )
    if global_tp_size % flashcomm2_oproj_tp_size != 0:
        raise AssertionError(
            f"Global tensor parallel size ({global_tp_size}) must be divisible by "
            f"flashcomm2_oproj_tensor_parallel_size ({flashcomm2_oproj_tp_size})"
        )
    if vllm_config.kv_transfer_config is None:
        logger.warning_once(
            "It is recommended to enable FLASHCOMM2 in P-scenario deployments, enable it in hybrid deployment "
            "may lead to decode performance degradation."
        )
    if vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_consumer:
        raise AssertionError(
            "FLASHCOMM2 primarily targets P-scenario deployments, with additional support "
            "for hybrid deployment scenarios. It is not applicable in D-scenario environments."
        )

    return flashcomm2_oproj_tp_size


def get_flashcomm2_reorgnized_batch_ids(global_tp_size) -> list[list[int]]:
    # Reorganize batch_ids so that, after the all2all and reduce-scatter operation,
    # each batch_id corresponds to the rank_id within the DP domain.
    # For example, when DP = [0, 1, 2, ..., 15] and flashcomm2_oproj_tensor_parallel_size = 2,
    # the reorganized batch_ids will be [[batch0, batch8], [batch1, batch9], ..., [batch7, batch15]].
    flashcomm2_otp_size = get_ascend_config().flashcomm2_oproj_tensor_parallel_size
    num_oproj_tensor_parallel_groups: int = global_tp_size // flashcomm2_otp_size

    reorgnized_batch_ids = []
    for i in range(num_oproj_tensor_parallel_groups):
        ranks = []
        for j in range(flashcomm2_otp_size):
            rank_idx = i + j * num_oproj_tensor_parallel_groups
            ranks.append(rank_idx)
        reorgnized_batch_ids.append(ranks)

    return reorgnized_batch_ids


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

    if model_config.is_hybrid:
        # Hybrid attention+mamba models rely on the model-specific sizing
        # logic rather than the generic platform default.
        return

    if cache_config.block_size != 128:
        if cache_config.enable_prefix_caching or scheduler_config.enable_chunked_prefill:
            logger.info("Block size is set to 128 if prefix cache or chunked prefill is enabled.")
            cache_config.block_size = 128
            return

    ascend_config = get_ascend_config()
    if ascend_config.xlite_graph_config.enabled and cache_config.block_size > 128:
        logger.warning("Setting block size to 128 for xlite compatibility.")
        cache_config.block_size = 128


def dispose_layer(layer: Any):
    for attr_name in dir(layer):
        attr_value = getattr(layer, attr_name)
        if isinstance(attr_value, torch.Tensor):
            dispose_tensor(attr_value)


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
                    f"Expected {vllm_tp}, but got {config_tp}."
                )
        if dp_key in config:
            config_dp = config[dp_key]
            vllm_dp = vllm_config.parallel_config.data_parallel_size
            if config_dp != vllm_dp:
                raise ValueError(
                    f"KV transfer '{name}' config has a conflicting data parallel size. "
                    f"Expected {vllm_dp}, but got {config_dp}."
                )

    if vllm_config.kv_transfer_config.is_kv_producer:
        _check("prefill", vllm_config.kv_transfer_config.get_from_extra_config("prefill", {}))
    if vllm_config.kv_transfer_config.is_kv_consumer:
        _check("decode", vllm_config.kv_transfer_config.get_from_extra_config("decode", {}))


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


# TODO: Temporarily use enable_sp to enable the dsa_cp feature of ds32.
# and subsequent updates will introduce new interfaces. --zzhx1
@lru_cache(maxsize=1)
def enable_dsa_cp() -> bool:
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    is_ds_v32 = hasattr(vllm_config.model_config, "hf_text_config") and hasattr(
        vllm_config.model_config.hf_text_config, "index_topk"
    )
    return bool(is_ds_v32 and enable_sp())


@lru_cache(maxsize=1)
def enable_dsa_cp_with_layer_shard() -> bool:
    if not enable_dsa_cp():
        return False
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    # because the broadcast in layer sharding needs to be overlapped with a heavy compute stream to be
    # effectively hidden, it is enabled only during the prefill stage.
    is_prefill_instance = vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.is_kv_producer
    return is_prefill_instance


@lru_cache(maxsize=1)
def enable_dsa_cp_with_o_proj_tp() -> bool:
    if not enable_dsa_cp():
        return False
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    # if is PD mix stage, using original TP o_proj weight, and also need to
    # full gather for o_proj weight for prefill stage.
    return vllm_config.kv_transfer_config is None


def check_gdn_layer(vllm_config) -> bool:
    """
    gdn layer is marked with `linear_attention`.
    So, if `linear_attention` is detected, we think the model has gdn-attention.
    """
    if not hasattr(vllm_config, "model_config"):
        return False

    model_config = vllm_config.model_config
    if not hasattr(model_config, "hf_config"):
        return False

    hf_config = model_config.hf_config

    # Use `or []` to prevent errors when layer_types is None
    layer_types = getattr(hf_config, "layer_types", None) or []
    if "linear_attention" in layer_types:
        return True

    text_config = getattr(hf_config, "text_config", None)
    if text_config:
        text_layer_types = getattr(text_config, "layer_types", None) or []
        if "linear_attention" in text_layer_types:
            return True

    return False


def get_rope_dim(vllm_config):
    model_config = vllm_config.model_config

    if model_config.use_mla:
        rope_dim = model_config.hf_text_config.qk_rope_head_dim
    else:
        rope_dim = model_config.get_head_size()
        # For models using partial rope like Qwen3-Next.
        if hasattr(model_config.hf_text_config, "partial_rotary_factor"):
            rope_dim = int(rope_dim * model_config.hf_text_config.partial_rotary_factor)
        elif hasattr(model_config.hf_text_config, "rotary_dim"):
            rope_dim = int(model_config.hf_text_config.rotary_dim)

    return rope_dim


def calc_split_factor(num_list: list[int]):
    total = sum(num_list)
    return [total / num for num in num_list]


# NOTE: The last two dimensions of ND are transferred to NZ
def trans_nd_to_nz(cache_tensor: torch.Tensor):
    assert len(cache_tensor.shape) >= 2
    batch = cache_tensor.shape[:-2]
    a, b = cache_tensor.shape[-2], cache_tensor.shape[-1]

    dtype = cache_tensor.dtype
    if dtype == torch.int8:
        a0, b0 = 16, 32
    else:
        a0, b0 = 16, 16

    nz_shape = list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

    # Generate the axis order for the transpose operation.
    offset = len(cache_tensor.shape) - 2
    base = [2, 0, 1, 3]
    array_trans = [i for i in range(offset)] + [i + offset for i in base]
    # Perform shape transformation and transpose operation.
    *_, n1, m1, m0, n0 = nz_shape
    cache_tensor = cache_tensor.reshape(nz_shape[:-4] + [m1, m0, n1, n0])
    cache_tensor = cache_tensor.permute(*array_trans)
    return cache_tensor


def parse_layer_idx(prefix: str) -> int | None:
    """Extract the layer index from a module prefix string like 'model.layers.0.self_attn'."""
    match = re.search(r"layers\.(\d+)", prefix)
    return int(match.group(1)) if match else None


def kv_cache_spec_uses_sparse_c8(kv_cache_spec) -> bool:
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    return isinstance(kv_cache_spec, MLAAttentionSpec) and bool(getattr(kv_cache_spec, "cache_sparse_c8", False))
