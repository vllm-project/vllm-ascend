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

import atexit
import functools
import math
import os
from contextlib import contextmanager
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch_npu  # noqa: F401  # noqa: F401
from packaging.version import InvalidVersion, Version
from torch_npu.npu.streams import Event
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

# NOTE: Currently, we can only capture 1800 graphs at most,
# due to the limitation of ACL graph. This number is bounded by
# the number of streams, which is 2048, we save 248 streams
# as a buffer.
# Maximum number of graphs that can be captured by ACL Graph
# TODO: Find out whether we need to solve allreduce function
MAX_CAPTURE_SIZE = 1800

ASCEND_QUANTIZATION_METHOD = "ascend"
SOC_VERSION_INFERENCE_SERIES = ["Ascend310P3"]
REGISTERED_ASCEND_OPS = {}

ACL_FORMAT_FRACTAL_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29

_CUSTOM_OP_ENABLED = None
_IS_310P = None
_SLEEP_MODE_ENABLED = None
_CURRENT_STREAM = None
_ASCEND_CUSTOMOP_IS_REIGISTERED = False


def is_310p():
    global _IS_310P
    if _IS_310P is None:
        from vllm_ascend import _build_info  # type: ignore
        _IS_310P = _build_info.__soc_version__.lower().startswith("ascend310p")
    return _IS_310P


def sleep_mode_enabled():
    global _SLEEP_MODE_ENABLED
    if _SLEEP_MODE_ENABLED is None:
        from vllm_ascend import _build_info  # type: ignore
        _SLEEP_MODE_ENABLED = _build_info.__sleep_mode_enabled__
    return _SLEEP_MODE_ENABLED


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
    return _custom_transpose(
        _custom_reshape(_custom_pad(in_tensor, pad_dims), aux_dims), 1,
        2).contiguous()


def nd_to_nz_spec(mask_tensor: torch.Tensor) -> torch.Tensor:
    num_tokens = mask_tensor.shape[0]
    max_seq_len = mask_tensor.shape[1]

    tokens_pad = (num_tokens + 15) // 16 * 16
    max_seq_len_pad = (max_seq_len + 15) // 16 * 16

    mask_tensor_pad = \
        torch.zeros((1, tokens_pad, max_seq_len_pad), dtype=mask_tensor.dtype, device=mask_tensor.device)
    mask_tensor_pad[0][:num_tokens, :max_seq_len] = mask_tensor
    mask = mask_tensor_pad.reshape(
        (1, tokens_pad, max_seq_len_pad // 16, 16)).permute(0, 2, 1, 3)
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
    new_tensor = torch.zeros(n_aligned,
                             *tensor.shape[1:],
                             dtype=tensor.dtype,
                             device=tensor.device)

    # Copy the original tensor to the first N positions of the new tensor
    new_tensor[:n] = tensor

    return new_tensor


def try_register_lib(lib_name: str, lib_info: str = ""):
    import importlib
    import importlib.util
    try:
        module_spec = importlib.util.find_spec(lib_name)
        if module_spec is not None:
            importlib.import_module(lib_name)
            if lib_info:
                logger.info(lib_info)
    except Exception:
        pass


def enable_custom_op():
    """
    Enable lazy init for vllm_ascend_C to avoid early initialization of CANN's RTS component.
    Ensure that ASCEND_RT_VISIBLE_DEVICES can be dynamically modified before torch.npu.set_device().
    """
    global _CUSTOM_OP_ENABLED
    if _CUSTOM_OP_ENABLED is not None:
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
        logger.warning(
            "Warning: Failed to register custom ops, all custom ops will be disabled"
        )
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
        logger.info("Found hccl from environment variable HCCL_SO_PATH=%s",
                    so_file)
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
            "format of x.y.z.")


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


def update_aclgraph_sizes(vllm_config: VllmConfig) -> None:
    """Update ACL graph capture sizes based on hardware limitations"""
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
            parallel_config.enable_expert_parallel)
        if is_moe_model(vllm_config):
            parallel_factor += (parallel_config.data_parallel_size > 1)
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
        compilation_config.init_with_cudagraph_sizes(sampled_sizes)

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


# TODO(wxy): Move to ops module
def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0, ), device=x.device, dtype=x.dtype))


class ProfileExecuteDuration:
    _instance = None
    _observations: List[Tuple[str, Event, Event]] = []
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                atexit.register(cls._instance.destroy)
            return cls._instance

    def destroy(self):
        with self._lock:
            self._observations.clear()

    @contextmanager
    def capture_async(self, duration_tag: str):
        if not envs_ascend.VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE:
            yield
            return

        observe_start = Event(enable_timing=True)
        observe_start.record()
        try:
            yield
        finally:
            observe_end = Event(enable_timing=True)
            observe_end.record()
            with self._lock:
                self._observations.append(
                    (duration_tag, observe_start, observe_end))

    def pop_captured_sync(self) -> dict:
        """Pop and synchronize all events in the observation list"""
        durations: dict[str, float] = {}
        if not envs_ascend.VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE:
            return durations

        while self._observations:
            with self._lock:
                tag, observe_start, observe_end = self._observations.pop()
            observe_end.synchronize()
            durations[tag] = observe_start.elapsed_time(observe_end)

        return durations


# TODO(wxy): Move to ops module
def npu_prefetch(input: torch.Tensor,
                 dependency: torch.Tensor,
                 max_size: int = 0,
                 *,
                 enabled: bool = True):
    if not enabled:
        return
    input_size = input.element_size() * input.numel()
    if max_size <= 0 or max_size > input_size:
        max_size = input_size
    torch_npu.npu_prefetch(input, dependency, max_size)


# TODO(ttanzhiqiang): rm_router_logits
# dp>1 will trigger
# In theory, this solution is only applicable to AllGather and AllGatherEP, because in the dp scenario, the previous operation was gate + two communications, and now it is changed to one communication + gate operation, which can save some communication time. In theory, all moe AllGather and AllGatherEP solutions can follow this logic, but now other moe models (qwen3-235b) dp solutions are not adjusted, so use the switch to control it to prevent code errors.
def get_rm_router_logits_state(ep_size: int, dp_size: int,
                               is_deepseek_v3_r1: bool):
    # the fusion operator torch_npu.npu_grouped_matmul_finalize_routing called by allgather ep
    # only supports deepseek v3/r1
    if dp_size > 1:
        if (envs_ascend.VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP and ep_size > 1
                and is_deepseek_v3_r1):
            return True
        elif ep_size == 1 and is_deepseek_v3_r1:
            return True
    return False


# TODO(ttanzhiqiang): all_reduce merge
# When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
# Currently, all_reduce_merge is enabled by default in the AllGather, AllGatherEP and NaiveMulticast scenarios of the deepseek model.
def get_all_reduce_merge_state(ep_size: int, is_deepseek_v3_r1: bool):
    # the fusion operator torch_npu.npu_grouped_matmul_finalize_routing called by allgather ep
    # only supports deepseek v3/r1
    if (envs_ascend.VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP and ep_size > 1
            and is_deepseek_v3_r1):
        return True
    elif ep_size == 1 and is_deepseek_v3_r1:
        return True
    return False


def register_ascend_customop(vllm_config: Optional[VllmConfig] = None):
    """Register Ascend CustomOP

    NOTE: if the register branch requires model type, please use `vllm.config.get_current_vllm_config`,
    and ensure this will execute after model config is initilazed.
    """
    global _ASCEND_CUSTOMOP_IS_REIGISTERED
    if _ASCEND_CUSTOMOP_IS_REIGISTERED:
        return
    from vllm.model_executor.custom_op import CustomOp

    from vllm_ascend.models.layers.mla import AscendMultiHeadLatentAttention
    from vllm_ascend.ops.activation import AscendQuickGELU, AscendSiluAndMul
    from vllm_ascend.ops.common_fused_moe import AscendFusedMoE
    from vllm_ascend.ops.layernorm import AscendQuantRMSNorm, AscendRMSNorm
    from vllm_ascend.ops.linear import (AscendColumnParallelLinear,
                                        AscendMergedColumnParallelLinear,
                                        AscendQKVParallelLinear,
                                        AscendRowParallelLinear)
    from vllm_ascend.ops.rotary_embedding import (
        AscendDeepseekScalingRotaryEmbedding, AscendRotaryEmbedding)
    from vllm_ascend.ops.vocab_parallel_embedding import (
        AscendLogitsProcessor, AscendParallelLMHead,
        AscendVocabParallelEmbedding)

    global REGISTERED_ASCEND_OPS
    REGISTERED_ASCEND_OPS = {
        "QuickGELU": AscendQuickGELU,
        "SiluAndMul": AscendSiluAndMul,
        "RotaryEmbedding": AscendRotaryEmbedding,
        "ColumnParallelLinear": AscendColumnParallelLinear,
        "RowParallelLinear": AscendRowParallelLinear,
        "MergedColumnParallelLinear": AscendMergedColumnParallelLinear,
        "QKVParallelLinear": AscendQKVParallelLinear,
        "DeepseekScalingRotaryEmbedding": AscendDeepseekScalingRotaryEmbedding,
        "VocabParallelEmbedding": AscendVocabParallelEmbedding,
        "ParallelLMHead": AscendParallelLMHead,
        "LogitsProcessor": AscendLogitsProcessor,
        "RMSNorm": AscendRMSNorm,
        "FusedMoE": AscendFusedMoE,
        "MultiHeadLatentAttention": AscendMultiHeadLatentAttention,
    }

    if vllm_config is not None and \
        vllm_config.quant_config is not None and \
        any("norm.bias" in name for name in vllm_config.quant_config.quant_description.keys()):
        REGISTERED_ASCEND_OPS["RMSNorm"] = AscendQuantRMSNorm

    for name, op_cls in REGISTERED_ASCEND_OPS.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)

    # NOTE: Keep this at last to ensure all custom actions are registered
    _ASCEND_CUSTOMOP_IS_REIGISTERED = True


# TODO(zzzzwwjj): Currently there is no clear SOC_VERSION policy for A2 and A3 in CANN.
# So we get the version dynamically. In the future, we should get the version info from _build_info like 310p does.
class AscendSocVersion(Enum):
    A2 = 0
    A3 = 1
    UNDEFINED = 2


_ascend_soc_version = None


def init_ascend_soc_version():
    soc_version = torch_npu.npu.get_soc_version()
    global _ascend_soc_version
    if 220 <= soc_version <= 225:
        _ascend_soc_version = AscendSocVersion.A2
    elif 250 <= soc_version <= 255:
        _ascend_soc_version = AscendSocVersion.A3
    else:
        _ascend_soc_version = AscendSocVersion.UNDEFINED


def get_ascend_soc_version():
    global _ascend_soc_version
    assert _ascend_soc_version is not None
    return _ascend_soc_version


def lmhead_tp_enable() -> bool:
    return get_ascend_config().lmhead_tensor_parallel_size is not None


def oproj_tp_enable() -> bool:
    return get_ascend_config().oproj_tensor_parallel_size is not None


def mlp_tp_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_MLP_OPTIMIZE


def sequence_parallel_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_SP


def context_parallel_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_CP


def matmul_allreduce_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE


def dense_optim_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE


def is_moe_model(vllm_config: VllmConfig):
    config = vllm_config.model_config.hf_config
    return any('experts' in key.lower() for key in config.to_dict())


def weak_ref_tensor(tensor: Any) -> Any:
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.ops._C_ascend.weak_ref_tensor(tensor)
    else:
        return tensor


def weak_ref_tensors(
    tensors: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]]
) -> Union[torch.Tensor, list[Any], tuple[Any], Any]:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError("Invalid type for tensors")
