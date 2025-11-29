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
from contextlib import contextmanager, nullcontext
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch_npu  # noqa: F401
from packaging.version import InvalidVersion, Version
from torch_npu.npu.streams import Event
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

ASCEND_QUANTIZATION_METHOD = "ascend"
COMPRESSED_TENSORS_METHOD = "compressed-tensors"
SOC_VERSION_INFERENCE_SERIES = ["Ascend310P3"]
REGISTERED_ASCEND_OPS = {}

ACL_FORMAT_FRACTAL_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29

_CUSTOM_OP_ENABLED = None
_SLEEP_MODE_ENABLED = None
_CURRENT_STREAM = None
_PREFETCH_STREAM = None
_SHARED_EXPERTS_CALCULATION_STREAM = None
_ASCEND_CUSTOMOP_IS_REIGISTERED = False
_DEFAULT_BUFFER_SIZE = 200
_MIN_DP_BUFFER_SIZE = 50
_IS_MOE_MODEL = None
_IS_VL_MODEL = None
_ENABLE_SP = None
_HAS_LAYER_IDX = None
_SUBSCRIBED_COMPUTE_STREAMS = set()
_GRAPH_PRINT_STREAM = None
_GRAPH_PRINT_STREAM_LOCK = Lock()


def _print_callback_on_stream(*args):
    """Callback function to print arguments on the dedicated print stream."""
    global _GRAPH_PRINT_STREAM
    with torch_npu.npu.stream(_GRAPH_PRINT_STREAM):
        print(*args, flush=True)


def acl_graph_print(*args):
    """
    Prints arguments from within an ACL graph.

    This function is provided for developers to print debug information when encountering
    issues within an ACL graph, pretty handy for dumping input/output tensor values, or
    resolving unexpected hangs. Usage:
    ```python
    from vllm_ascend.utils import acl_graph_print
    ...
    acl_graph_print("Debug info")
    ```

    This function launches a host function on the current compute stream to print
    the given arguments. It uses a dedicated stream for printing to avoid
    interfering with computation.

    NOTE: torch.compile does not support this function, only use this in non-compiled code.
    For example, those custom ops like `unified_attention_with_output` or `moe_forward`.
    """
    global _SUBSCRIBED_COMPUTE_STREAMS
    global _GRAPH_PRINT_STREAM

    current_compute_stream = torch_npu.npu.current_stream()

    with _GRAPH_PRINT_STREAM_LOCK:
        if _GRAPH_PRINT_STREAM is None:
            _GRAPH_PRINT_STREAM = torch_npu.npu.Stream()

        if current_compute_stream not in _SUBSCRIBED_COMPUTE_STREAMS:
            # Subscribe the compute stream to allow launching host functions.
            torch_npu.npu._subscribe_report(current_compute_stream)
            _SUBSCRIBED_COMPUTE_STREAMS.add(current_compute_stream)

    torch_npu.npu._launch_host_func(current_compute_stream,
                                    _print_callback_on_stream, args)


def _unregister_print_streams_on_exit():
    """Unsubscribe all compute streams used for printing at exit."""
    global _SUBSCRIBED_COMPUTE_STREAMS
    with _GRAPH_PRINT_STREAM_LOCK:
        for stream in _SUBSCRIBED_COMPUTE_STREAMS:
            torch_npu.npu._unsubscribe_report(stream)


atexit.register(_unregister_print_streams_on_exit)


def is_enable_nz():
    return envs_ascend.VLLM_ASCEND_ENABLE_NZ


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


def prefetch_stream() -> torch.npu.Stream:
    global _PREFETCH_STREAM
    if _PREFETCH_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _PREFETCH_STREAM = torch_npu.npu.Stream()
    return _PREFETCH_STREAM


def shared_experts_calculation_stream() -> torch.npu.Stream:
    global _SHARED_EXPERTS_CALCULATION_STREAM
    if _SHARED_EXPERTS_CALCULATION_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _SHARED_EXPERTS_CALCULATION_STREAM = torch_npu.npu.Stream()
    return _SHARED_EXPERTS_CALCULATION_STREAM


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


# Update cudagraph capture sizes for vllm config
def update_cudagraph_capture_sizes(vllm_config: VllmConfig,
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


def _is_default_capture_sizes(vllm_config: VllmConfig) -> bool:
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
    if vllm_config.model_config and vllm_config.model_config.hf_config.model_type == "qwen3_moe" \
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
            update_cudagraph_capture_sizes(vllm_config, enlarged_sizes)
            logger.info(
                "Adjusted ACL graphs: %s → %s for speculative decoding",
                original_sizes, enlarged_sizes)
        else:
            compilation_config.cudagraph_capture_sizes = original_sizes


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


def register_ascend_customop(vllm_config: Optional[VllmConfig] = None):
    """Register Ascend CustomOP

    NOTE: if the register branch requires model type, please use `vllm.config.get_current_vllm_config`,
    and ensure this will execute after model config is initilazed.
    """
    global _ASCEND_CUSTOMOP_IS_REIGISTERED
    if _ASCEND_CUSTOMOP_IS_REIGISTERED:
        return
    from vllm.model_executor.custom_op import CustomOp

    from vllm_ascend.ops.activation import AscendQuickGELU, AscendSiluAndMul
    from vllm_ascend.ops.fused_moe.fused_moe import (AscendFusedMoE,
                                                     AscendSharedFusedMoE)
    from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm
    from vllm_ascend.ops.linear import (AscendColumnParallelLinear,
                                        AscendMergedColumnParallelLinear,
                                        AscendQKVParallelLinear,
                                        AscendReplicatedLinear,
                                        AscendRowParallelLinear)
    from vllm_ascend.ops.mla import AscendMultiHeadLatentAttention
    from vllm_ascend.ops.rotary_embedding import (
        AscendDeepseekScalingRotaryEmbedding, AscendMRotaryEmbedding,
        AscendRotaryEmbedding, AscendYaRNRotaryEmbedding)
    from vllm_ascend.ops.vocab_parallel_embedding import (
        AscendLogitsProcessor, AscendParallelLMHead,
        AscendVocabParallelEmbedding)

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
    }

    for name, op_cls in REGISTERED_ASCEND_OPS.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)

    # NOTE: Keep this at last to ensure all custom actions are registered
    _ASCEND_CUSTOMOP_IS_REIGISTERED = True


class AscendDeviceType(Enum):
    _910B = 0  # A2
    _910_93 = 1  # A3
    _310P = 2
    _910_95 = 3  # A5


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
        cur_device_type = AscendDeviceType._910B
    elif 250 <= soc_version <= 255:
        cur_device_type = AscendDeviceType._910_93
    elif 200 <= soc_version <= 205:
        cur_device_type = AscendDeviceType._310P
    elif soc_version == 260:
        cur_device_type = AscendDeviceType._910_95
    else:
        raise RuntimeError(f"Can not support soc_version: {soc_version}.")

    assert _ascend_device_type == cur_device_type, f"Current device type: {cur_device_type} does not match the installed version's device type: {_ascend_device_type}, please check your installation package."


def get_ascend_device_type():
    global _ascend_device_type
    if _ascend_device_type is None:
        _init_ascend_device_type()
    return _ascend_device_type


def lmhead_tp_enable() -> bool:
    return get_ascend_config().lmhead_tensor_parallel_size is not None


def oproj_tp_enable() -> bool:
    return get_ascend_config().oproj_tensor_parallel_size is not None


def mlp_tp_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_MLP_OPTIMIZE


def matmul_allreduce_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE


def dense_optim_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE


def enable_sp(vllm_config=None) -> bool:
    global _ENABLE_SP
    if _ENABLE_SP is None:
        if vllm_config is None:
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
        _ENABLE_SP = (
            vllm_config.compilation_config.pass_config.
            enable_sequence_parallelism
            or envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1
            # Flash comm 1 should be enabled by env VLLM_ASCEND_ENABLE_FLASHCOMM1
            # We retain the env VLLM_ASCEND_ENABLE_FLASHCOMM here for backward compatibility.
            or bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM", '0'))))

        if not _ENABLE_SP:
            return _ENABLE_SP

        assert vllm_config.parallel_config.tensor_parallel_size > 1, \
            "Flash Comm v1 (Sequence Parallelism) is only supported when tp_size > 1."

        assert (
            not is_moe_model(vllm_config)
            or vllm_config.parallel_config.enable_expert_parallel
        ), "Flash Comm v1 (Sequence Parallelism) requires enable_expert_parallel=True for MoE models."

    return _ENABLE_SP


# TODO remove it after vllm has this func
def shared_expert_dp_enabled() -> bool:
    return get_ascend_config().enable_shared_expert_dp or enable_sp()


def prefill_context_parallel_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL


def is_moe_model(vllm_config: VllmConfig):
    """Checks if the model is a MoE model by config"""
    global _IS_MOE_MODEL
    if _IS_MOE_MODEL is None:
        model_configs = vllm_config.model_config.hf_config.to_dict()
        _IS_MOE_MODEL = _is_contain_expert(model_configs)
    return _IS_MOE_MODEL


def _is_contain_expert(config: Any):
    if isinstance(config, dict):
        for k, v in config.items():
            if "expert" in str(k):
                return True
            if _is_contain_expert(v):
                return True
    return False


def is_vl_model(vllm_config: VllmConfig):
    """Checks if the model is a VL model by config"""
    global _IS_VL_MODEL
    if _IS_VL_MODEL is None and vllm_config.model_config:
        model_configs = vllm_config.model_config.hf_config.to_dict()
        _IS_VL_MODEL = "VL" in model_configs["architectures"][0]
    return _IS_VL_MODEL


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
    raise ValueError("Invalid type for tensors")


def npu_stream_switch(target_stream: torch.npu.Stream,
                      *,
                      enabled: bool = True):
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


def get_hccl_config_for_pg_options(group_name: str) -> Optional[dict]:
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
        "dp": {
            "hccl_buffer_size": calculate_dp_buffer_size()
        },
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
    return (os.getenv("HCCL_INTRA_ROCE_ENABLE", "") == "0"
            and os.getenv("HCCL_INTRA_PCIE_ENABLE", "") == "1")


def has_layer_idx(model_instance: torch.nn.Module) -> bool:
    if model_instance is None:
        return False

    global _HAS_LAYER_IDX
    if _HAS_LAYER_IDX is None:
        _HAS_LAYER_IDX = hasattr(model_instance, "model") and \
            hasattr(model_instance.model, "start_layer")
    return _HAS_LAYER_IDX


def flashcomm2_enable() -> bool:
    return envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE > 0


def get_flashcomm2_oproj_tp_size_and_validate_config(ascend_config,
                                                     vllm_config):
    flashcomm2_oproj_tp_size = envs_ascend.VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE
    global_tp_size = vllm_config.parallel_config.tensor_parallel_size

    if not flashcomm2_enable():
        logger.info("FLASHCOMM2 not enable.")
        return flashcomm2_oproj_tp_size

    logger.info(
        f"Enable FLASHCOMM2 with flashcomm2_oproj_tensor_parallel_size={flashcomm2_oproj_tp_size} and global_tp_size={global_tp_size}"
    )
    if not envs_ascend.VLLM_ASCEND_ENABLE_FLASHCOMM1:
        logger.warning_once(
            "It is recommended to enable FLASHCOMM1 simultaneously when starting FLASHCOMM2 for optimal performance."
        )
    if ascend_config.oproj_tensor_parallel_size is not None:
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
            "FLASHCOMM2 primarily targets P-scenario deployments, "
            "with additional support for hybrid deployment scenarios. "
            "It is not applicable in D-scenario environments.")

    return flashcomm2_oproj_tp_size


def get_flashcomm2_reorgnized_batch_ids(global_tp_size) -> list[list[int]]:
    # Reorganize batch_ids so that, after the all2all and reduce-scatter operation, each batch_id corresponds to the rank_id within the DP domain.
    # For example, when DP = [0, 1, 2, ..., 15] and flashcomm2_oproj_tensor_parallel_size = 2,
    # the reorganized batch_ids will be [[batch0, batch8], [batch1, batch9], ..., [batch7, batch15]].
    flashcomm2_otp_size = get_ascend_config(
    ).flashcomm2_oproj_tensor_parallel_size
    num_oproj_tensor_parallel_groups: int = (global_tp_size //
                                             flashcomm2_otp_size)

    reorgnized_batch_ids = []
    for i in range(num_oproj_tensor_parallel_groups):
        ranks = []
        for j in range(flashcomm2_otp_size):
            rank_idx = i + j * num_oproj_tensor_parallel_groups
            ranks.append(rank_idx)
        reorgnized_batch_ids.append(ranks)

    return reorgnized_batch_ids
