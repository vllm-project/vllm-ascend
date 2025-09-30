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
import fcntl
import math
import os
import shutil
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch_npu
import torchair  # type: ignore[import]  # noqa: F401
from packaging.version import InvalidVersion, Version
from torch_npu.npu.streams import Event
from torchair.scope import super_kernel as _super_kernel
from vllm.logger import logger

import vllm_ascend.envs as envs

try:
    # Recent release of torchair has moved these ops to `.scope`.
    from torchair.scope import npu_stream_switch as _npu_stream_switch
    from torchair.scope import npu_wait_tensor as _npu_wait_tensor
except ImportError:
    from torchair.ops import NpuStreamSwitch as _npu_stream_switch
    from torchair.ops import npu_wait_tensor as _npu_wait_tensor

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

# NOTE: Currently, we can only capture 1920 graphs at most,
# due to the limitation of ACL graph. This number is bounded by
# the number of streams, which is 2048, we save 128 streams
# as a buffer.
# Maximum number of graphs that can be captured by ACL Graph
MAX_CAPTURE_SIZE = 1920

ASCEND_QUATIZATION_METHOD = "ascend"

CUSTOM_OP_ENABLED = None

ACL_FORMAT_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29

KV_CACHE_BYTES_CACHE_PATH_NAME = ".kv_cache_bytes"
KV_CACHE_BYTES_CACHE_FILE_NAME = "kv_cache_bytes"
TORCHAIR_CACHE_PATH_NAME = ".torchair_cache"
TORCHAIR_CACHE_DIR = os.path.join(
    os.getenv('TORCHAIR_CACHE_HOME', os.getcwd()), TORCHAIR_CACHE_PATH_NAME)


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
    global CUSTOM_OP_ENABLED

    if CUSTOM_OP_ENABLED is not None:
        return CUSTOM_OP_ENABLED

    else:
        try:
            # register custom ops into torch_library here
            import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401
            CUSTOM_OP_ENABLED = True

        except ImportError:
            CUSTOM_OP_ENABLED = False
            logger.warning(
                "Warning: Failed to register custom ops, all custom ops will be disabled"
            )

        return CUSTOM_OP_ENABLED


def find_hccl_library() -> str:
    """
    We either use the library file specified by the `HCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libhccl.so` can be
    found by `ctypes` automatically.
    """
    so_file = envs.HCCL_SO_PATH

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


_current_stream = None


def current_stream() -> torch.npu.Stream:
    """
    replace `torch.npu.current_stream()` with `vllm.utils.current_stream()`.
    it turns out that `torch.npu.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.npu.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.npu.current_stream()`.

    """
    global _current_stream
    if _current_stream is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _current_stream = torch.npu.current_stream()
    return _current_stream


def adapt_patch(is_global_patch: bool = False):
    if is_global_patch:
        from vllm_ascend.patch import platform  # noqa: F401
    else:
        from vllm_ascend.patch import worker  # noqa: F401


def vllm_version_is(target_vllm_version: str):
    if envs.VLLM_VERSION is not None:
        vllm_version = envs.VLLM_VERSION
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
        elif isinstance(d, list):
            for item in d:
                _rec_find(item)

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

    if compilation_config.full_cuda_graph:
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        truncated_sizes = [x for x in original_sizes if x <= max_num_seqs]
        compilation_config.init_with_cudagraph_sizes(truncated_sizes)

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
        return

    # Calculate parallel configuration factor
    hf_config = vllm_config.model_config.hf_config
    if hasattr(hf_config, 'num_hidden_layers'):
        num_hidden_layers = hf_config.num_hidden_layers
    else:
        num_hidden_layers = get_max_hidden_layers(hf_config)
    parallel_config = vllm_config.parallel_config

    # TODO: Find out whether we need to take into account the pp_size
    parallel_factor = 1 + sum(size > 1 for size in [
        parallel_config.data_parallel_size_local,
        parallel_config.tensor_parallel_size,
        parallel_config.expert_parallel_size,
        parallel_config.expert_tensor_parallel_size,
    ])

    # Calculate maximum supported batch sizes considering model architecture
    max_num_batch_sizes = math.floor(MAX_CAPTURE_SIZE /
                                     (num_hidden_layers + 1) / parallel_factor)
    logger.info("Calculated maximum supported batch sizes for ACL graph: %s",
                max_num_batch_sizes)

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
        if not envs.VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE:
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
        if not envs.VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE:
            return durations

        while self._observations:
            with self._lock:
                tag, observe_start, observe_end = self._observations.pop()
            observe_end.synchronize()
            durations[tag] = observe_start.elapsed_time(observe_end)

        return durations


def npu_stream_switch(tag: str, priority: int, *, enabled: bool = True):
    return _npu_stream_switch(tag, priority) if enabled else nullcontext()


def super_kernel(prefix: str, stream: str, enabled: bool = True):
    return _super_kernel(prefix, stream) if enabled else nullcontext()


def npu_wait_tensor(self: torch.Tensor,
                    dependency: torch.Tensor,
                    *,
                    enabled: bool = True):
    return _npu_wait_tensor(self, dependency) if enabled else self


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


class AscendSocVersion(Enum):
    A2 = 0
    A3 = 1
    MAX = 2


_ascend_soc_version = None


def init_ascend_soc_version():
    soc_version = torch_npu.npu.get_soc_version()
    global _ascend_soc_version
    if 220 <= soc_version <= 225:
        _ascend_soc_version = AscendSocVersion.A2
    elif 250 <= soc_version <= 255:
        _ascend_soc_version = AscendSocVersion.A3
    else:
        _ascend_soc_version = AscendSocVersion.MAX


def get_ascend_soc_version():
    global _ascend_soc_version
    assert _ascend_soc_version is not None
    return _ascend_soc_version


@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]


_graph_params: Optional[GraphParams] = None


def set_graph_params(aclgraph_capture_sizes: set[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        {size: []
         for size in aclgraph_capture_sizes},
        {size: None
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
        {size: []
         for size in aclgraph_capture_sizes},
    )


def get_graph_params():
    return _graph_params


def get_torchair_current_work_dir(file_name=None):
    if file_name is None:
        return TORCHAIR_CACHE_DIR
    return os.path.join(TORCHAIR_CACHE_DIR, file_name)


def check_torchair_cache_exist():
    res = False
    torch_air_abs_path = get_torchair_current_work_dir()
    if os.path.exists(torch_air_abs_path):
        file_list = os.listdir(torch_air_abs_path)
        if len(file_list) != 0:
            res = True
            return res


def check_kv_cache_bytes_cache_exist():
    res = False
    kv_cache_bytes_cache_abs_path = get_torchair_current_work_dir(
        KV_CACHE_BYTES_CACHE_PATH_NAME)
    if os.path.exists(kv_cache_bytes_cache_abs_path):
        file_list = os.listdir(kv_cache_bytes_cache_abs_path)
        if len(file_list) != 0:
            res = True
    return res


def read_kv_cache_bytes_from_file(rank) -> int:
    kv_cache_bytes = -1
    kv_cache_bytes_cache_abs_path = get_torchair_current_work_dir(
        KV_CACHE_BYTES_CACHE_PATH_NAME)
    kv_cache_bytes_file = os.path.join(
        kv_cache_bytes_cache_abs_path,
        f"{rank}_{KV_CACHE_BYTES_CACHE_FILE_NAME}")
    with open(kv_cache_bytes_file, "r", encoding="utf-8") as f:
        with file_lock(f, fcntl.LOCK_SH):
            kv_cache_bytes = int(f.readline())
    return kv_cache_bytes


@contextmanager
def file_lock(file_descriptor, lock_type):
    fcntl.flock(file_descriptor, lock_type)
    try:
        yield
    finally:
        fcntl.flock(file_descriptor, fcntl.LOCK_UN)


def write_kv_cache_bytes_to_file(rank, kv_cache_bytes):
    kv_cache_bytes_cache_abs_path = get_torchair_current_work_dir(
        KV_CACHE_BYTES_CACHE_PATH_NAME)
    os.makedirs(kv_cache_bytes_cache_abs_path, exist_ok=True)
    kv_cache_bytes_file = os.path.join(
        kv_cache_bytes_cache_abs_path,
        f"{rank}_{KV_CACHE_BYTES_CACHE_FILE_NAME}")
    with open(kv_cache_bytes_file, "w", encoding="utf-8") as f:
        with file_lock(f, fcntl.LOCK_EX):
            f.write(f"{kv_cache_bytes}")


def delete_torchair_cache_file():
    torch_air_abs_path = get_torchair_current_work_dir()
    try:
        shutil.rmtree(torch_air_abs_path)
    except FileNotFoundError:
        pass
