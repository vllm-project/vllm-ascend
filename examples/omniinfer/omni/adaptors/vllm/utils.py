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

import gc
import math
import os
import shutil
from typing import TYPE_CHECKING

import omni.adaptors.vllm.envs as envs
import torch
from packaging.version import InvalidVersion, Version
from vllm.logger import logger

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

ASCEND_COMPRESSED_TENSORS = "ascend_compressed_tensors"
NPU_W8A8_DYNAMIC = 'npu_w8a8_dynamic'
SUPPORTED_QUANTIZATION_METHODS = [ASCEND_COMPRESSED_TENSORS, NPU_W8A8_DYNAMIC]

BLOCK_NUM_CACHE_PATH_NAME = ".block_nums"
BLOCK_NUM_CACHE_FILE_NAME = "block_num"


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


def update_aclgraph_sizes(vllm_config: VllmConfig) -> None:
    """Update ACL graph capture sizes based on hardware limitations"""
    # Store original configuration and temporarily clear it
    compilation_config = vllm_config.compilation_config
    original_sizes, compilation_config.cudagraph_capture_sizes = \
        compilation_config.cudagraph_capture_sizes, None

    num_hidden_layers = vllm_config.model_config.hf_config.num_hidden_layers
    parallel_config = vllm_config.parallel_config
    parallel_factor = 1 + sum(size > 1 for size in [
        parallel_config.data_parallel_size,
        parallel_config.tensor_parallel_size
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


def get_current_work_dir(file_name=None):
    if file_name is None:
        return envs.TORCHAIR_CACHE_HOME
    return os.path.join(envs.TORCHAIR_CACHE_HOME, file_name)


def check_torchair_cache_exists():
    res = False
    torch_air_abs_path = get_current_work_dir()
    try:
        if os.path.exists(torch_air_abs_path):
            file_list = os.listdir(torch_air_abs_path)
            if len(file_list) != 0:
                res = True
    except PermissionError:
        logger.debug("No permission to read the torchair graph cache file")
    return res


def check_block_num_cache_exist():
    res = False
    block_num_cache_abs_path = get_current_work_dir(BLOCK_NUM_CACHE_PATH_NAME)
    try:
        if os.path.exists(block_num_cache_abs_path):
            file_list = os.listdir(block_num_cache_abs_path)
            if len(file_list) != 0:
                res = True
    except PermissionError:
        logger.debug("No permission to read the block num cache file")
    return res


def read_block_num_from_file(rank):
    block_bytes = -1
    block_num_cache_abs_path = get_current_work_dir(BLOCK_NUM_CACHE_PATH_NAME)
    try:
        block_num_file = os.path.join(block_num_cache_abs_path,
                                      f"{rank}_{BLOCK_NUM_CACHE_FILE_NAME}")
        with open(block_num_file, "r", encoding="utf-8") as f:
            block_bytes = int(f.readline())
    except Exception:
        logger.debug("Failed to read the %d block num cache file", rank)
    return block_bytes


def write_block_num_to_file(rank, block_bytes):
    block_num_cache_abs_path = get_current_work_dir(BLOCK_NUM_CACHE_PATH_NAME)
    if not os.path.exists(block_num_cache_abs_path):
        try:
            os.makedirs(block_num_cache_abs_path)
        except Exception:
            logger.debug("Path %s has created", block_num_cache_abs_path)
    try:
        block_num_file = os.path.join(block_num_cache_abs_path,
                                      f"{rank}_{BLOCK_NUM_CACHE_FILE_NAME}")
        with open(block_num_file, "w", encoding="utf-8") as f:
            f.write(f"{block_bytes}")
    except Exception:
        logger.debug("Failed to write block num into file:%s", block_num_file)


def delete_torchair_cache_file():
    torch_air_abs_path = get_current_work_dir()
    if os.path.exists(torch_air_abs_path):
        try:
            shutil.rmtree(torch_air_abs_path)
        except Exception:
            logger.debug("Failed to remove the file:%s", torch_air_abs_path)


def clear_var(*need_clear_tensors):
    for t in need_clear_tensors:
        del t
    gc.collect()
    torch.npu.empty_cache()


def get_attr_by_names(src_config, attrs, default_value):
    for attr in attrs:
        value = getattr(src_config, attr, 0)
        if value > 0:
            return value
    return default_value
