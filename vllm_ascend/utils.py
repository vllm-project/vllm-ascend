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
import torch
import torch_npu
from packaging.version import InvalidVersion, Version
from vllm.logger import logger

import vllm_ascend.envs as envs

# 310P3 202, 910B4 224
SOC_VERSION = None
SOC_VERSION_310P = 202

ACL_FORMAT_FRACTAL_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29

def is_310p():
    global SOC_VERSION
    if SOC_VERSION is None:
        torch.npu.get_device_name()
        SOC_VERSION = torch_npu._C._npu_get_soc_version()
    return SOC_VERSION == SOC_VERSION_310P


class NullHandle:

    def __init__(self):
        pass

    def wait(self):
        pass


def _round_up(x: int, align: int):
    if align == 0:
        return -1
    return (x + align - 1) // align * align


def _custom_pad(x, pad_dims):
    return torch.nn.functional.pad(x, pad_dims)


def _custom_reshape(x, target_shape):
    return x.reshape(target_shape)


def _custom_transpose(x, dim1, dim2):
    return x.transpose(dim1, dim2)


def nd_to_nz_2d(in_tensor: torch.Tensor) -> torch.Tensor:
    aux_dims = [0, 0, 0, 0]
    aux_dims[0] = 1
    aux_dims[1] = _round_up(in_tensor.size(0), 16)

    pad_dims = [0, 0, 0, 0]
    pad_dims[3] = _round_up(in_tensor.size(0), 16) - in_tensor.size(0)

    aux_dims[2] = _round_up(in_tensor.size(1), 16) // 16
    aux_dims[3] = 16
    pad_dims[1] = _round_up(in_tensor.size(1), 16) - in_tensor.size(1)

    return _custom_transpose(
        _custom_reshape(_custom_pad(in_tensor, pad_dims), aux_dims), 1, 2
    ).contiguous()


def communication_adaptation_310p():

    def broadcast310p(tensor, src, group=None, async_op=False):
        rank = torch.distributed.get_rank(group)
        world_size = torch.distributed.get_world_size(group)
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        tensor_list[rank] = tensor
        torch.distributed.all_gather(tensor_list, tensor, group=group)
        tensor[...] = tensor_list[src]
        if async_op:
            return NullHandle()
        else:
            return None

    torch.distributed.broadcast = broadcast310p

    def all_reduce_wrapper_310p(fn):

        def all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=None,
            async_op=False,
        ):
            if tensor.dtype != torch.int64:
                return fn(tensor, op, group, async_op)
            rank = torch.distributed.get_rank(group)
            world_size = torch.distributed.get_world_size(group)
            tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
            tensor_list[rank] = tensor
            torch.distributed.all_gather(tensor_list, tensor, group=group)
            if op == torch.distributed.ReduceOp.SUM:
                return torch.stack(tensor_list).sum(0)
            elif op == torch.distributed.ReduceOp.MAX:
                return torch.tensor(
                    torch.stack(tensor_list).cpu().numpy().max(0),
                    device=tensor.device,
                )
            else:
                raise RuntimeError(f"not implement op {op}")

        return all_reduce

    torch.distributed.all_reduce = all_reduce_wrapper_310p(
        torch.distributed.all_reduce
    )

    def reduce_scatter_310p(output_tensor, input_tensor, group=None):
        rank = torch.distributed.get_rank(group)
        world_size = torch.distributed.get_world_size(group)
        torch.distributed.all_reduce(
            input_tensor, torch.distributed.ReduceOp.SUM, group, async_op=False
        )
        interval = input_tensor.shape[0] // world_size
        output_tensor[:] = input_tensor[rank * interval : (rank + 1) * interval]

    torch.distributed._reduce_scatter_base = reduce_scatter_310p


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
            "to control it by hand. And please make sure the vaule follows the "
            "format of x.y.z.")
