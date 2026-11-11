# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
import os
from functools import cache

import torch
import torch_npu
from vllm.logger import logger


@cache
def get_selective_lim_library():
    path = os.environ.get("VLLM_ASCEND_SELECTIVE_LIM_LIBRARY")
    if not path:
        raise RuntimeError("Selective KV offload requires VLLM_ASCEND_SELECTIVE_LIM_LIBRARY for LIM and SDMA")
    return ctypes.CDLL(path)


@cache
def get_selective_lim_launcher():
    library = get_selective_lim_library()
    launch = library.aclrtlaunch_selective_lim_topk
    launch.argtypes = (
        [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 11 + [ctypes.c_int] * 9 + [ctypes.c_uint64] * 4
    )
    launch.restype = ctypes.c_uint32
    return launch


def prepare_selective_lim(launch, manager, layer_id, topk_indices, metadata, seq_lengths) -> None:
    topk = topk_indices.reshape(-1, manager.topk).to(torch.int64).contiguous()
    rows, width = topk.shape
    if not 0 < width <= manager.topk_buffer_size <= 2048:
        raise ValueError("Precomputed-TopK LIM requires 0 < TopK <= hot capacity <= 2048")
    if metadata.req_ids_tensor is None or metadata.selective_tail_rows is None:
        raise RuntimeError("Precomputed-TopK LIM requires request identities and selective tail metadata")
    table = metadata.block_table[:rows].to(torch.int32).contiguous()
    entries = 2 * rows * width
    src = manager.selective_copy_src[layer_id, :entries]
    dst = manager.selective_copy_dst[layer_id, :entries]
    sizes = manager.selective_copy_lengths[layer_id, :entries]
    inputs_outputs = (
        topk,
        table,
        seq_lengths[:rows].to(torch.int32).contiguous(),
        metadata.req_ids_tensor[:rows].to(torch.int64).contiguous(),
        metadata.selective_tail_rows[:rows].to(torch.int32).contiguous(),
        manager.selective_hot_keys[layer_id],
        manager.selective_hot_requests[layer_id],
        manager.selective_physical[layer_id],
        src,
        dst,
        sizes,
    )
    result = launch(
        rows,
        torch_npu.npu.current_stream().npu_stream,
        *(tensor.data_ptr() for tensor in inputs_outputs),
        rows,
        width,
        table.shape[1],
        manager.kv_cache_config.num_blocks,
        manager.block_size,
        manager.topk_buffer_size,
        manager.selective_row_stride,
        manager.token_size_bytes_k,
        manager.token_size_bytes_v,
        manager.gvas_k_bases[layer_id],
        manager.gvas_v_bases[layer_id],
        manager.topk_buffers_k[layer_id].data_ptr(),
        manager.topk_buffers_v[layer_id].data_ptr(),
    )
    if result != 0:
        raise RuntimeError(f"Precomputed-TopK LIM launch failed with status {result}")
    logger.info_once("Selective KV offload launched the native precomputed-TopK LIM kernel.")
    manager.prefetch_selective_kv(rows, src, dst, sizes)
