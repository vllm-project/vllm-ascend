# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""按token读取非连续ND MLA缓存，不重新排列或复制整个缓存。"""

import torch
import torch_npu

from vllm_ascend.ops.triton.pcp_kv_cache import copy_pcp_kv_cache

PCP_COPY_MAX_TILE_ROWS = 8


def try_load_strided_mla_cache(cache_k, cache_v, block_table, seq_lens, seq_offset, key, value) -> bool:
    """只处理已验证的4D、单物理头、FP16/BF16非连续ND缓存。"""
    if (
        cache_k.ndim != 4
        or cache_v.ndim != 4
        or cache_k.shape[2] != 1
        or cache_v.shape[:3] != cache_k.shape[:3]
        or cache_k.dtype not in (torch.float16, torch.bfloat16)
        or cache_v.dtype != cache_k.dtype
        or cache_k.device.type != "npu"
        or (cache_k.is_contiguous() and cache_v.is_contiguous())
    ):
        return False
    # 0/2是NCHW/ND的物理格式；NZ等布局维持原生reader路径。
    if int(torch_npu.get_npu_format(cache_k)) not in (0, 2) or int(torch_npu.get_npu_format(cache_v)) not in (0, 2):
        return False
    num_sequences = seq_lens.numel()
    num_tokens = key.shape[0]
    if (
        seq_lens.ndim != 1
        or block_table.ndim != 2
        or block_table.shape[0] < num_sequences
        or (seq_offset is not None and (seq_offset.ndim != 1 or seq_offset.numel() < num_sequences))
        or key.ndim != 3
        or value.ndim != 3
        or key.shape != (num_tokens, 1, cache_k.shape[-1])
        or value.shape != (num_tokens, 1, cache_v.shape[-1])
        or key.dtype != cache_k.dtype
        or value.dtype != cache_v.dtype
        or any(tensor.device != cache_k.device for tensor in (cache_v, block_table, seq_lens, key, value))
        or (seq_offset is not None and seq_offset.device != cache_k.device)
        or cache_k.shape[1] <= 0
    ):
        raise ValueError("Strided MLA cache reader received incompatible metadata or output buffers.")
    if num_tokens == 0:
        return True
    # GPU长度是base/DCP各自真实local长度；不使用global CPU累计长度。
    lengths = seq_lens.to(torch.int64)
    requests = torch.repeat_interleave(
        torch.arange(num_sequences, device=seq_lens.device, dtype=torch.int64),
        lengths,
        output_size=num_tokens,
    )
    prefix_starts = torch.cumsum(lengths, dim=0) - lengths
    positions = torch.arange(num_tokens, device=seq_lens.device, dtype=torch.int64)
    positions = positions - prefix_starts.index_select(0, requests)
    if seq_offset is not None:
        positions = positions + seq_offset.to(torch.int64).index_select(0, requests)
    block_size = cache_k.shape[1]
    physical_blocks = block_table[requests, positions // block_size].to(torch.int64)
    slots = physical_blocks * block_size + positions % block_size

    # 既有A3多行copy在不完整末tile上会报MTE；最大row tile为8。
    # 重复已有合法slot使全部地址有效，最多额外读取7个token，不读取新页。
    padded_tokens = (num_tokens + PCP_COPY_MAX_TILE_ROWS - 1) // PCP_COPY_MAX_TILE_ROWS * PCP_COPY_MAX_TILE_ROWS
    if padded_tokens != num_tokens:
        slots = torch.cat((slots, slots[:1].expand(padded_tokens - num_tokens)))
    packed = copy_pcp_kv_cache((cache_k, cache_v), slots)[:num_tokens]
    # split视图row stride包含另一组件；复制到原TND输出保持消费者契约。
    key.copy_(packed[:, : cache_k.shape[-1]].view(num_tokens, 1, cache_k.shape[-1]))
    value.copy_(packed[:, cache_k.shape[-1] :].view(num_tokens, 1, cache_v.shape[-1]))
    return True
