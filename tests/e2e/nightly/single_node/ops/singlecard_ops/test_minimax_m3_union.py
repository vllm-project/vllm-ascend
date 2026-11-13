# SPDX-License-Identifier: Apache-2.0
"""Union exact-mask correctness across request boundaries and KV dtypes."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention import utils  # noqa: F401 -- normal plugin import order
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile
from vllm_ascend.models.minimax_m3.ops.msa_m3_union import load_union_extensions, minimax_m3_sparse_attn_union

_PAGE_SIZE = 128
_HEAD_DIM = 128
_NUM_HEADS = 16
_FP8_MAX = 448.0


def _reference(query, key, value, lengths, indices):
    expected = []
    offset = 0
    for request, query_len in enumerate(lengths):
        seq_len = 512 if request == 0 else 384
        base = 0 if request == 0 else 4
        for token in range(query_len):
            positions = [
                p
                for block in indices[offset + token]
                if block >= 0
                for p in range(block * _PAGE_SIZE, (block + 1) * _PAGE_SIZE)
                if p <= seq_len - query_len + token
            ]
            k = torch.stack([key[base + p // _PAGE_SIZE, p % _PAGE_SIZE, 0].float() for p in positions])
            v = torch.stack([value[base + p // _PAGE_SIZE, p % _PAGE_SIZE, 0].float() for p in positions])
            scores = (query[offset + token].float() @ k.T) * (_HEAD_DIM**-0.5)
            expected.append(scores.softmax(-1) @ v)
        offset += query_len
    return torch.stack(expected)


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("lengths", [(1, 1), (7, 8), (9, 17)])
@torch.no_grad()
def test_union_exact_mask_and_graph_replay(kv_dtype, lengths):
    if get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED:
        pytest.skip("Union kernels require Ascend 950")
    if load_union_extensions() is None:
        pytest.skip("Matched optional Union kernels are unavailable")
    torch.npu.set_device(0)
    torch.manual_seed(524)
    key = torch.randn((7, _PAGE_SIZE, 1, _HEAD_DIM)).to(kv_dtype)
    value = torch.randn_like(key.float()).to(kv_dtype)
    query_dtype = torch.float16 if kv_dtype == torch.float16 else torch.bfloat16
    query_cpu = torch.randn((sum(lengths), _NUM_HEADS, _HEAD_DIM), dtype=query_dtype)
    query = query_cpu.npu()
    output = torch.empty_like(query)
    indices = []
    for request, query_len in enumerate(lengths):
        blocks = 4 if request == 0 else 3
        for token in range(query_len):
            selected = sorted({token % blocks, (token + 1) % blocks})
            indices.append(selected + [-1] * (3 - len(selected)))
    arguments = dict(
        q=query,
        kv_cache=(key.npu(), value.npu()),
        topk_idx=torch.tensor([indices], dtype=torch.int32, device="npu"),
        block_table=torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]], dtype=torch.int32, device="npu"),
        cu_seqlens_q=torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int64, device="npu"),
        cu_seqlens_kv=torch.tensor([0, 512, 896], dtype=torch.int64, device="npu"),
        seq_lens=torch.tensor([512, 384], dtype=torch.int64, device="npu"),
        num_union_groups=sum((n + 7) // 8 for n in lengths),
        max_kv_blocks=4,
        max_query_len=max(lengths),
        max_seq_len=512,
        sm_scale=_HEAD_DIM**-0.5,
        output=output,
    )

    def reference(q):
        if kv_dtype == torch.float8_e4m3fn:
            q = q.float().clamp(-_FP8_MAX, _FP8_MAX).to(kv_dtype)
        return _reference(q, key, value, lengths, indices)

    minimax_m3_sparse_attn_union(**arguments)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), reference(query_cpu), atol=0.02, rtol=0.02)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        minimax_m3_sparse_attn_union(**arguments)
    changed = (query_cpu * 0.75).to(query_dtype)
    query.copy_(changed.npu())
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu().float(), reference(changed), atol=0.02, rtol=0.02)
