# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch_npu

from vllm_ascend.ops import sol_attn
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op


@pytest.fixture(scope="module")
def npu_device():
    bootstrap_custom_op_env()
    if not torch.npu.is_available() or "910B" not in torch.npu.get_device_name():
        pytest.skip("Sol-Attn requires Ascend 910B")
    assert enable_custom_op()
    return torch.device("npu", torch.npu.current_device())


def _inputs(batch, tokens, heads):
    generator = torch.Generator().manual_seed(27)
    return tuple(torch.randn((batch, tokens, heads, 128), generator=generator).bfloat16() for _ in range(3))


def _assert_relative_l2(actual, expected, tolerance):
    difference = torch.linalg.vector_norm(actual.float() - expected.float())
    assert difference / torch.linalg.vector_norm(expected.float()) < tolerance


def test_sol_attn_all_exact(npu_device):
    q, k, v = (tensor.to(npu_device) for tensor in _inputs(1, 145, 4))
    scale = q.size(-1) ** -0.5
    out, lse = sol_attn.sol_attn(q, k, v, scale=scale, tau=-100.0, return_lse=True)
    dense, _ = torch_npu.npu_fused_infer_attention_score_v2(
        q,
        k,
        v,
        num_query_heads=q.size(2),
        num_key_value_heads=k.size(2),
        input_layout="BSND",
        softmax_scale=scale,
        sparse_mode=0,
        pre_tokens=2147483647,
        next_tokens=2147483647,
    )
    _assert_relative_l2(out.cpu(), dense.cpu(), 3e-3)
    scores = q.cpu().transpose(1, 2).float() @ k.cpu().permute(0, 2, 3, 1).float()
    torch.testing.assert_close(lse.cpu(), (scores * scale).logsumexp(dim=-1), rtol=1e-3, atol=3e-3)


def _sparse_reference(q, k, v, sink_block):
    q, k, v = (tensor.transpose(1, 2).float() for tensor in (q, k, v))
    scale = q.size(-1) ** -0.5
    key_blocks = k.split(64, dim=2)
    value_blocks = v.split(64, dim=2)
    outputs = []
    for tile, query in enumerate(q.split(128, dim=2)):
        scores = []
        values = []
        for block, (key, value) in enumerate(zip(key_blocks, value_blocks)):
            if block == sink_block or 2 * tile - 1 <= block <= 2 * tile + 2:
                scores.append((query @ key.transpose(2, 3)) * scale)
                values.append(value)
            else:
                key_mean = key.mean(dim=2, keepdim=True).bfloat16().float()
                value_mean = value.mean(dim=2, keepdim=True).bfloat16().float()
                scores.append((query @ key_mean.transpose(2, 3)) * scale + math.log(key.size(2)))
                values.append(value_mean)
        outputs.append(torch.cat(scores, dim=-1).softmax(dim=-1) @ torch.cat(values, dim=2))
    return torch.cat(outputs, dim=2).transpose(1, 2)


def test_sol_attn_sparse_chunked_heads(npu_device, monkeypatch):
    # Exercise slice writes even for a small sequence.
    monkeypatch.setattr(sol_attn, "_SCORE_BUFFER_BYTES", 1)
    q, k, v = _inputs(2, 399, 2)
    out = sol_attn.sol_attn(*(tensor.to(npu_device) for tensor in (q, k, v)), tau=100.0, sink_start=3, sink_tokens=17)
    _assert_relative_l2(out.cpu(), _sparse_reference(q, k, v, sink_block=0), 6e-3)


def test_sol_attn_route_pack_writes_unsigned_slice(npu_device):
    # Block 63 sets the high bit of the first mask word. Head 0 is a guard
    # region for the three output tensors passed as views of head 1.
    blocks, padded_blocks, query_tiles, padded_tiles = 65, 128, 33, 40
    scores = torch.zeros((1, blocks, padded_blocks), device=npu_device, dtype=torch.float32)
    threshold = torch.ones((1, blocks), device=npu_device, dtype=torch.float32)
    mask = torch.full((2, query_tiles, 2), -1, device=npu_device, dtype=torch.int64)
    count = torch.full((2, padded_tiles), -1, device=npu_device, dtype=torch.int32)
    index = torch.full((2, query_tiles, padded_blocks), -1, device=npu_device, dtype=torch.int16)
    torch.ops._C_ascend.sol_attn_prepare_route_pack(
        scores, threshold, blocks, blocks, 1.0, mask[1:].view(torch.uint64), count[1:], index[1:].view(torch.uint16)
    )
    mask, count, index = mask.cpu(), count.cpu(), index.cpu()
    assert (mask[0] == -1).all() and (count[0] == -1).all() and (index[0] == -1).all()
    assert mask[1, 0, 0] == -8
    assert count[1, 0] == 3
    assert index[1, 0, :3].tolist() == [0, 1, 2]
