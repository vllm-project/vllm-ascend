# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 packed FP8 C8: LSE, compact indices, and empty DCP shards."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires the A5 packed FP8 kernel.",
)


@pytest.mark.parametrize("length,count", [(4096, 2048), (1024, 512), (128, 32), (128, 0)])
@pytest.mark.parametrize("heads", [16, 32, 64])
def test_a5_c8_lse(length, count, heads):
    # A5 selectively uses this extension; keep registration lazy.
    import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401

    torch.manual_seed(16656)
    page_size, nope_dim, rope_dim, tile_size = 128, 512, 64, 128
    query = torch.randn(1, heads, nope_dim + rope_dim).to(torch.bfloat16)
    key_nope = (torch.randn(length, 1, nope_dim) * 2).to(torch.float8_e4m3fn)
    key_rope = torch.randn(length, 1, rope_dim).to(torch.bfloat16)
    scales = torch.rand(length, 1, nope_dim // tile_size) * 0.1 + 0.05
    cache = (
        torch.cat([key_nope.view(torch.uint8), key_rope.view(torch.uint8), scales.view(torch.uint8)], -1)
        .reshape(length // page_size, page_size, 1, -1)
        .view(torch.float8_e4m3fn)
    )
    indices = torch.full((1, 1, 2048), -1, dtype=torch.int32)
    indices[0, 0, :count] = torch.randperm(length)[:count].to(torch.int32)
    scale = (nope_dim + rope_dim) ** -0.5
    kwargs = dict(
        query=query.npu(),
        key=cache.npu(),
        value=cache.npu(),
        sparse_indices=indices.npu(),
        scale_value=scale,
        sparse_block_size=1,
        block_table=torch.arange(length // page_size, dtype=torch.int32).view(1, -1).npu(),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor([length], dtype=torch.int32).npu(),
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=tile_size,
        key_quant_mode=2,
        value_quant_mode=2,
        rope_head_dim=rope_dim,
    )
    output, max_value, exp_sum = torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm(
        **kwargs, return_softmax_lse=True
    )
    output, max_value, exp_sum = output.cpu().float(), max_value.cpu().float(), exp_sum.cpu().float()
    assert torch.isfinite(output).all()
    if count == 0:
        assert torch.count_nonzero(output) == 0
        assert torch.count_nonzero(exp_sum) == 0
        assert torch.isneginf(max_value + exp_sum.log()).all()
        return
    selected = indices[0, 0, :count].long()
    dequant = (key_nope.float() * scales.repeat_interleave(tile_size, -1)).to(torch.bfloat16)
    key = torch.cat([dequant, key_rope], -1)[selected, 0].float()
    scores = query[0].float() @ key.T * scale
    reference = torch.softmax(scores, -1) @ dequant[selected, 0].float()
    torch.testing.assert_close(output[0], reference, atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(
        (max_value + exp_sum.log()).reshape(-1), torch.logsumexp(scores, -1), atol=1e-5, rtol=1e-5
    )
    if count == 2048:
        builtin = torch_npu.npu_kv_quant_sparse_flash_attention(**kwargs).cpu().float()
        torch.testing.assert_close(output, builtin, atol=0, rtol=0)
