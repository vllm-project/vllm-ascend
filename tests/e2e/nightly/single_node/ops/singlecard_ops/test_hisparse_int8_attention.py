# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 sparse Attention accuracy against an independent CPU reference."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import AscendDeviceType, bootstrap_custom_op_env, get_ascend_device_type


@pytest.fixture(autouse=True)
def _require_arch22():
    bootstrap_custom_op_env()
    if not torch.npu.is_available() or get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        pytest.skip("requires A2/A3")
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-not-found, import-untyped]  # noqa: F401


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("stride_rows", [128, 256])
@pytest.mark.parametrize("rows", [1, 7, 35])
def test_int8_physical_attention_reference(rows, stride_rows, capture):
    torch.manual_seed(47)
    blocks, block, heads, width = 17, 128, 4, 656
    nope = torch.randint(-127, 128, (blocks * block, 512), dtype=torch.int8)
    rope = torch.randn(blocks * block, 64, dtype=torch.bfloat16)
    scales = torch.rand(blocks * block, 4) * 0.02 + 0.001
    # Build the operator's packed rows independently of the HiSparse adapter.
    source = torch.cat((nope, rope.view(torch.int8), scales.view(torch.int8)), dim=-1).reshape(blocks, block, width)
    backing = torch.full((blocks * stride_rows + block, width), 13, dtype=torch.int8, device="npu")
    cache = backing.as_strided((blocks, block, width), (stride_rows * width, width, 1), storage_offset=block * width)
    cache.copy_(source)
    query = torch.randn(rows, heads, 512, dtype=torch.bfloat16)
    query_rope = torch.randn(rows, heads, 64, dtype=torch.bfloat16)
    physical = torch.randint(0, blocks * block, (rows, 2048), dtype=torch.int32)
    physical[:, 3::7] = -1
    physical[0] = -1
    physical[0, :3] = torch.tensor([0, 0, 1], dtype=torch.int32)
    if rows > 1:
        physical[-1] = -1
    packed_query = torch.cat((query, query_rope), dim=-1).npu()
    # The native operator consumes compact, physical row addresses.
    mapped = torch.full_like(physical, -1)
    for row in range(rows):
        selected = physical[row][physical[row] >= 0]
        mapped[row, : selected.numel()] = selected // block * stride_rows + selected % block
    indices = mapped.npu()
    addressable_rows = (blocks - 1) * stride_rows + block
    view = cache.as_strided((addressable_rows, 1, width), (width, width, 1))
    query_ends = torch.tensor([rows], dtype=torch.int32, device="npu")
    key_length = torch.tensor([addressable_rows], dtype=torch.int32, device="npu")
    scale = 576**-0.5

    def forward():
        result = torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm(
            query=packed_query,
            key=view,
            value=view,
            sparse_indices=indices.unsqueeze(1),
            scale_value=scale,
            sparse_block_size=1,
            actual_seq_lengths_query=query_ends,
            actual_seq_lengths_kv=key_length,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=0,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            key_quant_mode=2,
            value_quant_mode=2,
            rope_head_dim=64,
            return_softmax_lse=False,
        )
        return result[0] if isinstance(result, tuple) else result

    graph = None
    try:
        actual = forward()
        if capture:
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                actual = forward()
            indices.copy_(mapped.roll(1, dims=0).npu())
            physical = physical.roll(1, dims=0)
            graph.replay()
        actual = actual.float().cpu()
        latent = (nope.float() * scales.repeat_interleave(128, dim=-1)).bfloat16().float()
        keys = torch.cat((latent, rope.float()), dim=-1)
        queries = torch.cat((query.float(), query_rope.float()), dim=-1)
        expected = torch.zeros_like(actual)
        for row in range(rows):
            selected = physical[row][physical[row] >= 0].long()
            if selected.numel():
                scores = queries[row] @ keys[selected].T * scale
                expected[row] = scores.softmax(-1) @ latent[selected]
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.02)
        assert torch.isfinite(actual).all()
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
