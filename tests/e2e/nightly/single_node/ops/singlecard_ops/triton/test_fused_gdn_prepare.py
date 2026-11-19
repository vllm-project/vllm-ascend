# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.fla.fused_gdn_prepare import fused_gdn_prepare


def _inputs(tokens, tp_size, dtype, row_stride=False, device="npu:0"):
    width, heads, dim = 8192 // tp_size, 32 // tp_size, 128
    # Qwen3.6-35B: 16 K heads, 32 V heads, Dk=Dv=128.
    qkvz_width = width + heads * dim
    qkvz = torch.randn((tokens, qkvz_width + (17 if row_stride else 0)), dtype=dtype, device=device)
    ba = torch.randn((tokens, 2 * heads + (3 if row_stride else 0)), dtype=dtype, device=device)
    return qkvz[:, :qkvz_width], ba[:, : 2 * heads], width, heads, dim


def _check(outputs, qkvz, ba, width, heads, dim):
    qkv, z, b, a, core = outputs
    torch.testing.assert_close(qkv, qkvz[:, :width], rtol=0, atol=0)
    torch.testing.assert_close(z, qkvz[:, width:].reshape(qkvz.shape[0], heads, dim), rtol=0, atol=0)
    torch.testing.assert_close(b, ba[:, :heads], rtol=0, atol=0)
    torch.testing.assert_close(a, ba[:, heads:], rtol=0, atol=0)
    torch.testing.assert_close(core, torch.zeros_like(core), rtol=0, atol=0)
    assert all(value.is_contiguous() for value in (qkv, b, a, core))
    if qkvz.shape[0]:
        assert z.untyped_storage().data_ptr() == qkvz.untyped_storage().data_ptr()
        assert z.storage_offset() == qkvz.storage_offset() + width


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens,tp_size", [(0, 2), (1, 2), (4, 2), (17, 2), (128, 2), (4, 1), (4, 4), (4, 8), (4, 16)])
def test_qwen36_preparation_bit_exact(tokens, tp_size, dtype):
    qkvz, ba, width, heads, dim = _inputs(tokens, tp_size, dtype)
    outputs = fused_gdn_prepare(qkvz, ba, width, heads, dim)
    _check(outputs, qkvz, ba, width, heads, dim)
    supplied = torch.full_like(outputs[-1], float("nan"))
    outputs = fused_gdn_prepare(qkvz, ba, width, heads, dim, supplied)
    assert outputs[-1].data_ptr() == supplied.data_ptr()
    _check(outputs, qkvz, ba, width, heads, dim)


@pytest.mark.parametrize("device", ["npu:0", "npu:1"])
def test_row_strides_offsets_and_mixed_ba_dtype(device):
    if torch.npu.device_count() <= int(device.split(":")[1]):
        pytest.skip("second die is not visible")
    qkvz, ba, width, heads, dim = _inputs(18, 2, torch.bfloat16, True, device)
    qkvz, ba = qkvz[1::2], ba[1::2].float()
    with torch.npu.device(device):
        outputs = fused_gdn_prepare(qkvz, ba, width, heads, dim)
        _check(outputs, qkvz, ba, width, heads, dim)
        assert outputs[2].dtype == torch.float32


def test_fullgraph_compile_preserves_values_and_alias():
    qkvz, ba, width, heads, dim = _inputs(4, 2, torch.bfloat16)
    compiled = torch.compile(fused_gdn_prepare, backend="eager", fullgraph=True)
    for offset in (0, 1):
        qkvz.add_(offset)
        outputs = compiled(qkvz, ba, width, heads, dim)
        _check(outputs, qkvz, ba, width, heads, dim)


def test_aclgraph_replay_reinitializes_padding():
    qkvz, ba, width, heads, dim = _inputs(4, 2, torch.bfloat16)
    supplied = torch.empty((4, heads, dim), dtype=qkvz.dtype, device=qkvz.device)

    def step():
        outputs = fused_gdn_prepare(qkvz, ba, width, heads, dim, supplied)
        # Mimic a core writing only actual tokens inside a padded graph batch.
        outputs[-1][:2].fill_(3)
        return outputs

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            step()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        outputs = step()
    for _ in range(2):
        supplied.fill_(float("nan"))
        qkvz.add_(1)
        ba.add_(1)
        graph.replay()
        torch.npu.synchronize()
        qkv, z, b, a, core = outputs
        torch.testing.assert_close(qkv, qkvz[:, :width], rtol=0, atol=0)
        torch.testing.assert_close(b, ba[:, :heads], rtol=0, atol=0)
        torch.testing.assert_close(a, ba[:, heads:], rtol=0, atol=0)
        torch.testing.assert_close(z, qkvz[:, width:].reshape(4, heads, dim), rtol=0, atol=0)
        assert torch.all(core[:2] == 3)
        assert torch.count_nonzero(core[2:]) == 0
