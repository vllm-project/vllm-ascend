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


@pytest.mark.parametrize("tokens", [4, 1025])
def test_aclgraph_replay_reinitializes_padding(tokens):
    qkvz, ba, width, heads, dim = _inputs(tokens, 2, torch.bfloat16)
    supplied = torch.empty((tokens, heads, dim), dtype=qkvz.dtype, device=qkvz.device)

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
        torch.testing.assert_close(z, qkvz[:, width:].reshape(tokens, heads, dim), rtol=0, atol=0)
        assert torch.all(core[:2] == 3)
        assert torch.count_nonzero(core[2:]) == 0


@pytest.mark.parametrize("tokens", [4, 16393])
def test_aot_prefill_avoids_cloning_fresh_outputs(tokens):
    from torch._dynamo.backends.common import aot_autograd

    qkvz, ba, width, heads, dim = _inputs(tokens, 1, torch.bfloat16)
    backend = aot_autograd(fw_compiler=lambda graph, _: graph.forward)
    compiled = torch.compile(fused_gdn_prepare, backend=backend, fullgraph=True)
    outputs = compiled(qkvz, ba, width, heads, dim)
    torch.npu.synchronize()
    _check(outputs, qkvz, ba, width, heads, dim)

    qkvz.add_(1)
    ba.add_(1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        outputs = compiled(qkvz, ba, width, heads, dim)
        torch.npu.synchronize()
    _check(outputs, qkvz, ba, width, heads, dim)
    clone_count = sum(event.count for event in prof.key_averages() if event.key == "aten::clone")
    assert clone_count == 0, f"AOT cloned {clone_count} fresh prepare outputs"


@pytest.mark.parametrize("tokens", [1023, 1024, 1025, 32768])
def test_large_prefill_boundary_and_tail(tokens):
    qkvz, ba, width, heads, dim = _inputs(tokens, 1, torch.bfloat16, True)
    ba = ba.float()
    supplied = torch.full((tokens, heads, dim), float("nan"), dtype=qkvz.dtype, device=qkvz.device)
    outputs = fused_gdn_prepare(qkvz, ba, width, heads, dim, supplied)
    assert outputs[-1].data_ptr() == supplied.data_ptr()
    _check(outputs, qkvz, ba, width, heads, dim)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_prefill_strided_tokens(tp_size, dtype):
    qkvz, ba, width, heads, dim = _inputs(2051, tp_size, dtype, True)
    qkvz, ba = qkvz[1::2], ba.float()[1::2]
    _check(fused_gdn_prepare(qkvz, ba, width, heads, dim), qkvz, ba, width, heads, dim)


def test_large_prefill_non_power_of_two_dimensions():
    tokens, width, heads, dim = 1025, 1277, 7, 33
    qkvz = torch.randn((tokens, width + heads * dim), dtype=torch.bfloat16, device="npu:0")
    ba = torch.randn((tokens, 2 * heads), dtype=torch.float32, device=qkvz.device)
    _check(fused_gdn_prepare(qkvz, ba, width, heads, dim), qkvz, ba, width, heads, dim)
