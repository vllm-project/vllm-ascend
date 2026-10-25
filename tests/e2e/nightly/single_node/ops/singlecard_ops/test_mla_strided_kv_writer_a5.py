# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 wide-draft writer parity against the original dense KV writer."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

BLOCK_SIZE = 128
NOPE_DIM = 512
ROPE_DIM = 64
FUSED_DIM = NOPE_DIM + ROPE_DIM
NUM_BLOCKS = 4


@pytest.fixture
def impl():
    if not torch.npu.is_available() or not get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
        pytest.skip("The strided wide-draft adapter requires A5.")
    writer = AscendMLAImpl.__new__(AscendMLAImpl)
    writer.use_mla_rope = True
    writer.qk_rope_head_dim = ROPE_DIM
    writer.kv_lora_rank = NOPE_DIM
    writer.q_lora_rank = 1536
    writer.num_kv_heads = 1
    writer.dtype = torch.bfloat16
    writer.fa_quant_layer = False
    writer.enable_kv_nz = False
    writer.support_fp8_attention = True
    writer.is_draft_model = True
    writer.pcp_enabled = False
    writer.dcp_enabled = False
    writer.is_pcp_decode_sharded = False
    writer.fused_qkv_a_proj = SimpleNamespace(input_size=14336, weight=torch.empty(1, device="npu"))
    writer.q_proj = SimpleNamespace(_chunk_size=0)
    writer.q_a_layernorm = object()
    writer.kv_a_layernorm = SimpleNamespace(
        weight=torch.linspace(0.5, 1.5, NOPE_DIM, device="npu", dtype=torch.bfloat16), variance_epsilon=1e-6
    )
    writer.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=257),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 128, 257]),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
    )
    writer._strided_kv_workspace = None
    writer._prepare_strided_kv_workspace()
    assert not writer.supports_prolog_v3_writer()
    assert writer.supports_strided_mla_writer()
    return writer


def make_cache(token_fused):
    if token_fused:
        backing = torch.full((NUM_BLOCKS, BLOCK_SIZE, 1, FUSED_DIM), -7, dtype=torch.bfloat16, device="npu")
        return backing[..., :NOPE_DIM], backing[..., NOPE_DIM:]
    stride = 2 * BLOCK_SIZE * FUSED_DIM
    backing = torch.full((NUM_BLOCKS * stride,), -7, dtype=torch.bfloat16, device="npu")
    return (
        backing.as_strided((NUM_BLOCKS, BLOCK_SIZE, 1, NOPE_DIM), (stride, NOPE_DIM, NOPE_DIM, 1)),
        backing.as_strided(
            (NUM_BLOCKS, BLOCK_SIZE, 1, ROPE_DIM), (stride, ROPE_DIM, ROPE_DIM, 1), BLOCK_SIZE * NOPE_DIM
        ),
    )


def make_inputs(num_tokens):
    kv = torch.randn(num_tokens, FUSED_DIM, dtype=torch.bfloat16, device="npu")
    angles = torch.randn(num_tokens, 1, 1, ROPE_DIM, device="npu")
    slots = torch.arange(num_tokens, dtype=torch.int64, device="npu") + BLOCK_SIZE - 1
    if num_tokens > 1:
        slots[-1] = -1
    return kv, angles.cos().to(torch.bfloat16), angles.sin().to(torch.bfloat16), slots


def dense_reference(impl, kv, cos, sin, slots):
    nope = torch.full((NUM_BLOCKS, BLOCK_SIZE, 1, NOPE_DIM), -7, dtype=torch.bfloat16, device="npu")
    rope = torch.full((NUM_BLOCKS, BLOCK_SIZE, 1, ROPE_DIM), -7, dtype=torch.bfloat16, device="npu")
    _, _, current_rope, current_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
        kv.view(-1, 1, 1, FUSED_DIM),
        impl.kv_a_layernorm.weight,
        cos,
        sin,
        slots,
        rope,
        nope,
        epsilon=impl.kv_a_layernorm.variance_epsilon,
        cache_mode="PA",
        is_output_kv=True,
    )
    return (nope, rope), (current_rope.reshape(-1, 1, ROPE_DIM), current_nope.reshape(-1, 1, NOPE_DIM))


@pytest.mark.parametrize("token_fused", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 128, 129, 257])
@pytest.mark.parametrize("prefill", [False, True])
def test_wide_draft_cache_matches_dense_writer(impl, token_fused, num_tokens, prefill):
    cache = make_cache(token_fused)
    inputs = make_inputs(num_tokens)
    expected_cache, expected_current = dense_reference(impl, *inputs)
    if prefill:
        current = impl.exec_kv_prefill(*inputs[:3], cache, inputs[3])
    else:
        history_rope, history_nope, *current = impl.exec_kv_decode(
            *inputs[:3], cache, inputs[3], return_current_kv=True
        )
        assert history_nope is cache[0] and history_rope is cache[1]
    for actual, expected in zip(cache, expected_cache):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    # Padding slots need not produce current outputs in the original writer.
    valid_tokens = num_tokens - (num_tokens > 1)
    for actual, expected in zip(current, expected_current):
        torch.testing.assert_close(actual[:valid_tokens].cpu(), expected[:valid_tokens].cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("token_fused", [False, True])
def test_wide_draft_graph_replay_updates_original_cache(impl, token_fused):
    cache = make_cache(token_fused)
    kv, cos, sin, slots = make_inputs(129)
    workspace = impl._strided_kv_workspace
    assert workspace is not None
    addresses = tuple(tensor.data_ptr() for tensor in workspace)
    impl._exec_strided_kv_writer(kv, cos, sin, cache, slots)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        impl._exec_strided_kv_writer(kv, cos, sin, cache, slots)
    kv.mul_(0.75)
    cos.mul_(0.5)
    slots[:-1].add_(1)
    for tensor in cache:
        tensor.fill_(-7)
    expected_cache, _ = dense_reference(impl, kv, cos, sin, slots)
    graph.replay()
    torch.npu.synchronize()
    for actual, expected in zip(cache, expected_cache):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    impl._prepare_strided_kv_workspace()
    assert impl._strided_kv_workspace is workspace
    assert tuple(tensor.data_ptr() for tensor in workspace) == addresses
