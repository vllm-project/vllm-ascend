# SPDX-License-Identifier: Apache-2.0
"""A5 smoke coverage for the Qwen GDN fla_npu adapter."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.device.device_config import is_950
from vllm_ascend.ops.gdn_a5 import (
    A5GDNAdapter,
    GDNPrefillMetadata,
    GDNRuntimeSignature,
    parse_gdn_backend_config,
)
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices

pytestmark = pytest.mark.skipif(not is_950(), reason="A5-only GDN operator smoke")
torch_npu.npu.set_compile_mode(jit_compile=False)


def _adapter(mode: str) -> A5GDNAdapter:
    return A5GDNAdapter(
        parse_gdn_backend_config(mode, ""),
        GDNRuntimeSignature(
            soc="ascend950",
            dtype="bfloat16",
            state_dtype="float32",
            num_key_heads=1,
            num_value_heads=2,
            key_dim=128,
            value_dim=128,
        ),
        layer_name="smoke.linear_attn",
        is_a5=True,
    )


def _prefill_metadata(tokens: int) -> GDNPrefillMetadata:
    cu_cpu = torch.tensor([0, tokens], dtype=torch.int64)
    chunk64_cpu = prepare_chunk_indices(cu_cpu, 64)
    return GDNPrefillMetadata(
        cu_seqlens=cu_cpu.npu(),
        cu_seqlens_host=(0, tokens),
        chunk_indices=chunk64_cpu.npu(),
        chunk_indices_host=tuple(int(value) for value in chunk64_cpu.flatten().tolist()),
        block_indices_cumsum=prepare_chunk_indices(cu_cpu, 2048).npu(),
        chunk_indices_large_block=prepare_chunk_indices(cu_cpu, 1216).npu(),
    )


@pytest.mark.parametrize("tokens", [1, 63, 64, 65])
def test_gdn_a5_prefill_fla_matches_native(tokens):
    torch.manual_seed(7)
    q = torch.randn((1, tokens, 1, 128), dtype=torch.bfloat16, device="npu")
    k = torch.randn_like(q)
    v = torch.randn((1, tokens, 2, 128), dtype=torch.bfloat16, device="npu")
    g = F.logsigmoid(torch.randn((1, tokens, 2), dtype=torch.float32, device="npu"))
    beta = torch.sigmoid(torch.randn((1, tokens, 2), dtype=torch.bfloat16, device="npu"))
    state = torch.randn((1, 2, 128, 128), dtype=torch.float32, device="npu")
    has_state = torch.tensor([True], dtype=torch.bool, device="npu")
    metadata = _prefill_metadata(tokens)
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": state,
        "has_initial_state": has_state,
        "scale": 128**-0.5,
        "metadata": metadata,
    }

    native_output, native_state = _adapter("native").prefill(**kwargs)
    fla_output, fla_state = _adapter("fla_npu").prefill(**kwargs)
    torch.npu.synchronize()

    assert torch.isfinite(fla_output.float()).all()
    assert torch.isfinite(fla_state.float()).all()
    torch.testing.assert_close(fla_output, native_output, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(fla_state, native_state, rtol=5e-3, atol=5e-3)


def test_gdn_a5_causal_conv_fla_matches_native_cache_update():
    torch.manual_seed(11)
    tokens, channels, width = 5, 16, 4
    x = torch.randn((tokens, channels), dtype=torch.bfloat16, device="npu")
    weight = torch.randn((width, channels), dtype=torch.bfloat16, device="npu")
    bias = torch.randn((channels,), dtype=torch.bfloat16, device="npu")
    state = torch.randn((1, width, channels), dtype=torch.bfloat16, device="npu")
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device="npu")
    cache_indices = torch.tensor([0], dtype=torch.int32, device="npu")
    initial_state_mode = torch.tensor([1], dtype=torch.int32, device="npu")
    kwargs = {
        "x": x,
        "weight": weight,
        "bias": bias,
        "query_start_loc": query_start_loc,
        "cache_indices": cache_indices,
        "initial_state_mode": initial_state_mode,
        "activation_mode": 1,
        "pad_slot_id": -1,
        "run_mode": 0,
    }
    native_state = state.clone()
    fla_state = state.clone()

    native_output = _adapter("native").causal_conv1d(conv_state=native_state, **kwargs)
    fla_output = _adapter("fla_npu").causal_conv1d(conv_state=fla_state, **kwargs)
    torch.npu.synchronize()

    torch.testing.assert_close(fla_output, native_output, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(fla_state, native_state, rtol=5e-3, atol=5e-3)


def test_gdn_a5_ordinary_decode_preserves_native_recurrent_path():
    torch.manual_seed(13)
    state = torch.randn((2, 2, 128, 128), dtype=torch.float32, device="npu")
    kwargs = {
        "q": torch.randn((1, 2, 1, 128), dtype=torch.bfloat16, device="npu"),
        "k": torch.randn((1, 2, 1, 128), dtype=torch.bfloat16, device="npu"),
        "v": torch.randn((1, 2, 2, 128), dtype=torch.bfloat16, device="npu"),
        "g": F.logsigmoid(torch.randn((1, 2, 2), dtype=torch.float32, device="npu")),
        "beta": torch.sigmoid(torch.randn((1, 2, 2), dtype=torch.bfloat16, device="npu")),
        "scale": 128**-0.5,
        "actual_seq_lengths": torch.tensor([0, 1, 1], dtype=torch.int32, device="npu"),
        "ssm_state_indices": torch.tensor([0, 1], dtype=torch.int32, device="npu"),
    }
    native_state = state.clone()
    auto_state = state.clone()

    native_output = _adapter("native").decode(state=native_state, **kwargs)
    auto_output = _adapter("auto").decode(state=auto_state, **kwargs)
    torch.npu.synchronize()

    torch.testing.assert_close(auto_output, native_output, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(auto_state, native_state, rtol=5e-3, atol=5e-3)
