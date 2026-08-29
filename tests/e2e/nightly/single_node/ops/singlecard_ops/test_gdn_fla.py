# SPDX-License-Identifier: Apache-2.0
"""Cross-SoC smoke coverage for the Qwen GDN fla_npu adapter."""

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.device.device_config import get_fla_gdn_soc, is_fla_gdn_supported
from vllm_ascend.ops.gdn_fla import (
    FlaGDNAdapter,
    GDNPrefillMetadata,
    GDNRuntimeSignature,
    parse_gdn_backend_config,
)
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices

pytestmark = pytest.mark.skipif(
    not is_fla_gdn_supported(), reason="requires A2/A3/A5 FLA GDN support"
)
torch_npu.npu.set_compile_mode(jit_compile=False)


def _assert_output_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    cosine = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
    assert cosine.item() >= 0.999
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)


def _adapter(mode: str, overrides: str = "") -> FlaGDNAdapter:
    soc = get_fla_gdn_soc()
    assert soc is not None
    return FlaGDNAdapter(
        parse_gdn_backend_config(mode, overrides),
        GDNRuntimeSignature(
            soc=soc,
            dtype="bfloat16",
            state_dtype="float32",
            num_key_heads=1,
            num_value_heads=2,
            key_dim=128,
            value_dim=128,
        ),
        layer_name="smoke.linear_attn",
        is_supported_soc=True,
    )


def _prefill_metadata(tokens: int) -> GDNPrefillMetadata:
    return _prefill_metadata_from_cu(torch.tensor([0, tokens], dtype=torch.int64))


def _prefill_metadata_from_cu(cu_cpu: torch.Tensor) -> GDNPrefillMetadata:
    chunk64_cpu = prepare_chunk_indices(cu_cpu, 64)
    return GDNPrefillMetadata(
        cu_seqlens=cu_cpu.npu(),
        cu_seqlens_host=tuple(int(value) for value in cu_cpu.tolist()),
        chunk_indices=chunk64_cpu.npu(),
        chunk_indices_host=tuple(int(value) for value in chunk64_cpu.flatten().tolist()),
        block_indices_cumsum=prepare_chunk_indices(cu_cpu, 2048).npu(),
        chunk_indices_large_block=prepare_chunk_indices(cu_cpu, 1216).npu(),
    )


def _prefill_inputs(tokens: int):
    torch.manual_seed(7)
    q = torch.randn((1, tokens, 1, 128), dtype=torch.bfloat16, device="npu")
    return {
        "q": q,
        "k": torch.randn_like(q),
        "v": torch.randn((1, tokens, 2, 128), dtype=torch.bfloat16, device="npu"),
        "g": F.logsigmoid(torch.randn((1, tokens, 2), dtype=torch.float32, device="npu")),
        "beta": torch.sigmoid(torch.randn((1, tokens, 2), dtype=torch.bfloat16, device="npu")),
        "initial_state": torch.randn((1, 2, 128, 128), dtype=torch.float32, device="npu"),
        "has_initial_state": torch.tensor([True], dtype=torch.bool, device="npu"),
        "scale": 128**-0.5,
        "metadata": _prefill_metadata(tokens),
    }


@pytest.mark.parametrize("tokens", [1, 63, 64, 65])
def test_gdn_fla_prefill_matches_native(tokens):
    kwargs = _prefill_inputs(tokens)

    native_output, native_state = _adapter("native").prefill(**kwargs)
    fla_output, fla_state = _adapter("fla_npu").prefill(**kwargs)
    torch.npu.synchronize()

    assert torch.isfinite(fla_output.float()).all()
    assert torch.isfinite(fla_state.float()).all()
    _assert_output_close(fla_output, native_output)
    torch.testing.assert_close(fla_state, native_state, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize(
    "operator",
    [
        "chunk_local_cumsum",
        "chunk_scaled_dot_kkt",
        "solve_tri",
        "recompute_w_u_fwd",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_fwd_o",
    ],
)
def test_gdn_fla_each_prefill_replacement_matches_native(operator):
    kwargs = _prefill_inputs(65)
    native_output, native_state = _adapter("native").prefill(**kwargs)
    candidate_output, candidate_state = _adapter("native", f"{operator}=fla_npu").prefill(**kwargs)
    torch.npu.synchronize()

    _assert_output_close(candidate_output, native_output)
    torch.testing.assert_close(candidate_state, native_state, rtol=5e-3, atol=5e-3)


def test_gdn_fla_varlen_multiple_sequences_matches_native():
    kwargs = _prefill_inputs(64)
    kwargs["initial_state"] = torch.randn((2, 2, 128, 128), dtype=torch.float32, device="npu")
    kwargs["has_initial_state"] = torch.tensor([False, True], dtype=torch.bool, device="npu")
    kwargs["metadata"] = _prefill_metadata_from_cu(torch.tensor([0, 1, 64], dtype=torch.int64))

    native_output, native_state = _adapter("native").prefill(**kwargs)
    fla_output, fla_state = _adapter("fla_npu").prefill(**kwargs)
    torch.npu.synchronize()

    _assert_output_close(fla_output, native_output)
    torch.testing.assert_close(fla_state, native_state, rtol=5e-3, atol=5e-3)


def test_gdn_fla_causal_conv_matches_native_cache_update():
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
    native_adapter = _adapter("native")
    fla_adapter = _adapter("native", "causal_conv1d=fla_npu")

    native_output = native_adapter.causal_conv1d(conv_state=native_state, **kwargs)
    fla_output = fla_adapter.causal_conv1d(conv_state=fla_state, **kwargs)
    kwargs["x"] = x * 0.5
    native_output_next = native_adapter.causal_conv1d(conv_state=native_state, **kwargs)
    fla_output_next = fla_adapter.causal_conv1d(conv_state=fla_state, **kwargs)
    torch.npu.synchronize()

    _assert_output_close(fla_output, native_output)
    _assert_output_close(fla_output_next, native_output_next)
    torch.testing.assert_close(fla_state, native_state, rtol=5e-3, atol=5e-3)


def test_gdn_fla_ordinary_decode_matches_native_reference():
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
    native_adapter = _adapter("native")
    auto_adapter = _adapter("auto")

    native_output = native_adapter.decode(state=native_state, **kwargs)
    auto_output = auto_adapter.decode(state=auto_state, **kwargs)
    kwargs["q"] = kwargs["q"] * 0.5
    native_output_next = native_adapter.decode(state=native_state, **kwargs)
    auto_output_next = auto_adapter.decode(state=auto_state, **kwargs)
    torch.npu.synchronize()

    _assert_output_close(auto_output, native_output)
    _assert_output_close(auto_output_next, native_output_next)
    torch.testing.assert_close(auto_state, native_state, rtol=5e-3, atol=5e-3)
