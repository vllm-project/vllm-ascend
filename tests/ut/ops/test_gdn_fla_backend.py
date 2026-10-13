# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend import envs
from vllm_ascend.device import device_op
from vllm_ascend.ops import gdn


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("ops_available", [False, True])
def test_prefill_metadata_backend_selection(monkeypatch, enabled, supported, available, ops_available):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLA_GDN", str(int(enabled)))
    monkeypatch.setattr(gdn, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: supported))
    fla = Mock(return_value=object() if available else None)
    monkeypatch.setattr(gdn, "_get_fla_gdn_prefill_op", fla)
    monkeypatch.setattr(gdn.AscendGatedDeltaNetAttention, "_probe_fused_chunk", lambda: ops_available)
    assert gdn.AscendGatedDeltaNetAttention._supports_host_metadata_prefill() == (
        (enabled and supported and available) or ops_available
    )
    assert fla.call_count == int(enabled and supported)


def test_fla_environment_defaults_and_validation(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_ENABLE_FLA_GDN", raising=False)
    monkeypatch.delenv("VLLM_ASCEND_FLA_CHUNK_SIZE", raising=False)
    assert envs.VLLM_ASCEND_ENABLE_FLA_GDN is True
    assert envs.VLLM_ASCEND_FLA_CHUNK_SIZE == 128
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FLA_GDN", "2")
    with pytest.raises(ValueError, match="must be either"):
        _ = envs.VLLM_ASCEND_ENABLE_FLA_GDN
    monkeypatch.setenv("VLLM_ASCEND_FLA_CHUNK_SIZE", "32")
    with pytest.raises(ValueError, match="must be 64 or 128"):
        device_op._fla_gdn_chunk_metadata(SimpleNamespace())


@pytest.mark.parametrize("adaptor", [device_op.BaseDeviceAdaptor, device_op.A5DeviceAdaptor])
@pytest.mark.parametrize("chunk_size", [64, 128])
@pytest.mark.parametrize("lengths", [(1, 65, 257), (0, 65, 0, 257, 0), (0, 0)])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_fla_layout_chunks_and_empty_states(monkeypatch, adaptor, chunk_size, lengths, state_dtype):
    monkeypatch.setenv("VLLM_ASCEND_FLA_CHUNK_SIZE", str(chunk_size))
    monkeypatch.setattr(device_op, "l2norm_fwd", lambda x: torch.nn.functional.normalize(x, dim=-1))
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    keep = [i for i, length in enumerate(lengths) if length]
    compact = tuple([0] + [cu[i + 1] for i in keep])
    has_empty = len(keep) != len(lengths)
    meta = SimpleNamespace(
        cu_seqlens_host=tuple(cu),
        cu_seqlens_kern=compact if has_empty else None,
        keep_meta=torch.tensor(keep, dtype=torch.long) if has_empty else None,
    )
    # Unequal K/V dimensions expose accidental state transposes in the adapter.
    q = torch.randn(1, cu[-1], 2, 3)
    k = torch.randn_like(q)
    v = torch.randn(1, cu[-1], 2, 5)
    g = -torch.rand(1, cu[-1], 2)
    beta = torch.rand_like(g)
    state = torch.randn(len(lengths), 2, 5, 3).to(state_dtype)
    original = state.clone()
    a5 = adaptor is device_op.A5DeviceAdaptor

    def fused(q_arg, k_arg, v_arg, g_arg, beta_arg, **kwargs):
        assert kwargs["chunk_size"] == chunk_size
        assert kwargs["cu_seqlens"] == compact
        expected_indices = []
        for seq, length in enumerate(length for length in lengths if length):
            for chunk in range((length + chunk_size - 1) // chunk_size):
                expected_indices.extend((seq, chunk))
        assert kwargs["chunk_indices"] == tuple(expected_indices)
        assert kwargs["layout"] == ("BSND" if a5 else "BNSD")
        assert kwargs["state_v_first"] == a5
        assert kwargs["use_qk_l2norm_in_kernel"] == a5
        expected_q = q if a5 else torch.nn.functional.normalize(q, dim=-1).transpose(1, 2)
        torch.testing.assert_close(q_arg, expected_q)
        expected_state = state[keep] if a5 else state[keep].transpose(-1, -2)
        torch.testing.assert_close(kwargs["initial_state"], expected_state)
        # A different output dtype exercises state scatter conversion.
        return v.clone(), (expected_state.float() + 1).to(torch.bfloat16), None

    op = Mock(side_effect=fused)
    output, final = adaptor.fla_gdn_prefill(q, k, v, g, beta, state, 3**-0.5, meta, op)
    torch.testing.assert_close(state, original)
    assert output.shape == v.shape
    if cu[-1] == 0:
        op.assert_not_called()
        torch.testing.assert_close(final, original)
        assert final.data_ptr() != state.data_ptr()
    else:
        op.assert_called_once()
        torch.testing.assert_close(output, v)
        expected = original.clone()
        expected[keep] = (original[keep].float() + 1).to(torch.bfloat16).to(state_dtype)
        torch.testing.assert_close(final.to(state_dtype), expected)
