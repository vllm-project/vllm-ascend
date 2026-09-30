# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.ops.kda as kda


@pytest.fixture
def fla_module(monkeypatch):
    module = ModuleType("fla_npu.ops.ascendc")
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", module)
    kda._get_fla_kda_ops.cache_clear()
    yield module
    kda._get_fla_kda_ops.cache_clear()


def test_resolves_public_fla_entrypoints(fla_module):
    fla_module.chunk_kda_fwd = MagicMock()
    fla_module.recurrent_kda = MagicMock()

    assert kda._get_fla_kda_ops() == (fla_module.chunk_kda_fwd, fla_module.recurrent_kda)


@pytest.mark.parametrize("missing", ["package", "chunk_kda_fwd", "recurrent_kda"])
def test_missing_fla_dependency_reports_installation_hint(fla_module, monkeypatch, missing):
    if missing == "package":
        monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", None)
    else:
        setattr(fla_module, "recurrent_kda" if missing == "chunk_kda_fwd" else "chunk_kda_fwd", MagicMock())

    with pytest.raises(ImportError, match="Build and install a compatible wheel"):
        kda._get_fla_kda_ops()


@pytest.mark.parametrize("as_tensor", [False, True])
def test_chunk_passes_host_metadata_and_normalized_inputs(monkeypatch, as_tensor):
    q, k, v, gate = (torch.randn(1, 3, 2, 256)[..., :128] for _ in range(4))
    beta = torch.rand(1, 3, 2)
    state = torch.zeros(2, 2, 128, 128)
    cu = torch.tensor([0, 1, 3]) if as_tensor else (0, 1, 3)
    chunks = torch.tensor([[0, 0], [1, 0]]) if as_tensor else (0, 0, 1, 0)
    normalized = [torch.full_like(q, 0.1), torch.full_like(k, 0.2)]
    normalize = MagicMock(side_effect=normalized)
    final_state = torch.ones_like(state)
    output = torch.empty_like(v)

    def chunk(q_arg, k_arg, v_arg, g_arg, beta_arg, scale, chunk_size, **kwargs):
        # Match the launcher's host-sequence truth test; multi-element tensors fail here.
        assert kwargs["cu_seqlens"] and kwargs["chunk_indices"]
        assert list(kwargs["cu_seqlens"]) == [0, 1, 3]
        assert list(kwargs["chunk_indices"]) == [0, 0, 1, 0]
        assert q_arg is normalized[0] and k_arg is normalized[1]
        assert v_arg.is_contiguous() and g_arg.is_contiguous()
        torch.testing.assert_close(v_arg, v)
        torch.testing.assert_close(g_arg, gate)
        assert beta_arg is beta
        assert scale == 128**-0.5 and chunk_size == 64
        assert kwargs["initial_state"] is state
        assert kwargs["state_v_first"] and kwargs["output_final_state"]
        assert kwargs["use_gate_in_kernel"]
        assert not kwargs["disable_recompute"] and not kwargs["return_intermediate_states"]
        return output, final_state, *([None] * 10)

    monkeypatch.setattr(kda, "_get_fla_kda_ops", lambda: (chunk, None))
    monkeypatch.setattr(kda, "l2norm_fwd", normalize)
    actual, actual_state = kda.run_chunk_kda(
        q, k, v, gate, beta, state, cu, chunks, torch.zeros(2), torch.zeros(256), lower_bound=None
    )
    assert actual is output and actual_state is final_state
    assert normalize.call_count == 2
    assert all(call.args[0].is_contiguous() for call in normalize.call_args_list)


def test_chunk_metadata_does_not_copy_device_tensors_to_host():
    with pytest.raises(ValueError, match="prepared on CPU"):
        kda._host_metadata(torch.empty(3, device="meta", dtype=torch.int64))
