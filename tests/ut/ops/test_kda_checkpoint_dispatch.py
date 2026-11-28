# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA checkpoint capability and cache contracts, using CPU tensors."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import kda as mod


def _layer(heads=2, enabled=True):
    return NS(
        _ascend_internal_checkpoint_enabled=enabled,
        _ascend_kda_checkpoint_spec=((3, heads, 4, 4), torch.float32),
        local_num_heads=heads,
        head_dim=4,
    )


@pytest.fixture(autouse=True)
def workspace_manager(monkeypatch):
    manager = NS(get_simultaneous=Mock(side_effect=lambda spec: (torch.empty(spec[0], dtype=spec[1]),)))
    monkeypatch.setattr("vllm.v1.worker.workspace.current_workspace_manager", lambda: manager)
    return manager


def test_profile_reserves_upstream_workspace_layout(workspace_manager):
    first, second, other, off = _layer(), _layer(), _layer(3), _layer(enabled=False)
    for layer in (first, second, other, off):
        mod.ensure_kda_checkpoint_workspace(layer)
    assert workspace_manager.get_simultaneous.call_count == 3
    calls = workspace_manager.get_simultaneous.call_args_list
    assert calls[0].args == (((3, 2, 4, 4), torch.float32),)
    assert calls[1].args == calls[0].args
    assert calls[2].args == (((3, 3, 4, 4), torch.float32),)


def _inputs():
    # A NULL slot's device offset is zero while its host plan still has 64.
    checkpoint = NS(
        checkpoint_offsets=torch.tensor([64, 0], dtype=torch.int32),
        offsets=[64, 64],
        query_lens=[65, 129],
        state_indices=torch.tensor([2, 0]),
    )
    chunk = NS(
        cu_seqlens_host=(0, 65, 194), cu_seqlens_kern=None, chunk_indices_chunk64_host=(0, 0, 0, 1, 1, 0, 1, 1, 1, 2)
    )
    metadata = NS(
        checkpoint=checkpoint,
        prefill_query_start_loc=torch.tensor([0, 65, 194], dtype=torch.int32),
        non_spec_prefill_metadata=NS(chunk=chunk),
    )
    layer = _layer()
    mod.ensure_kda_checkpoint_workspace(layer)
    # Preserve padded page strides; export must not copy the whole cache.
    layer.kv_cache = (torch.empty(7, 3, 24)[..., :12], torch.empty(7, 2, 4, 4))
    layer.A_log, layer.dt_bias = torch.zeros(2), torch.zeros(8)
    layer.gate_lower_bound = -5.0
    layer._checkpoint_exporter = NS(export=Mock())
    return layer, metadata, torch.empty(194, 12), torch.zeros(2, 2, 4, 4)


def test_prefill_passes_masked_offsets_and_padded_cache_views(monkeypatch):
    layer, metadata, raw, initial = _inputs()
    output, final = torch.empty(1), torch.ones_like(initial)
    backend = Mock(return_value=(output, final))
    monkeypatch.setattr(mod, "run_chunk_kda", backend)
    result, state = mod.run_kda_checkpoint_prefill(layer, metadata, raw, *(torch.empty(1) for _ in range(5)), initial)
    assert result is output and state is final
    assert backend.call_count == 1
    assert backend.call_args.kwargs["checkpoint_offsets"] is metadata.checkpoint.checkpoint_offsets
    scratch = backend.call_args.kwargs["checkpoint_state"]
    assert scratch.shape == initial.shape and scratch.dtype == torch.float32
    exported = layer._checkpoint_exporter.export.call_args
    assert exported.args[0] is metadata.checkpoint
    assert exported.kwargs["raw_qkv"] is raw
    assert exported.kwargs["recurrent_checkpoint"] is scratch
    assert exported.kwargs["recurrent_state"] is layer.kv_cache[1]
    assert exported.kwargs["conv_state"].data_ptr() == layer.kv_cache[0].data_ptr()
    assert exported.kwargs["conv_state"].stride() == layer.kv_cache[0].transpose(-1, -2).stride()
    assert exported.kwargs["cu_seqlens"] is metadata.prefill_query_start_loc


@pytest.mark.parametrize("offset,length", [(1, 65), (64, 64), (128, 65), (-64, 65)])
def test_invalid_host_boundary_never_launches_or_exports(monkeypatch, offset, length):
    layer, metadata, raw, initial = _inputs()
    metadata.checkpoint.offsets[0], metadata.checkpoint.query_lens[0] = offset, length
    backend = Mock()
    monkeypatch.setattr(mod, "run_chunk_kda", backend)
    with pytest.raises(ValueError, match="internal chunk64 boundary"):
        mod.run_kda_checkpoint_prefill(layer, metadata, raw, *(torch.empty(1) for _ in range(5)), initial)
    backend.assert_not_called()
    layer._checkpoint_exporter.export.assert_not_called()


def test_disabled_capability_does_not_advertise_upstream_cuda_support(monkeypatch):
    layer = NS(_checkpoint_alignment=16)
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_ascend_config", lambda: NS(kda_internal_prefill_checkpoints=False)
    )
    mod.initialize_kda_checkpoints(layer, NS())
    assert layer._checkpoint_alignment is None
    assert not layer._ascend_internal_checkpoint_enabled


def test_enabled_capability_uses_explicit_kimi_config(monkeypatch, workspace_manager):
    # Kimi does not inherit Qwen GDN's layer.vllm_config member.
    import vllm_ascend.device.hardware_profile as hardware

    layer = NS(head_dim=128, local_num_heads=2, conv_size=4, get_state_dtype=lambda: (torch.bfloat16, torch.float32))
    config = NS(
        cache_config=NS(mamba_cache_mode="align"),
        model_config=NS(dtype=torch.bfloat16),
        parallel_config=NS(use_ubatching=False),
        scheduler_config=NS(max_num_seqs=3),
    )
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_ascend_config", lambda: NS(kda_internal_prefill_checkpoints=True)
    )
    monkeypatch.setattr(hardware, "HardwareCapability", NS(KDA_INTERNAL_PREFILL_CHECKPOINT=object()))
    monkeypatch.setattr(hardware, "get_current_hardware_profile", lambda: NS(supports=lambda _: True))
    monkeypatch.setattr("vllm.v1.worker.workspace.is_workspace_manager_initialized", lambda: True)

    def native_backend(*args, checkpoint_offsets=None, checkpoint_state=None):
        raise AssertionError("initialization must not launch the operator")

    monkeypatch.setattr(mod, "chunk_kda_fwd", native_backend)
    mod.initialize_kda_checkpoints(layer, config)
    assert layer._ascend_internal_checkpoint_enabled
    assert layer._checkpoint_alignment == 64
    assert layer._checkpoint_exporter.state_len == 3
    workspace_manager.get_simultaneous.assert_called_once_with(((3, 2, 128, 128), torch.float32))


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("has_checkpoint", [False, True])
def test_recover_prefill_rows_exclude_decode_speculation_and_padding(speculative, has_checkpoint):
    starts = torch.tensor([0, 2, 67, 67, 71, 200]) if speculative else torch.tensor([0, 1, 66, 66, 195])
    m = NS(
        query_start_loc_cpu=starts,
        query_start_loc=starts,
        num_reqs=starts.numel() - 1,
        is_prefilling=torch.tensor([False, True, False, False, True])
        if speculative
        else torch.tensor([False, True, False, True]),
    )
    cp = (
        NS(checkpoint_offsets=torch.tensor([64, 128], dtype=torch.int32), state_indices=torch.tensor([7, 0]))
        if has_checkpoint
        else None
    )
    builder = NS(
        num_spec=3,
        checkpoint_builder=NS(build=Mock(return_value=cp)),
        _non_spec_cpu_starts=lambda *_: torch.tensor([0, 65, 194]) if speculative else starts,
    )
    metadata = NS(
        num_decodes=0 if speculative else 1,
        num_decode_tokens=0 if speculative else 1,
        num_spec_decodes=2 if speculative else 0,
        non_spec_state_indices_tensor=torch.tensor([12, 13]) if speculative else torch.tensor([11, 12, 0, 13]),
        has_initial_state=torch.tensor([True, False]) if speculative else torch.tensor([False, True, True, False]),
    )
    result = mod.prepare_kda_checkpoint_prefill(
        builder, m, metadata, torch.tensor([0, -1, -1, 2, -1]) if speculative else None
    )
    assert result.tolist() == [0, 65, 194]
    assert builder.checkpoint_builder.build.call_args.args[1] == ([1, 4] if speculative else [1, 3])
    assert metadata.prefill_state_indices.tolist() == [12, 13]
    assert metadata.prefill_has_initial_state.tolist() == [True, False]
    assert metadata.checkpoint is cp
    if cp is not None:
        assert cp.checkpoint_offsets.tolist() == [64, 0]
