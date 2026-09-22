# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_ascend.attention.context_parallel import dsa_cp
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl


class TestAscendDSACPLayerMetadata:
    def test_routes_by_cache_prefix(self):
        impl = AscendDSACPImpl.__new__(AscendDSACPImpl)
        impl.compress_ratio = 4
        impl.swa_cache_layer = SimpleNamespace(prefix="swa_cache")
        impl.compressor = SimpleNamespace(state_cache=SimpleNamespace(prefix="compressor.state_cache"))
        impl.indexer = SimpleNamespace(
            k_cache=SimpleNamespace(prefix="indexer.k_cache"),
            compressor=SimpleNamespace(state_cache=SimpleNamespace(prefix="indexer.compressor.state_cache")),
        )
        attention_metadata = object()
        compressor_state_metadata = object()
        indexer_cache_metadata = object()
        indexer_state_metadata = object()
        swa_metadata = object()

        metadata: Any = {
            "layer": attention_metadata,
            "compressor.state_cache": compressor_state_metadata,
            "indexer.k_cache": indexer_cache_metadata,
            "indexer.compressor.state_cache": indexer_state_metadata,
            "swa_cache": swa_metadata,
        }
        layer_metadata = impl._get_layer_metadata("layer", metadata)

        assert layer_metadata.swa is swa_metadata
        assert layer_metadata.compressor_cache is attention_metadata
        assert layer_metadata.compressor_state is compressor_state_metadata
        assert layer_metadata.indexer_cache is indexer_cache_metadata
        assert layer_metadata.indexer_state is indexer_state_metadata


@pytest.mark.parametrize("fail_at", [None, "project", "copy"])
def test_o_projection_keeps_weight_lifetime_in_caller(monkeypatch, fail_at):
    events = []
    metadata = SimpleNamespace(attn_state=dsa_cp.AscendAttentionState.PrefillNoCache)
    tensor = torch.ones(2, 4)

    def project(value, *, full_gather_wo_a_enabled):
        assert full_gather_wo_a_enabled
        events.append("project")
        if fail_at == "project":
            raise RuntimeError("projection failed")
        return value

    class Output:
        def __setitem__(self, index, value):
            events.append("copy")
            if fail_at == "copy":
                raise RuntimeError("copy failed")

    impl = SimpleNamespace(
        tp_size=2,
        enable_dsa_cp_full_o_proj=True,
        _get_layer_metadata=lambda *args: SimpleNamespace(compressor_cache=metadata),
        _forward=lambda *args: tensor,
        _restore_tp_head_layout=lambda *args, **kwargs: tensor,
        _switch_o_proj_to_full_weight=lambda: events.append("full"),
        _switch_o_proj_to_local_weight=lambda: events.append("local"),
        _forward_o_proj=project,
    )
    monkeypatch.setattr(dsa_cp, "wait_for_kv_layer_from_connector", lambda *args: None)
    monkeypatch.setattr(dsa_cp, "maybe_save_kv_layer_to_connector", lambda *args: events.append("save"))
    output = Output()
    if fail_at:
        with pytest.raises(RuntimeError, match="failed"):
            AscendDSACPImpl.forward(impl, "layer", tensor, (), {}, output)
        assert events[-1] == "local"
        assert "save" not in events
    else:
        assert AscendDSACPImpl.forward(impl, "layer", tensor, (), {}, output) is output
        assert events == ["full", "project", "copy", "local", "save"]


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("fp8_attention", [False, True])
@pytest.mark.parametrize("with_output", [False, True])
def test_o_projection_common_computation_uses_active_weight_layout(monkeypatch, full, fp8_attention, with_output):
    def batch_matmul(value, weight, **kwargs):
        return torch.bmm(value.transpose(0, 1), weight).transpose(0, 1)

    monkeypatch.setattr(dsa_cp.torch_npu, "npu_transpose_batchmatmul", batch_matmul, raising=False)
    monkeypatch.setattr(dsa_cp, "_has_weight_scale", lambda layer: True)
    method = object.__new__(dsa_cp.AscendUnquantizedLinearMethod)
    groups = 2 if full else 1
    value = torch.arange(4 * groups * 3, dtype=torch.float32).reshape(4, groups * 3)
    weight = torch.arange(groups * 3 * 2, dtype=torch.float32).reshape(groups, 3, 2)
    wo_b = torch.arange(groups * 2 * 5, dtype=torch.float32).reshape(groups * 2, 5)
    impl = SimpleNamespace(
        n_group=2,
        n_local_groups=1,
        support_fp8_attention=fp8_attention,
        wo_a=SimpleNamespace(quant_method=method),
        _get_batched_wo_a_weight=lambda count: weight,
        _apply_wo_b=lambda value, gathered: value @ wo_b,
    )
    expected = torch.einsum("tgh,ghr->tgr", value.reshape(4, groups, 3), weight).reshape(4, -1) @ wo_b
    output = torch.empty_like(expected) if with_output else None
    actual = AscendDSACPImpl._forward_o_proj(impl, value, output, full_gather_wo_a_enabled=full)
    if with_output:
        assert actual is output
    torch.testing.assert_close(actual, expected)
