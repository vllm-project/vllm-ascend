# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise real Sink loaders, checkpoint mappings and layerwise reload on CPU.

TP sharding belongs to load_weights, before the parameter's current loader is
called. DSA CP keeps the full Sink. Direct copy_ bypasses the reload wrapper and
loses updates made while the parameter is on meta.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)

from vllm_ascend.models.deepseek_v4 import dspark as dspark_module
from vllm_ascend.models.deepseek_v4 import model as target_module
from vllm_ascend.models.deepseek_v4 import mtp as mtp_module

TOTAL_HEADS = 16
TARGET_LAYER_IDX = 3
DSPARK_STAGE = 1
# TP=1 and every rank of TP=2/4/8, including nonzero offsets.
TP_RANKS = [(1, 0)] + [(size, rank) for size in (2, 4, 8) for rank in range(size)]


class _SinkAttention(nn.Module):
    def __init__(self, num_heads: int, with_weight_loader: bool) -> None:
        super().__init__()
        self.attn_sink = nn.Parameter(torch.full((num_heads,), -1.0), requires_grad=False)
        self.loader_calls: list[tuple[nn.Parameter, torch.Tensor]] = []
        if with_weight_loader:
            self.attn_sink.weight_loader = self.record_load

    def record_load(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        # Deliberately do not copy: load_weights must leave the parameter alone.
        self.loader_calls.append((param, loaded_weight.clone()))


class _SinkOnlyModel(nn.Module):
    """Minimal hierarchy needed by the production load_weights methods."""

    def __init__(self, attention: _SinkAttention, path: str) -> None:
        super().__init__()
        layer = nn.Module()
        if path == "mtp":
            layer.mtp_block = nn.Module()
            layer.mtp_block.self_attn = attention
        else:
            layer.self_attn = attention
        layer_idx = TARGET_LAYER_IDX + DSPARK_STAGE if path == "dspark" else 0
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({str(layer_idx): layer})
        self.model.num_dspark_layers = 2
        self.model.get_expert_mapping = lambda: []
        self.config = SimpleNamespace(
            num_attention_heads=TOTAL_HEADS,
            num_hidden_layers=TARGET_LAYER_IDX,
            n_routed_experts=8,
            n_shared_experts=1,
        )
        self.num_redundant_experts = 0
        self.quant_config = None
        self.rotation_path = None

    # Keep the actual predicates/mappings, including the MTP block container.
    no_mtp_block_in_name = mtp_module.DeepSeekV4MTP.no_mtp_block_in_name
    _remap_dspark_name = dspark_module.DSparkDeepseekV4ForCausalLM._remap_dspark_name


@pytest.fixture(params=["target", "mtp", "dspark"])
def sink_path(request, monkeypatch):
    """Patch distributed/config dependencies, keeping real loaders and mappings."""
    path = request.param
    modules = {"target": target_module, "mtp": mtp_module, "dspark": dspark_module}
    loaders = {
        "target": target_module.AscendDeepseekV4ForCausalLM.load_weights,
        "mtp": mtp_module.DeepSeekV4MTP.load_weights,
        "dspark": dspark_module.DSparkDeepseekV4ForCausalLM.load_weights,
    }
    checkpoint_names = {
        "target": "model.layers.0.attn.attn_sink",
        "mtp": "mtp.0.attn.attn_sink",
        "dspark": f"mtp.{DSPARK_STAGE}.attn.attn_sink",
    }
    parameter_names = {
        "target": "model.layers.0.self_attn.attn_sink",
        "mtp": "model.layers.0.mtp_block.self_attn.attn_sink",
        "dspark": f"model.layers.{TARGET_LAYER_IDX + DSPARK_STAGE}.self_attn.attn_sink",
    }
    for module in (target_module, mtp_module):
        monkeypatch.setattr(module, "fused_moe_make_expert_params_mapping", lambda *_, **__: [])
        monkeypatch.setattr(module, "get_ascend_config", lambda: SimpleNamespace(mix_placement=False))
    monkeypatch.setattr(target_module, "is_pp_missing_parameter", lambda *_: False)
    monkeypatch.setattr(dspark_module, "process_eagle_weight", lambda *_: None)

    def build(*, tp_size, tp_rank, dsa_cp=False, with_weight_loader=False):
        module = modules[path]
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
        monkeypatch.setattr(module, "enable_dsa_cp", lambda: dsa_cp)
        attention = _SinkAttention(TOTAL_HEADS if dsa_cp else TOTAL_HEADS // tp_size, with_weight_loader)
        model = _SinkOnlyModel(attention, path)
        load_weights = loaders[path].__get__(model)
        return SimpleNamespace(
            model=model,
            attention=attention,
            parameter_name=parameter_names[path],
            load=lambda weight: load_weights([(checkpoint_names[path], weight)]),
        )

    return build


def _expected_sink(weight, tp_size, tp_rank, dsa_cp):
    if dsa_cp:
        return weight
    heads_per_rank = TOTAL_HEADS // tp_size
    return weight[tp_rank * heads_per_rank : (tp_rank + 1) * heads_per_rank]


def _check_load(case, *, tp_size, tp_rank, dsa_cp, with_weight_loader):
    weight = torch.arange(TOTAL_HEADS, dtype=torch.float32)
    param = case.attention.attn_sink
    original_value = param.detach().clone()
    expected = _expected_sink(weight, tp_size, tp_rank, dsa_cp)

    assert case.load(weight) == {case.parameter_name}
    assert dict(case.model.named_parameters())[case.parameter_name] is param
    if with_weight_loader:
        assert len(case.attention.loader_calls) == 1
        destination, received = case.attention.loader_calls[0]
        assert destination is param
        assert received.shape == param.shape
        torch.testing.assert_close(received, expected)
        torch.testing.assert_close(param.detach(), original_value)
    else:
        assert not case.attention.loader_calls
        torch.testing.assert_close(param.detach(), expected)


@pytest.mark.parametrize(("tp_size", "tp_rank"), TP_RANKS)
@pytest.mark.parametrize("with_weight_loader", [True, False], ids=["custom", "default"])
def test_attention_sink_tp_loading(sink_path, tp_size, tp_rank, with_weight_loader):
    case = sink_path(tp_size=tp_size, tp_rank=tp_rank, with_weight_loader=with_weight_loader)
    _check_load(case, tp_size=tp_size, tp_rank=tp_rank, dsa_cp=False, with_weight_loader=with_weight_loader)


@pytest.mark.parametrize("with_weight_loader", [True, False], ids=["custom", "default"])
def test_attention_sink_dsa_cp_loading(sink_path, with_weight_loader):
    # A nonzero rank distinguishes the complete Sink from the ordinary TP slice.
    case = sink_path(tp_size=2, tp_rank=1, dsa_cp=True, with_weight_loader=with_weight_loader)
    _check_load(case, tp_size=2, tp_rank=1, dsa_cp=True, with_weight_loader=with_weight_loader)


@pytest.mark.parametrize("dsa_cp", [False, True], ids=["tp", "dsa_cp"])
def test_attention_sink_layerwise_reload(sink_path, dsa_cp):
    """The real upstream wrapper must replay two updates into original storage."""
    case = sink_path(tp_size=2, tp_rank=1, dsa_cp=dsa_cp)
    initial_weight = torch.arange(TOTAL_HEADS, dtype=torch.float32)
    assert case.load(initial_weight) == {case.parameter_name}
    original_param = case.attention.attn_sink
    original_value = original_param.detach().clone()
    storage_ptr = original_param.untyped_storage().data_ptr()
    torch.testing.assert_close(original_value, _expected_sink(initial_weight, 2, 1, dsa_cp))
    record_metadata_for_reloading(case.model)

    for offset in (100, 200):
        initialize_layerwise_reload(case.model)
        assert case.attention.attn_sink.is_meta
        assert case.attention.attn_sink.weight_loader.__name__ == "online_process_loader"
        new_weight = initial_weight + offset
        assert case.load(new_weight) == {case.parameter_name}
        finalize_layerwise_reload(case.model, SimpleNamespace(dtype=torch.float32))
        sink = case.attention.attn_sink
        assert sink.device.type == "cpu"
        assert sink.untyped_storage().data_ptr() == storage_ptr
        torch.testing.assert_close(sink.detach(), _expected_sink(new_weight, 2, 1, dsa_cp))
        torch.testing.assert_close(original_param.detach(), sink.detach())
    assert not torch.equal(original_param.detach(), original_value)
