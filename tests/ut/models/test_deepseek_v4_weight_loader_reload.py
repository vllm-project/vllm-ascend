# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Attention sinks must be applied through the parameter's weight loader.

A live weight update runs inside vLLM's layerwise reload: the layer is parked on
the meta device and only writes made through ``param.weight_loader`` are
buffered and replayed onto the materialized layer by
``finalize_layerwise_reload``. Writing the sink with a direct
``param.data.copy_`` therefore lands on the meta tensor and is dropped, which
left the attention sinks at their dummy initialisation after every update.

The sink is a plain parameter, so it resolves to vLLM's
``default_weight_loader``, which only asserts the shapes and copies. The tensor
parallel sharding stays owned by ``load_weights``: the checkpoint ships one sink
per head and each rank must receive exactly ``num_attention_heads // tp_size``
consecutive heads starting at ``tp_rank * heads_per_rank``. These tests pin that
offset at TP > 1 so a future rank-aware loader (``row_parallel_weight_loader``
narrows 1-D parameters as well) cannot silently double-shard the sink.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_ascend.models.deepseek_v4 import model as deepseek_v4_module
from vllm_ascend.models.deepseek_v4 import mtp as mtp_module

TOTAL_HEADS = 16
# The served parameter keeps the checkpoint's ``attn_sink`` leaf name; only the
# container module differs (``layers.N.attn.attn_sink`` in the checkpoint vs
# ``model.layers.N.self_attn.attn_sink`` in the model).
SINK_PARAM_NAME = "model.layers.0.self_attn.attn_sink"
SINK_WEIGHT_NAME = "model.layers.0.self_attn.attn_sink"
MTP_SINK_PARAM_NAME = "model.layers.0.self_attn.attn_sink"
MTP_SINK_WEIGHT_NAME = "mtp.0.attn.attn_sink"

# (tp_size, tp_rank) pairs covering TP=1 and every rank of TP=2/4/8.
TP_RANKS = [(1, 0)] + [(size, rank) for size in (2, 4, 8) for rank in range(size)]


class _SinkLayer(nn.Module):
    """Minimal stand-in for the decoder layer holding only the attention sink."""

    def __init__(self, *, with_weight_loader: bool, heads_per_rank: int) -> None:
        super().__init__()
        self.self_attn = _SinkAttention(with_weight_loader=with_weight_loader, heads_per_rank=heads_per_rank)

    @property
    def attn_sink(self) -> nn.Parameter:
        return self.self_attn.attn_sink

    @property
    def loader_calls(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self.self_attn.loader_calls


class _SinkAttention(nn.Module):
    """Minimal stand-in for ``DeepseekV4Attention`` holding only the sink."""

    def __init__(self, *, with_weight_loader: bool, heads_per_rank: int) -> None:
        super().__init__()
        self.attn_sink = nn.Parameter(torch.zeros(heads_per_rank, dtype=torch.float32))
        self.loader_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        if with_weight_loader:

            def _record(weight: torch.Tensor, loaded_weight: torch.Tensor) -> None:
                self.loader_calls.append((weight, loaded_weight.clone()))

            self.attn_sink.weight_loader = _record  # type: ignore[assignment]


class _SinkOnlyModel(nn.Module):
    """``ForCausalLM`` stub whose parameter set is exactly one sink.

    ``load_weights`` walks ``self.model`` and looks the sink up as
    ``model.layers.{i}.self_attn.attn_sink``, so the stub mirrors that
    hierarchy: ``model`` -> ``layers`` -> ``{i}`` -> ``self_attn`` -> sink.
    """

    def __init__(self, layer: _SinkLayer) -> None:
        super().__init__()
        self.model = nn.ModuleDict({"layers": nn.ModuleDict({"0": layer})})
        # The sink branch only reads ``num_attention_heads`` from the config.
        self.config = SimpleNamespace(num_attention_heads=TOTAL_HEADS, n_routed_experts=8, n_shared_experts=1)
        self.num_redundant_experts = 0


class _SinkOnlyMtp(nn.Module):
    """``DeepSeekV4MTP`` stub.

    The MTP checkpoint name ``mtp.0.attn.attn_sink`` is rewritten by
    ``load_weights`` onto ``model.layers.0.self_attn.attn_sink``, which is what
    the oracle's ``deepseek_v4_checkpoint_name`` produces for the MTP layers too.
    """

    def __init__(self, layer: _SinkLayer) -> None:
        super().__init__()
        self.model = nn.ModuleDict({"layers": nn.ModuleDict({"0": layer})})
        self.config = SimpleNamespace(num_attention_heads=TOTAL_HEADS, n_routed_experts=8, n_shared_experts=1)
        self.quant_config = None
        self.num_redundant_experts = 0

    def no_mtp_block_in_name(self, layer_name: str) -> bool:
        # Mirrors the real predicate for the ``mtp.0.attn.attn_sink`` name shape.
        return True


@pytest.fixture
def parallel_rank(monkeypatch):
    """Force the TP world size/rank the loaders read at call time.

    ``load_weights`` imports both helpers into its own module namespace, so the
    patch has to land on the imported name rather than on ``vllm.distributed``.
    """

    def _set(tp_size: int, tp_rank: int) -> None:
        monkeypatch.setattr(deepseek_v4_module, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(deepseek_v4_module, "get_tensor_model_parallel_rank", lambda: tp_rank)
        monkeypatch.setattr(mtp_module, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(mtp_module, "get_tensor_model_parallel_rank", lambda: tp_rank)

    return _set


@pytest.fixture
def load_sink(monkeypatch, parallel_rank):
    """Drive the real ``load_weights`` over a model holding only a sink."""
    monkeypatch.setattr(deepseek_v4_module, "enable_dsa_cp", lambda: False)
    monkeypatch.setattr(deepseek_v4_module, "is_pp_missing_parameter", lambda *_: False)
    monkeypatch.setattr(deepseek_v4_module, "get_spec_layer_idx_from_weight_name", lambda *_: None)
    monkeypatch.setattr(deepseek_v4_module, "fused_moe_make_expert_params_mapping", lambda *_, **__: [])
    monkeypatch.setattr(deepseek_v4_module, "get_ascend_config", lambda: MagicMock(mix_placement=False))

    def _run(layer: _SinkLayer, sink: torch.Tensor, *, tp_size: int, tp_rank: int) -> set[str]:
        parallel_rank(tp_size, tp_rank)
        loader = deepseek_v4_module.AscendDeepseekV4ForCausalLM.load_weights.__get__(_SinkOnlyModel(layer))
        return loader([(SINK_WEIGHT_NAME, sink)])

    return _run


@pytest.fixture
def load_mtp_sink(monkeypatch, parallel_rank):
    """Same, for ``DeepSeekV4MTP.load_weights`` and its own name mapping."""
    monkeypatch.setattr(mtp_module, "enable_dsa_cp", lambda: False)
    monkeypatch.setattr(mtp_module, "get_spec_layer_idx_from_weight_name", lambda *_: 0)
    monkeypatch.setattr(mtp_module, "fused_moe_make_expert_params_mapping", lambda *_, **__: [])
    monkeypatch.setattr(mtp_module, "get_ascend_config", lambda: MagicMock(mix_placement=False))

    def _run(layer: _SinkLayer, sink: torch.Tensor, *, tp_size: int, tp_rank: int) -> set[str]:
        parallel_rank(tp_size, tp_rank)
        loader = mtp_module.DeepSeekV4MTP.load_weights.__get__(_SinkOnlyMtp(layer))
        return loader([(MTP_SINK_WEIGHT_NAME, sink)])

    return _run


def _narrow_reference(sink: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    heads_per_rank = TOTAL_HEADS // tp_size
    return sink[tp_rank * heads_per_rank : (tp_rank + 1) * heads_per_rank]


@pytest.mark.parametrize(("tp_size", "tp_rank"), TP_RANKS, ids=lambda value: str(value))
def test_mtp_attention_sink_goes_through_its_weight_loader(load_mtp_sink, tp_size, tp_rank):
    """The MTP copy of the sink branch must shard and route identically."""
    heads_per_rank = TOTAL_HEADS // tp_size
    layer = _SinkLayer(with_weight_loader=True, heads_per_rank=heads_per_rank)
    sink = torch.arange(TOTAL_HEADS, dtype=torch.float32)

    loaded = load_mtp_sink(layer, sink, tp_size=tp_size, tp_rank=tp_rank)

    assert loaded == {MTP_SINK_PARAM_NAME}
    assert len(layer.loader_calls) == 1, "the sink must be applied exactly once"
    weight, written = layer.loader_calls[0]
    assert weight is layer.attn_sink
    assert written.shape == layer.attn_sink.shape
    torch.testing.assert_close(written, _narrow_reference(sink, tp_size, tp_rank))


@pytest.mark.parametrize(("tp_size", "tp_rank"), TP_RANKS, ids=lambda value: str(value))
def test_attention_sink_goes_through_its_weight_loader(load_sink, tp_size, tp_rank):
    """A loader-backed sink must reach the loader with this rank's head slice."""
    heads_per_rank = TOTAL_HEADS // tp_size
    layer = _SinkLayer(with_weight_loader=True, heads_per_rank=heads_per_rank)
    sink = torch.arange(TOTAL_HEADS, dtype=torch.float32)

    loaded = load_sink(layer, sink, tp_size=tp_size, tp_rank=tp_rank)

    assert loaded == {SINK_PARAM_NAME}
    assert len(layer.loader_calls) == 1, "the sink must be applied exactly once"
    weight, written = layer.loader_calls[0]
    assert weight is layer.attn_sink
    # The slice matches the parameter, so ``default_weight_loader``'s shape
    # assertion holds and no rank re-partitions the weight a second time.
    assert written.shape == layer.attn_sink.shape
    torch.testing.assert_close(written, _narrow_reference(sink, tp_size, tp_rank))


@pytest.mark.parametrize(("tp_size", "tp_rank"), TP_RANKS, ids=lambda value: str(value))
def test_attention_sink_falls_back_to_the_default_loader(load_sink, tp_size, tp_rank):
    """A plain parameter keeps the previous direct-copy semantics."""
    heads_per_rank = TOTAL_HEADS // tp_size
    layer = _SinkLayer(with_weight_loader=False, heads_per_rank=heads_per_rank)
    sink = torch.arange(TOTAL_HEADS, dtype=torch.float32)

    loaded = load_sink(layer, sink, tp_size=tp_size, tp_rank=tp_rank)

    assert loaded == {SINK_PARAM_NAME}
    assert not layer.loader_calls
    torch.testing.assert_close(layer.attn_sink.detach(), _narrow_reference(sink, tp_size, tp_rank))
