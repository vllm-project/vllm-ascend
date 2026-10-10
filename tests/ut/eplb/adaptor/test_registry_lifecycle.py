import gc
from types import SimpleNamespace
from unittest.mock import patch
from weakref import ref

import pytest
import torch
from torch import nn

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.quantization.quant_type import QuantType


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(VllmEplbAdaptor, "_registered_moe_layers", [])


class SmallExperts(nn.Module):
    def __init__(self, marker):
        super().__init__()
        self.marker = marker
        self.local_num_experts = 1
        self.ep_rank = 0
        self.quant_type = QuantType.NONE
        self.w13_weight = nn.Parameter(torch.full((1, 2, 2), float(marker)))
        self.w2_weight = nn.Parameter(torch.full((1, 2, 2), float(marker)))

    def get_log2phy_map(self):
        return torch.zeros(1, dtype=torch.int64)


def make_adaptor(*layers):
    model = nn.Module()
    model.config = SimpleNamespace(first_k_dense_replace=0)
    model.quant_config = None
    for index, layer in enumerate(layers):
        model.add_module(f"layer_{index}", layer)
        VllmEplbAdaptor.register_layer(layer)
    with (
        patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_rank", return_value=0),
        patch("vllm_ascend.eplb.adaptor.vllm_adaptor.dist.get_world_size", return_value=1),
        patch(
            "vllm_ascend.eplb.adaptor.vllm_adaptor.get_ascend_config", return_value=SimpleNamespace(enable_fused_mc2=0)
        ),
    ):
        return VllmEplbAdaptor(model)


def test_discarded_layer_and_parameter_are_collectible():
    layer = SmallExperts(1)
    layer_ref = ref(layer)
    weight_ref = ref(layer.w13_weight)
    VllmEplbAdaptor.register_layer(layer)
    del layer
    gc.collect()
    assert layer_ref() is None
    assert weight_ref() is None


def test_adaptor_keeps_registration_order_and_weight_indices():
    layers = [SmallExperts(marker) for marker in (4, 1, 3)]
    adaptor = make_adaptor(*layers)
    assert adaptor.moe_layers == layers
    assert adaptor.num_moe_layers == 3
    for index, layer in enumerate(layers):
        assert adaptor.expert_param_per_layer[index][0][0].data_ptr() == layer.w13_weight[0].data_ptr()
        assert adaptor.expert_param_per_layer[index][0][1].data_ptr() == layer.w2_weight[0].data_ptr()


def test_adaptor_snapshot_is_unchanged_by_later_registrations():
    first = SmallExperts(1)
    adaptor = make_adaptor(first)
    later = SmallExperts(2)
    VllmEplbAdaptor.register_layer(later)
    assert adaptor.moe_layers == [first]
    assert adaptor.num_moe_layers == len(adaptor.moe_layers)


def test_duplicate_layer_registrations_keep_layer_positions():
    layer = SmallExperts(1)
    adaptor = make_adaptor(layer, layer)
    assert adaptor.moe_layers == [layer, layer]
    assert adaptor.num_moe_layers == 2


def test_dead_layers_do_not_shift_live_weight_positions():
    discarded = SmallExperts(9)
    VllmEplbAdaptor.register_layer(discarded)
    discarded_ref = ref(discarded)
    del discarded
    gc.collect()
    layers = [SmallExperts(marker) for marker in (2, 7)]
    adaptor = make_adaptor(*layers)
    assert discarded_ref() is None
    assert adaptor.moe_layers == layers
    for index, layer in enumerate(layers):
        assert adaptor.expert_param_per_layer[index][0][0].data_ptr() == layer.w13_weight[0].data_ptr()


def test_registering_new_layers_prunes_dead_reference_records():
    discarded = SmallExperts(1)
    VllmEplbAdaptor.register_layer(discarded)
    del discarded
    gc.collect()
    live = SmallExperts(2)
    VllmEplbAdaptor.register_layer(live)
    assert VllmEplbAdaptor.get_registered_layers() == [live]
    assert len(VllmEplbAdaptor._registered_moe_layers) == 1
