# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import sys
import typing
from collections.abc import Iterable
from types import SimpleNamespace

import pytest
import regex as re
import torch
from torch.nn import Module

from .rfork_test_support import MODELS_ROOT, _exec_source_definitions, _stub

_DEEPSEEK_DSPARK_CLASSES = {
    "deepseek_v4": "DSparkDeepseekV4ForCausalLM",
    "deepseek_v41": "DSparkDeepseekV41ForCausalLM",
}

_DEEPSEEK_DSPARK_CHECKPOINT_WITHOUT_OWN_HEADS = {
    "embed.weight",
    "head.weight",
    "hc_head_fn",
    "layers.0.attn.wkv.weight",
    "mtp.0.main_proj.weight",
    "mtp.0.attn.wkv.weight",
    "mtp.1.attn.wkv.weight",
    "mtp.0.attn.attn_sink",
    "mtp.0.hc_head_fn",
    "mtp.1.ffn.experts.0.w1.weight",
    "mtp.1.ffn.gate.e_score_correction_bias_vl",
    "mtp.2.markov_head.embed.weight",
    "mtp.2.markov_head.head.weight",
    "mtp.2.confidence_head.weight",
    "mtp.2.norm.weight",
}


class _AcceptingParams(dict):
    def __missing__(self, name):
        return SimpleNamespace(weight_loader=lambda *args, **kwargs: True, copy_=lambda *args: None)


def _process_eagle_weight(model, name):
    if "embed_tokens" in name:
        model.has_own_embed_tokens = True
    if "lm_head" in name:
        model.has_own_lm_head = True


def _metadata_configs():
    return SimpleNamespace(model="draft", revision=None), SimpleNamespace(
        load_format="rfork", model_loader_extra_config={}
    )


@pytest.fixture
def eagle_utils(monkeypatch):
    _stub(monkeypatch, "vllm.model_executor.models", __path__=[])
    _stub(monkeypatch, "vllm.model_executor.models.utils", process_eagle_weight=_process_eagle_weight)


@pytest.fixture(params=sorted(_DEEPSEEK_DSPARK_CLASSES))
def deepseek_dspark(request, monkeypatch, eagle_utils):
    family = request.param
    class_name = _DEEPSEEK_DSPARK_CLASSES[family]
    namespace = {
        "Iterable": Iterable,
        "typing": typing,
        "torch": torch,
        "re": re,
        "process_eagle_weight": _process_eagle_weight,
        "get_tensor_model_parallel_rank": lambda: 0,
        "get_tensor_model_parallel_world_size": lambda: 1,
        "default_weight_loader": lambda param, weight: param.data.copy_(weight),
        "logger": SimpleNamespace(info_once=lambda *args: None),
        "dict": _AcceptingParams,
    }
    _exec_source_definitions(
        MODELS_ROOT / "deepseek_v4" / "dspark.py", {"_CONTEXT_WKV_RE", "_duplicate_context_wkv_weights"}, namespace
    )
    methods = {"load_weights", "_remap_dspark_name"}
    if family == "deepseek_v41":
        methods.add("_remap_checkpoint_name")
    _exec_source_definitions(MODELS_ROOT / family / "dspark.py", methods, namespace, class_name)

    class DraftModel(Module):
        has_own_embed_tokens = False
        has_own_lm_head = False
        num_redundant_experts = 0

        def __init__(self, rotation_path, layers):
            super().__init__()
            self.rotation_path = rotation_path
            self.config = SimpleNamespace(num_attention_heads=1, num_hidden_layers=3)
            self.model = Module()
            self.model.num_dspark_layers = layers
            self.model.get_expert_mapping = lambda **kwargs: []
            self.model.embed_tokens = torch.nn.Embedding(2, 2)
            self.lm_head = torch.nn.Linear(2, 2, bias=False)

    for method in methods:
        setattr(DraftModel, method, namespace[method])
    _stub(monkeypatch, f"vllm_ascend.models.{family}.dspark", **{class_name: DraftModel})
    return DraftModel


@pytest.mark.parametrize(
    ("names", "rotation", "layers", "expected"),
    [
        ({"mtp.0.embed.weight", "mtp.2.head.weight"}, "rotation", 3, (True, True)),
        ({"mtp.0.embed.weight"}, "rotation", 3, (True, False)),
        ({"mtp.2.head.weight"}, "rotation", 3, (False, True)),
        ({"unrelated.weight"}, "rotation", 3, (False, False)),
        ({"embed.weight", "head.weight"}, None, 3, (True, True)),
        ({"embed.weight", "head.weight"}, "rotation", 3, (False, False)),
        ({"mtp.0.embed.weight", "mtp.3.head.weight"}, "rotation", 4, (True, True)),
        ({"mtp.1.embed.weight", "mtp.1.head.weight"}, "rotation", 3, (False, False)),
        (_DEEPSEEK_DSPARK_CHECKPOINT_WITHOUT_OWN_HEADS, "rotation", 3, (False, False)),
        (_DEEPSEEK_DSPARK_CHECKPOINT_WITHOUT_OWN_HEADS, None, 3, (True, True)),
        (_DEEPSEEK_DSPARK_CHECKPOINT_WITHOUT_OWN_HEADS - {"embed.weight"}, None, 3, (False, True)),
        (
            _DEEPSEEK_DSPARK_CHECKPOINT_WITHOUT_OWN_HEADS | {"mtp.0.embed.weight", "mtp.2.head.weight"},
            "rotation",
            3,
            (True, True),
        ),
    ],
)
def test_deepseek_dspark_preparation_matches_local_loading(
    rfork_helpers, deepseek_dspark, monkeypatch, names, rotation, layers, expected
):
    local = deepseek_dspark(rotation, layers)
    receiver = deepseek_dspark(rotation, layers)
    before = {name: param.detach().clone() for name, param in receiver.named_parameters()}
    local.load_weights((name, torch.ones(2, 2)) for name in names)
    monkeypatch.setattr(rfork_helpers.loader, "_get_checkpoint_weight_names", lambda *args: names)

    rfork_helpers.loader._prepare_rfork_model_state(receiver, SimpleNamespace(), SimpleNamespace())

    assert (local.has_own_embed_tokens, local.has_own_lm_head) == expected
    assert (receiver.has_own_embed_tokens, receiver.has_own_lm_head) == expected
    for name, param in receiver.named_parameters():
        torch.testing.assert_close(param, before[name])


def test_deepseek_dspark_preparation_reads_selected_checkpoint_index(
    rfork_helpers, deepseek_dspark, checkpoint_loader, tmp_path
):
    weight_map = dict.fromkeys(("mtp.0.embed.weight", "mtp.2.head.weight", "layers.0.attn.wkv.weight"), "a.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    checkpoint_loader.folders["draft"] = str(tmp_path)
    checkpoint_loader.selections["draft"] = ([str(tmp_path / "a.safetensors")], True)
    receiver = deepseek_dspark("rotation", 3)

    rfork_helpers.loader._prepare_rfork_model_state(receiver, *_metadata_configs())

    assert (receiver.has_own_embed_tokens, receiver.has_own_lm_head) == (True, True)


@pytest.mark.parametrize("model_index_owns_heads", [True, False])
def test_deepseek_dspark_preparation_uses_loader_selected_index(
    rfork_helpers, deepseek_dspark, checkpoint_loader, tmp_path, model_index_owns_heads
):
    own_heads = {"mtp.0.embed.weight": "a.safetensors", "mtp.2.head.weight": "a.safetensors"}
    shared_heads = {"layers.0.attn.wkv.weight": "a.safetensors"}
    indexes = {
        "model.safetensors.index.json": own_heads if model_index_owns_heads else shared_heads,
        "consolidated.safetensors.index.json": shared_heads if model_index_owns_heads else own_heads,
    }
    for index_name, weight_map in indexes.items():
        (tmp_path / index_name).write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    checkpoint_loader.folders["draft"] = str(tmp_path)
    checkpoint_loader.selections["draft"] = ([str(tmp_path / "a.safetensors")], True)
    receiver = deepseek_dspark("rotation", 3)

    rfork_helpers.loader._prepare_rfork_model_state(receiver, *_metadata_configs())

    expected = (model_index_owns_heads, model_index_owns_heads)
    assert (receiver.has_own_embed_tokens, receiver.has_own_lm_head) == expected


def test_deepseek_dspark_preparation_rejects_checkpoints_without_safetensors_metadata(
    rfork_helpers, deepseek_dspark, checkpoint_loader
):
    checkpoint_loader.selections["draft"] = (["model.bin"], False)
    receiver = deepseek_dspark(None, 3)

    with pytest.raises(ValueError, match="requires safetensors checkpoint metadata"):
        rfork_helpers.loader._prepare_rfork_model_state(receiver, *_metadata_configs())
    assert (receiver.has_own_embed_tokens, receiver.has_own_lm_head) == (False, False)


def test_deepseek_dspark_detection_uses_already_imported_model_classes(rfork_helpers, monkeypatch):
    class DSparkV4(Module):
        pass

    class DSparkV4Subclass(DSparkV4):
        pass

    is_dspark = rfork_helpers.loader._is_deepseek_dspark
    v4_module, v41_module = (f"vllm_ascend.models.{family}.dspark" for family in _DEEPSEEK_DSPARK_CLASSES)
    monkeypatch.delitem(sys.modules, v4_module, raising=False)
    monkeypatch.delitem(sys.modules, v41_module, raising=False)

    assert not is_dspark(Module())
    assert v4_module not in sys.modules and v41_module not in sys.modules

    _stub(monkeypatch, v4_module, DSparkDeepseekV4ForCausalLM=DSparkV4)
    _stub(monkeypatch, v41_module)
    assert is_dspark(DSparkV4())
    assert is_dspark(DSparkV4Subclass())
    assert not is_dspark(Module())


def test_model_state_preparation_ignores_unregistered_dspark_lookalike(rfork_helpers, monkeypatch):
    class Lookalike(Module):
        has_own_embed_tokens = False
        has_own_lm_head = False
        rotation_path = None

        def _remap_dspark_name(self, name):
            return "lm_head.weight"

    monkeypatch.setattr(
        rfork_helpers.loader,
        "_get_checkpoint_weight_names",
        lambda *args: pytest.fail("non-DSpark models must not read checkpoint metadata"),
    )
    model = Lookalike()

    rfork_helpers.loader._prepare_rfork_model_state(model, SimpleNamespace(), SimpleNamespace())

    assert (model.has_own_embed_tokens, model.has_own_lm_head) == (False, False)
