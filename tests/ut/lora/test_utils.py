# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm.lora.utils as lora_utils
from vllm.config import LoRAConfig
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

from vllm_ascend.lora.fused_moe import AscendFusedMoE3DWithLoRA, AscendFusedMoEWithLoRA
from vllm_ascend.lora.utils import (
    AscendColumnParallelLinearWithLoRA,
    AscendMergedColumnParallelLinearWithLoRA,
    AscendMergedQKVParallelLinearWithLoRA,
    AscendReplicatedLinearWithLoRA,
    AscendRowParallelLinearWithLoRA,
    refresh_all_lora_classes,
    warn_unmatched_lora_modules,
)
from vllm_ascend.ops.linear import AscendQKVParallelLinear


def test_refresh_all_lora_classes_prepends_ascend_wrappers() -> None:
    sentinel = object()
    original = lora_utils._all_lora_classes
    try:
        lora_utils._all_lora_classes = (sentinel,)
        refresh_all_lora_classes()
        classes = list(lora_utils._all_lora_classes)
        expected_prefix = [
            AscendReplicatedLinearWithLoRA,
            AscendRowParallelLinearWithLoRA,
            AscendColumnParallelLinearWithLoRA,
            AscendMergedColumnParallelLinearWithLoRA,
            AscendMergedQKVParallelLinearWithLoRA,
            AscendFusedMoEWithLoRA,
            AscendFusedMoE3DWithLoRA,
        ]
        assert classes[: len(expected_prefix)] == expected_prefix
        assert classes[-1] is sentinel
        # Upstream model_manager still matches the GPU class names.
        assert AscendFusedMoEWithLoRA.__name__ == "FusedMoEWithLoRA"
        assert AscendFusedMoE3DWithLoRA.__name__ == "FusedMoE3DWithLoRA"
    finally:
        lora_utils._all_lora_classes = original


def test_single_linear_wrapper_tracks_loaded_lora_slots() -> None:
    layer = object.__new__(AscendReplicatedLinearWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.base_layer = object.__new__(ReplicatedLinear)
    layer.n_slices = 1
    layer.input_size = 4
    layer.output_size = 6
    layer.tp_size = 1
    layer.device = torch.device("cpu")

    lora_config = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16)
    layer.create_lora_weights(max_loras=1, lora_config=lora_config)
    assert layer._lora_slots == set()

    layer.set_lora(
        0,
        torch.ones(8, 4, dtype=torch.bfloat16),
        torch.ones(6, 8, dtype=torch.bfloat16),
    )
    assert layer._lora_slots == {0}

    layer.reset_lora(0)
    assert layer._lora_slots == set()


def test_ascend_dense_wrappers_replace_and_track_slots_for_multi_lora() -> None:
    lora_config = LoRAConfig(max_loras=2, max_lora_rank=8, lora_dtype=torch.bfloat16)

    for wrapper_cls, base_cls, packed in (
        (AscendReplicatedLinearWithLoRA, maybe_get_oot_by_class(ReplicatedLinear), []),
        (AscendRowParallelLinearWithLoRA, maybe_get_oot_by_class(RowParallelLinear), []),
        (AscendColumnParallelLinearWithLoRA, maybe_get_oot_by_class(ColumnParallelLinear), []),
        (
            AscendMergedColumnParallelLinearWithLoRA,
            maybe_get_oot_by_class(MergedColumnParallelLinear),
            ["gate_proj", "up_proj"],
        ),
        (AscendMergedQKVParallelLinearWithLoRA, AscendQKVParallelLinear, ["q", "k", "v"]),
    ):
        source = object.__new__(base_cls)
        assert wrapper_cls.can_replace_layer(
            source_layer=source,
            lora_config=lora_config,
            packed_modules_list=packed,
            model_config=None,
        )

    single = object.__new__(AscendReplicatedLinearWithLoRA)
    torch.nn.Module.__init__(single)
    single.base_layer = object.__new__(ReplicatedLinear)
    single.base_layer.prefix = "model.layers.0.self_attn.wq_a"
    single.n_slices = 1
    single.input_size = 4
    single.output_size = 6
    single.tp_size = 1
    single.device = torch.device("cpu")
    single.create_lora_weights(max_loras=2, lora_config=lora_config)
    assert single.lora_b_packed is None
    assert single._lora_slots == set()

    single.set_lora(
        1,
        torch.ones(8, 4, dtype=torch.bfloat16),
        torch.ones(6, 8, dtype=torch.bfloat16),
    )
    assert single._lora_slots == {1}

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    y = torch.zeros(2, 6, dtype=torch.bfloat16)
    with patch.object(BaseLinearLayerWithLoRA, "_apply_lora_to_output", return_value="upstream") as upstream:
        assert single._apply_lora_to_output(x, y) == "upstream"
    upstream.assert_called_once_with(x, y)

    single.reset_lora(1)
    assert single._lora_slots == set()


def test_ascend_merged_wrapper_uses_upstream_expand_for_multi_lora() -> None:
    lora_config = LoRAConfig(max_loras=2, max_lora_rank=8, lora_dtype=torch.bfloat16)
    merged = object.__new__(AscendMergedColumnParallelLinearWithLoRA)
    torch.nn.Module.__init__(merged)
    merged.base_layer = object.__new__(MergedColumnParallelLinear)
    merged.base_layer.prefix = "model.layers.0.mlp.shared_experts.gate_up_proj"
    merged.input_size = 4
    merged.output_sizes = [6, 6]
    merged.output_slices = (6, 6)
    merged.n_slices = 2
    merged.tp_size = 1
    merged.output_ids = (0, 0)
    merged.device = torch.device("cpu")
    merged.create_lora_weights(max_loras=2, lora_config=lora_config)
    assert merged.lora_a_packed is None
    assert merged.lora_b_packed is None
    assert merged._lora_slots == set()

    merged.set_lora(
        1,
        [torch.ones(8, 4, dtype=torch.bfloat16), torch.ones(8, 4, dtype=torch.bfloat16)],
        [torch.ones(6, 8, dtype=torch.bfloat16), torch.ones(6, 8, dtype=torch.bfloat16)],
    )
    assert merged._lora_slots == {1}

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    y = torch.zeros(2, 12, dtype=torch.bfloat16)
    with patch.object(BaseLinearLayerWithLoRA, "_apply_lora_to_output", return_value="upstream") as upstream:
        assert merged._apply_lora_to_output(x, y) == "upstream"
    upstream.assert_called_once_with(x, y)


def test_set_lora_shape_mismatch_reports_module_slot_and_shapes() -> None:
    lora_config = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16)
    layer = object.__new__(AscendRowParallelLinearWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.base_layer = object.__new__(RowParallelLinear)
    layer.base_layer.prefix = "language_model.model.layers.3.self_attn.wo_a"
    layer.n_slices = 1
    layer.input_size = 5120 // 512
    layer.output_size = 8
    layer.tp_size = 1
    layer.device = torch.device("cpu")
    layer.create_lora_weights(max_loras=1, lora_config=lora_config)

    with pytest.raises(ValueError, match="self_attn.wo_a.*slot 0.*\\(8, 12\\)"):
        layer.set_lora(
            0,
            torch.ones(8, 12, dtype=torch.bfloat16),
            torch.ones(8, 8, dtype=torch.bfloat16),
        )


def test_single_linear_set_lora_validates_full_dims_across_tp_and_allows_dummy_warmup() -> None:
    lora_config = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16)

    replicated = object.__new__(AscendReplicatedLinearWithLoRA)
    torch.nn.Module.__init__(replicated)
    replicated.base_layer = object.__new__(ReplicatedLinear)
    replicated.base_layer.prefix = "model.layers.0.self_attn.wq_a"
    replicated.n_slices = 1
    replicated.input_size = 4096
    replicated.output_size = 1024
    replicated.tp_size = 1
    replicated.device = torch.device("cpu")
    replicated.create_lora_weights(max_loras=1, lora_config=lora_config)
    with pytest.raises(ValueError, match="self_attn.wq_a.*in=4096, out=1024"):
        replicated.set_lora(
            0,
            torch.ones(8, 5120, dtype=torch.bfloat16),
            torch.ones(1280, 8, dtype=torch.bfloat16),
        )
    with pytest.raises(ValueError, match="self_attn.wq_a.*in=4096, out=1024"):
        replicated.set_lora(
            0,
            torch.ones(8, 2048, dtype=torch.bfloat16),
            torch.ones(1024, 8, dtype=torch.bfloat16),
        )

    row = object.__new__(AscendRowParallelLinearWithLoRA)
    torch.nn.Module.__init__(row)
    row.base_layer = object.__new__(RowParallelLinear)
    row.base_layer.prefix = "model.layers.0.self_attn.wo_b"
    row.n_slices = 1
    row.input_size = 512
    row.output_size = 4096
    row.tp_size = 8
    row.tp_rank = 0
    row.device = torch.device("cpu")
    row.create_lora_weights(max_loras=1, lora_config=lora_config)
    for bad_in in (5120, 2048):
        with pytest.raises(ValueError, match="self_attn.wo_b.*in=4096, out=4096"):
            row.set_lora(
                0,
                torch.ones(8, bad_in, dtype=torch.bfloat16),
                torch.ones(4096, 8, dtype=torch.bfloat16),
            )
    row.set_lora(
        0,
        torch.ones(8, 4096, dtype=torch.bfloat16),
        torch.ones(4096, 8, dtype=torch.bfloat16),
    )
    row.set_lora(
        0,
        torch.zeros(8, 512, dtype=torch.bfloat16),
        torch.zeros(4096, 8, dtype=torch.bfloat16),
    )

    col = object.__new__(AscendColumnParallelLinearWithLoRA)
    torch.nn.Module.__init__(col)
    col.base_layer = object.__new__(ColumnParallelLinear)
    col.base_layer.prefix = "model.layers.0.self_attn.wo_a"
    col.n_slices = 1
    col.input_size = 1024
    col.output_size = 512
    col.tp_size = 8
    col.tp_rank = 0
    col.is_merged_col_linear = False
    col.device = torch.device("cpu")
    col.create_lora_weights(max_loras=1, lora_config=lora_config)
    for bad_out in (5120, 2048):
        with pytest.raises(ValueError, match="self_attn.wo_a.*in=1024, out=4096"):
            col.set_lora(
                0,
                torch.ones(8, 1024, dtype=torch.bfloat16),
                torch.ones(bad_out, 8, dtype=torch.bfloat16),
            )
    col.set_lora(
        0,
        torch.ones(8, 1024, dtype=torch.bfloat16),
        torch.ones(4096, 8, dtype=torch.bfloat16),
    )
    col.set_lora(
        0,
        torch.zeros(8, 1024, dtype=torch.bfloat16),
        torch.zeros(512, 8, dtype=torch.bfloat16),
    )


def test_warn_unmatched_lora_modules_surfaces_unmatched_and_zero_match_adapters(caplog) -> None:
    class FakeAdapterManager:
        def __init__(self, modules, packed_modules=None):
            self.modules = modules
            self.packed_modules = packed_modules or {}
            self._registered_adapters = {}

        def _get_lora_layer_weights(self, lora_model, module_name):
            return lora_model.loras.get(module_name)

        def _get_text_lora_name(self, lora_model, module_name):
            return None

    request = SimpleNamespace(lora_name="moe-adapter", lora_int_id=1)

    manager_3d_unmatched = FakeAdapterManager(
        modules={"language_model.model.layers.0.mlp.experts": object()},
        packed_modules={
            "language_model.model.layers.0.mlp.experts": [
                "language_model.model.layers.0.mlp.experts.0.w1",
            ]
        },
    )
    manager_3d_unmatched._registered_adapters[1] = SimpleNamespace(
        loras={
            "model.layers.0.ffn.experts.base_layer": object(),
            "model.layers.0.ffn.experts": object(),
        }
    )
    with caplog.at_level("WARNING"):
        unmatched = warn_unmatched_lora_modules(SimpleNamespace(_adapter_manager=manager_3d_unmatched), request)
    assert unmatched == ["model.layers.0.ffn.experts", "model.layers.0.ffn.experts.base_layer"]
    assert "is_3d_lora_weight=true" in caplog.text

    caplog.clear()
    manager_matched = FakeAdapterManager(
        modules={"language_model.model.layers.0.mlp.experts": object()},
        packed_modules={
            "language_model.model.layers.0.mlp.experts": [
                "language_model.model.layers.0.mlp.experts.1.gate_proj",
            ]
        },
    )
    manager_matched._registered_adapters[1] = SimpleNamespace(
        loras={
            "language_model.model.layers.0.mlp.experts": object(),
            # Non-local peer EP subkey listed in packed_modules must not warn.
            "language_model.model.layers.0.mlp.experts.1.gate_proj": object(),
        }
    )
    with caplog.at_level("WARNING"):
        assert warn_unmatched_lora_modules(SimpleNamespace(_adapter_manager=manager_matched), request) == []
    assert caplog.text == ""

    caplog.clear()
    manager_empty = FakeAdapterManager(modules={"language_model.model.layers.0.self_attn.wq_a": object()})
    manager_empty._registered_adapters[1] = SimpleNamespace(loras={})
    with caplog.at_level("WARNING"):
        assert warn_unmatched_lora_modules(SimpleNamespace(_adapter_manager=manager_empty), request) == []
    assert "matched 0 target modules" in caplog.text
