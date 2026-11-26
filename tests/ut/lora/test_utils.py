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
from vllm.lora.layers import (
    ColumnParallelLinearWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    LoRAMapping,
    MergedColumnParallelLinearVariableSliceWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA,
    ReplicatedLinearWithLoRA,
    RowParallelLinearWithLoRA,
    RowParallelLinearWithShardedLoRA,
)
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.layers.fused_moe import FusedMoE3DWithLoRA, FusedMoEWithLoRA
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.custom_op import maybe_get_oot_by_class
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.models import SupportsLoRA

from vllm_ascend.lora.fused_moe import AscendFusedMoE3DWithLoRA, AscendFusedMoEWithLoRA
from vllm_ascend.lora.punica_npu import PunicaWrapperNPU
from vllm_ascend.lora.utils import (
    AscendColumnParallelLinearWithLoRA,
    AscendMergedColumnParallelLinearWithLoRA,
    AscendMergedQKVParallelLinearWithLoRA,
    AscendReplicatedLinearWithLoRA,
    AscendRowParallelLinearWithLoRA,
    _expand_skipped_module_names,
    install_dp_moe_lora_load_guard,
    is_lora_load_guard_installed,
    refresh_all_lora_classes,
    warn_unmatched_lora_modules,
)
from vllm_ascend.ops.linear import AscendQKVParallelLinear


def test_refresh_all_lora_classes_prepends_ascend_wrappers() -> None:
    sentinel = object()
    original = lora_utils._all_lora_classes
    try:
        lora_utils._all_lora_classes = (FusedMoEWithLoRA, sentinel, FusedMoE3DWithLoRA)
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
        assert classes[len(expected_prefix) :] == [sentinel]
        assert FusedMoEWithLoRA not in classes
        assert FusedMoE3DWithLoRA not in classes
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


def test_refresh_all_lora_classes_replaces_upstream_linear_fallbacks_with_guards() -> None:
    original = lora_utils._all_lora_classes
    upstream_linears = (
        ColumnParallelLinearWithLoRA,
        ColumnParallelLinearWithShardedLoRA,
        MergedColumnParallelLinearWithLoRA,
        MergedColumnParallelLinearWithShardedLoRA,
        MergedColumnParallelLinearVariableSliceWithLoRA,
        QKVParallelLinearWithLoRA,
        QKVParallelLinearWithShardedLoRA,
        MergedQKVParallelLinearWithLoRA,
        MergedQKVParallelLinearWithShardedLoRA,
        RowParallelLinearWithLoRA,
        RowParallelLinearWithShardedLoRA,
        ReplicatedLinearWithLoRA,
    )
    try:
        lora_utils._all_lora_classes = (
            *upstream_linears,
            FusedMoEWithLoRA,
            FusedMoE3DWithLoRA,
        )
        refresh_all_lora_classes()
        first_classes = list(lora_utils._all_lora_classes)
        # Running refresh twice must remain idempotent and never wrap twice.
        refresh_all_lora_classes()
        classes = list(lora_utils._all_lora_classes)
        assert classes == first_classes

        for unguarded_cls in upstream_linears:
            assert unguarded_cls not in classes
        assert FusedMoEWithLoRA not in classes
        assert FusedMoE3DWithLoRA not in classes

        guarded_map = {
            orig_cls: next(
                cls
                for cls in classes
                if cls.__bases__ == (orig_cls,) and getattr(cls, "_ascend_unsupported_lora_guarded", False)
            )
            for orig_cls in upstream_linears
        }

        normal_col = object.__new__(maybe_get_oot_by_class(ColumnParallelLinear))
        unsupported_wo_a = object.__new__(maybe_get_oot_by_class(ColumnParallelLinear))
        unsupported_wo_a._ascend_dsa_n_local_groups = 2
        single_group_wo_a = object.__new__(maybe_get_oot_by_class(ColumnParallelLinear))
        single_group_wo_a._ascend_dsa_n_local_groups = 1

        normal_merged = object.__new__(maybe_get_oot_by_class(MergedColumnParallelLinear))
        normal_merged.output_sizes = [8, 8, 8]
        unsupported_merged = object.__new__(maybe_get_oot_by_class(MergedColumnParallelLinear))
        unsupported_merged.output_sizes = [8, 8, 8]
        unsupported_merged._ascend_lora_unsupported_reason = "merged projection unsupported"

        unsupported_qkv = object.__new__(maybe_get_oot_by_class(QKVParallelLinear))
        unsupported_qkv._ascend_lora_unsupported_reason = "qkv projection unsupported"
        unsupported_ascend_qkv = object.__new__(AscendQKVParallelLinear)
        unsupported_ascend_qkv._ascend_lora_unsupported_reason = "ascend qkv unsupported"

        normal_row = object.__new__(maybe_get_oot_by_class(RowParallelLinear))
        unsupported_row = object.__new__(maybe_get_oot_by_class(RowParallelLinear))
        unsupported_row._ascend_lora_unsupported_reason = "row projection unsupported"

        unsupported_rep = object.__new__(maybe_get_oot_by_class(ReplicatedLinear))
        unsupported_rep._ascend_lora_unsupported_reason = "compressor projections unsupported"

        default_cfg = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16)
        sharded_cfg = LoRAConfig(
            max_loras=1,
            max_lora_rank=8,
            lora_dtype=torch.bfloat16,
            fully_sharded_loras=True,
        )

        assert AscendColumnParallelLinearWithLoRA.can_replace_layer(
            source_layer=normal_col, lora_config=default_cfg, packed_modules_list=["wo_a"], model_config=None
        )
        assert not AscendColumnParallelLinearWithLoRA.can_replace_layer(
            source_layer=unsupported_wo_a, lora_config=default_cfg, packed_modules_list=["wo_a"], model_config=None
        )
        assert not AscendMergedColumnParallelLinearWithLoRA.can_replace_layer(
            source_layer=unsupported_merged,
            lora_config=default_cfg,
            packed_modules_list=["gate_proj", "up_proj"],
            model_config=None,
        )
        assert not AscendMergedQKVParallelLinearWithLoRA.can_replace_layer(
            source_layer=unsupported_ascend_qkv,
            lora_config=default_cfg,
            packed_modules_list=["q_proj", "k_proj", "v_proj"],
            model_config=None,
        )
        assert not guarded_map[ColumnParallelLinearWithLoRA].can_replace_layer(
            source_layer=unsupported_wo_a, lora_config=default_cfg, packed_modules_list=["wo_a"], model_config=None
        )
        assert guarded_map[ColumnParallelLinearWithShardedLoRA].can_replace_layer(
            source_layer=normal_col, lora_config=sharded_cfg, packed_modules_list=["k_proj"], model_config=None
        )
        assert not guarded_map[ColumnParallelLinearWithShardedLoRA].can_replace_layer(
            source_layer=single_group_wo_a, lora_config=sharded_cfg, packed_modules_list=["wo_a"], model_config=None
        )
        assert guarded_map[MergedColumnParallelLinearWithLoRA].can_replace_layer(
            source_layer=normal_merged,
            lora_config=default_cfg,
            packed_modules_list=["gate_proj", "up_proj"],
            model_config=None,
        )
        assert not guarded_map[MergedColumnParallelLinearWithLoRA].can_replace_layer(
            source_layer=unsupported_merged,
            lora_config=default_cfg,
            packed_modules_list=["gate_proj", "up_proj"],
            model_config=None,
        )
        assert not guarded_map[MergedColumnParallelLinearWithShardedLoRA].can_replace_layer(
            source_layer=unsupported_merged,
            lora_config=sharded_cfg,
            packed_modules_list=["gate_proj", "up_proj"],
            model_config=None,
        )
        assert guarded_map[MergedColumnParallelLinearVariableSliceWithLoRA].can_replace_layer(
            source_layer=normal_merged,
            lora_config=default_cfg,
            packed_modules_list=["q", "k", "v"],
            model_config=None,
        )
        assert not guarded_map[MergedColumnParallelLinearVariableSliceWithLoRA].can_replace_layer(
            source_layer=unsupported_merged,
            lora_config=default_cfg,
            packed_modules_list=["q", "k", "v"],
            model_config=None,
        )
        assert not guarded_map[QKVParallelLinearWithLoRA].can_replace_layer(
            source_layer=unsupported_qkv, lora_config=default_cfg, packed_modules_list=["qkv"], model_config=None
        )
        assert not guarded_map[QKVParallelLinearWithShardedLoRA].can_replace_layer(
            source_layer=unsupported_qkv, lora_config=sharded_cfg, packed_modules_list=["qkv"], model_config=None
        )
        assert not guarded_map[MergedQKVParallelLinearWithLoRA].can_replace_layer(
            source_layer=unsupported_qkv,
            lora_config=default_cfg,
            packed_modules_list=["q_proj", "k_proj", "v_proj"],
            model_config=None,
        )
        assert not guarded_map[MergedQKVParallelLinearWithShardedLoRA].can_replace_layer(
            source_layer=unsupported_qkv,
            lora_config=sharded_cfg,
            packed_modules_list=["q_proj", "k_proj", "v_proj"],
            model_config=None,
        )
        assert guarded_map[RowParallelLinearWithLoRA].can_replace_layer(
            source_layer=normal_row, lora_config=default_cfg, packed_modules_list=[], model_config=None
        )
        assert not guarded_map[RowParallelLinearWithLoRA].can_replace_layer(
            source_layer=unsupported_row, lora_config=default_cfg, packed_modules_list=[], model_config=None
        )
        assert guarded_map[RowParallelLinearWithShardedLoRA].can_replace_layer(
            source_layer=normal_row, lora_config=sharded_cfg, packed_modules_list=[], model_config=None
        )
        assert not guarded_map[RowParallelLinearWithShardedLoRA].can_replace_layer(
            source_layer=unsupported_row, lora_config=sharded_cfg, packed_modules_list=[], model_config=None
        )
        assert not AscendReplicatedLinearWithLoRA.can_replace_layer(
            source_layer=unsupported_rep, lora_config=default_cfg, packed_modules_list=["wkv"], model_config=None
        )
        assert not guarded_map[ReplicatedLinearWithLoRA].can_replace_layer(
            source_layer=unsupported_rep, lora_config=default_cfg, packed_modules_list=["wkv"], model_config=None
        )
    finally:
        lora_utils._all_lora_classes = original


def test_expand_skipped_module_names_normalizes_mm_prefix_without_trailing_dot() -> None:
    model = SimpleNamespace(get_mm_mapping=lambda: SimpleNamespace(language_model=["language_model"]))
    candidates = _expand_skipped_module_names(
        model,
        SimpleNamespace(),
        "language_model.model.layers.0.self_attn.compressor.wkv",
    )
    assert "model.layers.0.self_attn.compressor.wkv" in candidates
    assert "layers.0.self_attn.compressor.wkv" in candidates
    assert ".model.layers.0.self_attn.compressor.wkv" not in candidates


def test_install_dp_moe_lora_load_guard_warns_on_offline_and_lru_reload_without_duplicate_in_add_lora(
    caplog,
) -> None:
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    class TinyToyModel(torch.nn.Module, SupportsLoRA):
        config = None
        is_3d_moe_weight = False
        is_non_gated_moe = False
        embedding_modules: dict[str, str] = {}
        packed_modules_mapping: dict[str, list[str]] = {}
        lora_skip_prefixes: list[str] = []

        def __init__(self) -> None:
            super().__init__()
            replicated_cls = maybe_get_oot_by_class(ReplicatedLinear)
            wkv = object.__new__(replicated_cls)
            torch.nn.Module.__init__(wkv)
            wkv.input_size = 8
            wkv.output_size = 8
            wkv.output_partition_sizes = [8]
            wkv.input_size_per_partition = 8
            wkv.tp_size = 1
            wkv.tp_rank = 0
            wkv.skip_bias_add = False
            wkv.return_bias = False
            wkv.bias = None
            wkv.prefix = "model.layers.0.self_attn.wkv"
            wkv.quant_method = None
            wkv.weight = torch.nn.Parameter(torch.zeros(8, 8, dtype=torch.bfloat16))
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].self_attn = torch.nn.Module()
            self.model.layers[0].self_attn.wkv = wkv

    def _make_lora_weights(module_name: str) -> LoRALayerWeights:
        return LoRALayerWeights(
            module_name=module_name,
            rank=8,
            lora_alpha=8,
            lora_a=torch.ones(8, 8, dtype=torch.bfloat16),
            lora_b=torch.ones(8, 8, dtype=torch.bfloat16),
        )

    original = lora_utils._all_lora_classes
    try:
        refresh_all_lora_classes()
        lora_config = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.bfloat16, max_cpu_loras=1)
        vllm_config = SimpleNamespace(
            lora_config=lora_config,
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
            scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(max_position_embeddings=16)),
                get_vocab_size=lambda: 32,
                is_encoder_decoder=False,
            ),
        )
        punica_wrapper = object.__new__(PunicaWrapperNPU)
        PunicaWrapperBase.__init__(punica_wrapper, 16, 2, "cpu")
        punica_wrapper._single_lora_mask = None
        worker_manager = LRUCacheWorkerLoRAManager(vllm_config, torch.device("cpu"), embedding_modules={})
        with patch("vllm.lora.model_manager.get_punica_wrapper", return_value=punica_wrapper):
            worker_manager.create_lora_manager(TinyToyModel(), vllm_config)

        # Even with DP=1 and zero skipped unsupported modules, the load guard must install.
        install_dp_moe_lora_load_guard(worker_manager)
        assert is_lora_load_guard_installed(worker_manager)

        adapter_by_id = {
            1: LoRAModel(
                lora_model_id=1,
                rank=8,
                loras={
                    "model.layers.0.self_attn.wkv": _make_lora_weights("model.layers.0.self_attn.wkv"),
                    "model.layers.0.attn.missing_proj": _make_lora_weights("model.layers.0.attn.missing_proj"),
                },
            ),
            2: LoRAModel(
                lora_model_id=2,
                rank=8,
                loras={
                    "model.layers.0.self_attn.wkv": _make_lora_weights("model.layers.0.self_attn.wkv"),
                },
            ),
        }
        req_unmatched = LoRARequest(lora_name="adapter-1", lora_int_id=1, lora_path="/unused/1")
        req_clean = LoRARequest(lora_name="adapter-2", lora_int_id=2, lora_path="/unused/2")
        req_dummy = LoRARequest(lora_name="warmup-dummy", lora_int_id=3, lora_path="/unused/3")

        with patch.object(
            worker_manager._load_adapter,
            "_orig_load_adapter",
            side_effect=lambda req: adapter_by_id[req.lora_int_id].clone(req.lora_int_id),
        ):
            mapping_1 = LoRAMapping((1,), (1,))
            mapping_2 = LoRAMapping((2,), (2,))

            # Offline LLM.generate path: set_active_adapters -> _apply_adapters -> add_adapter -> _load_adapter
            caplog.clear()
            with caplog.at_level("WARNING"):
                worker_manager.set_active_adapters({req_unmatched}, mapping_1)
            first_warnings = [rec.message for rec in caplog.records if "missing_proj" in rec.message]
            assert len(first_warnings) == 1

            # Reusing the cached adapter does not reload or duplicate warnings.
            caplog.clear()
            with caplog.at_level("WARNING"):
                worker_manager.set_active_adapters({req_unmatched}, mapping_1)
            assert [rec for rec in caplog.records if "missing_proj" in rec.message] == []

            # Loading a second adapter evicts adapter 1 (max_cpu_loras=1); reactivating
            # adapter 1 lazily reloads via _DPMoELoadGuard and warns again.
            caplog.clear()
            with caplog.at_level("WARNING"):
                worker_manager.set_active_adapters({req_clean}, mapping_2)
            assert [rec for rec in caplog.records if "did not match" in rec.message or "0 target" in rec.message] == []
            assert 1 not in worker_manager.list_adapters()

            caplog.clear()
            with caplog.at_level("WARNING"):
                worker_manager.set_active_adapters({req_unmatched}, mapping_1)
            reload_warnings = [rec.message for rec in caplog.records if "missing_proj" in rec.message]
            assert len(reload_warnings) == 1

            # Warmup dummy LoRA does not load via _load_adapter and must not emit warnings.
            worker_manager.remove_all_adapters()
            caplog.clear()
            with caplog.at_level("WARNING"):
                worker_manager.add_dummy_lora(req_dummy, rank=8)
            assert [rec for rec in caplog.records if "did not match" in rec.message or "0 target" in rec.message] == []

            # Online NPUModelRunner.add_lora must warn once via _DPMoELoadGuard (never duplicate).
            worker_manager.remove_all_adapters()
            runner = object.__new__(NPUModelRunner)
            runner.lora_manager = worker_manager
            caplog.clear()
            with caplog.at_level("WARNING"):
                assert runner.add_lora(req_unmatched) is True
            add_lora_warnings = [rec.message for rec in caplog.records if "missing_proj" in rec.message]
            assert len(add_lora_warnings) == 1

            # Fallback warning still fires if a caller bypassed load_lora_model and has no load guard.
            unguarded_manager = SimpleNamespace(
                _adapter_manager=worker_manager._adapter_manager,
                add_adapter=lambda _req: True,
            )
            runner.lora_manager = unguarded_manager
            caplog.clear()
            with caplog.at_level("WARNING"):
                assert runner.add_lora(req_unmatched) is True
            fallback_warnings = [rec.message for rec in caplog.records if "missing_proj" in rec.message]
            assert len(fallback_warnings) == 1
    finally:
        lora_utils._all_lora_classes = original
