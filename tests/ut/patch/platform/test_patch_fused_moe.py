# Copyright Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This file is a part of the vllm-ascend project.
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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.patch.platform import patch_fused_moe


def _set_eplb_config(monkeypatch, *, dynamic_eplb=True, num_redundant_experts=2):
    eplb_config = SimpleNamespace(
        dynamic_eplb=dynamic_eplb,
        expert_map_path=None,
        num_redundant_experts=num_redundant_experts,
    )
    monkeypatch.setattr(
        patch_fused_moe,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=eplb_config),
    )


def test_expert_mapping_loads_all_ascend_redundant_slots(monkeypatch):
    _set_eplb_config(monkeypatch)
    model = MagicMock()
    model.named_parameters.return_value = []

    mapping = patch_fused_moe._ascend_fused_moe_make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=8,
        num_redundant_experts=0,
    )

    assert len(mapping) == (8 + 2) * 3
    assert [entry[2] for entry in mapping[-6:]] == [8, 8, 8, 9, 9, 9]
    assert all("experts.0." in entry[1] for entry in mapping[-6:-3])
    assert all("experts.1." in entry[1] for entry in mapping[-3:])


def test_redundancy_resolver_keeps_upstream_value_without_ascend_eplb(monkeypatch):
    _set_eplb_config(monkeypatch, dynamic_eplb=False, num_redundant_experts=0)

    assert patch_fused_moe._resolve_num_redundant_experts(4) == 4


def test_redundancy_resolver_rejects_conflicting_counts(monkeypatch):
    _set_eplb_config(monkeypatch, num_redundant_experts=2)

    with pytest.raises(
        ValueError,
        match="Conflicting EPLB redundant expert counts: vLLM=4, Ascend=2",
    ):
        patch_fused_moe._resolve_num_redundant_experts(4)
