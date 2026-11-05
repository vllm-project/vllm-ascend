#
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
#

import unittest
from types import SimpleNamespace

# isort: off
import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ced_incremental_save import (
    CED_LAYERWISE_CONFIG_KEY,
    build_ced_trigger_map,
    is_ced_model,
    resolve_ced_layerwise,
)

# isort: on

V41_SOURCES = (2, 8, 14, 20)


def _vllm_config(extra, architectures=("DeepseekV41ForCausalLM",)):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        model_config=SimpleNamespace(architectures=list(architectures)),
    )


def _v41_layout(
    num_layers=40,
    sources=V41_SOURCES,
    with_state=True,
    with_folded=False,
    with_draft=False,
):
    """Production-shaped V4.1 cache-group layer-name layout."""
    groups = {}
    mla_members = []
    for source in sources:
        mla_members.append(f"model.layers.{source}.self_attn.long_kv_cache")
        mla_members.append(f"model.layers.{source}.self_attn.indexer.k_cache")
        if with_folded:
            mla_members.append(f"model.layers.{source}.self_attn.indexer.k_cache_folded")
    groups[0] = mla_members
    next_group = 1
    if with_state:
        groups[next_group] = [f"model.layers.{source}.self_attn.compressor.state_cache" for source in sources[:-1]]
        next_group += 1
    for start in range(0, num_layers, 4):
        groups[next_group] = [f"model.layers.{layer}.self_attn.swa_cache" for layer in range(start, start + 4)]
        next_group += 1
    if with_draft:
        groups[next_group] = [f"model.mtp.{i}.self_attn.swa_cache" for i in range(3)]
    return groups


class TestResolveCEDLayerwise(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertFalse(resolve_ced_layerwise(_vllm_config({})))
        self.assertFalse(resolve_ced_layerwise(_vllm_config({CED_LAYERWISE_CONFIG_KEY: False})))

    def test_enabled_on_v41(self):
        config = _vllm_config({CED_LAYERWISE_CONFIG_KEY: True})
        self.assertTrue(resolve_ced_layerwise(config))

    def test_rejects_use_layerwise_combination(self):
        config = _vllm_config({CED_LAYERWISE_CONFIG_KEY: True, "use_layerwise": True})
        with self.assertRaises(ValueError):
            resolve_ced_layerwise(config)

    def test_ignored_on_non_v41_model(self):
        config = _vllm_config({CED_LAYERWISE_CONFIG_KEY: True}, architectures=("Qwen3ForCausalLM",))
        self.assertFalse(resolve_ced_layerwise(config))

    def test_is_ced_model_architectures(self):
        self.assertTrue(is_ced_model(_vllm_config({}, architectures=("DeepseekV41ForCausalLM",))))
        self.assertTrue(is_ced_model(_vllm_config({}, architectures=("DeepseekV41ForConditionalGeneration",))))
        self.assertFalse(is_ced_model(_vllm_config({}, architectures=("DeepseekV3ForCausalLM",))))


class TestBuildCEDTriggerMap(unittest.TestCase):
    def test_bounded_off_full_layout(self):
        groups = _v41_layout()
        trigger_map = build_ced_trigger_map(groups, V41_SOURCES)

        mla_slices = trigger_map.slices_by_group[0]
        self.assertEqual(
            [(s.member_start, s.member_end, s.trigger_layer) for s in mla_slices],
            [(0, 2, 2), (2, 4, 8), (4, 6, 14), (6, 8, 20)],
        )
        # The compressor-state group keeps the step-end bulk save.
        self.assertNotIn(1, trigger_map.slices_by_group)
        # Ten SWA groups, one slice each, triggered by the group-tail layer.
        for group_id in range(2, 12):
            tail_layer = (group_id - 2) * 4 + 3
            self.assertEqual(len(trigger_map.slices_by_group[group_id]), 1)
            self.assertEqual(trigger_map.slices_by_group[group_id][0].trigger_layer, tail_layer)
            self.assertEqual(trigger_map.publish_layers[group_id], tail_layer)
        self.assertEqual(trigger_map.publish_layers[0], 20)
        self.assertEqual(
            set(trigger_map.slices_by_trigger),
            {2, 8, 14, 20} | {3, 7, 11, 15, 19, 23, 27, 31, 35, 39},
        )
        # Slices of the MLA group cover the full member list without gaps.
        covered = [member for s in mla_slices for member in range(s.member_start, s.member_end)]
        self.assertEqual(covered, list(range(len(groups[0]))))

    def test_bounded_on_only_mla_group_pooled(self):
        groups = {0: _v41_layout()[0]}
        trigger_map = build_ced_trigger_map(groups, V41_SOURCES)
        self.assertEqual(set(trigger_map.slices_by_group), {0})
        self.assertEqual(trigger_map.trigger_layers(), (2, 8, 14, 20))

    def test_folded_indexer_stays_inside_slot_slice(self):
        groups = _v41_layout(with_folded=True)
        trigger_map = build_ced_trigger_map({0: groups[0]}, V41_SOURCES)
        self.assertEqual(
            [(s.member_start, s.member_end, s.trigger_layer) for s in trigger_map.slices_by_group[0]],
            [(0, 3, 2), (3, 6, 8), (6, 9, 14), (9, 12, 20)],
        )

    def test_draft_group_keeps_bulk_save(self):
        groups = _v41_layout(with_draft=True)
        trigger_map = build_ced_trigger_map(groups, V41_SOURCES)
        self.assertNotIn(12, trigger_map.slices_by_group)
        self.assertEqual(len(trigger_map.slices_by_group), 11)

    def test_non_ced_layout_yields_empty_map(self):
        groups = {
            0: [f"model.layers.{i}.self_attn.kv_cache" for i in range(8)],
            1: [f"model.layers.{i}.self_attn.kv_cache" for i in range(8, 16)],
        }
        trigger_map = build_ced_trigger_map(groups, V41_SOURCES)
        self.assertEqual(trigger_map.slices_by_group, {})
        self.assertEqual(trigger_map.slices_by_trigger, {})
        self.assertEqual(trigger_map.publish_layers, {})

    def test_partial_source_coverage_reports_missing_slots(self):
        groups = {0: _v41_layout()[0][:4]}
        with self.assertLogs(level="WARNING") as captured:
            trigger_map = build_ced_trigger_map(groups, V41_SOURCES)
        self.assertEqual(len(trigger_map.slices_by_group[0]), 2)
        self.assertIn("no members", "; ".join(captured.output))


if __name__ == "__main__":
    unittest.main()
