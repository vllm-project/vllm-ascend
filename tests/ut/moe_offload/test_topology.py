# SPDX-License-Identifier: Apache-2.0
"""Topology / slot rotation logic of the offload manager (pure host logic,
verified against both validated model families)."""

import pytest

from vllm_ascend.moe_offload import manager as _mgr_mod


# --------------------------------------------------------------------- #
# compute_folded_topology (pure function)                                #
# --------------------------------------------------------------------- #
class TestFoldedTopology:
    def test_even_count_no_resident_tail(self):
        t = _mgr_mod.compute_folded_topology(offset=1, count=76, slots=2)
        assert type(t) is _mgr_mod.FoldedTopology
        assert t.rotating_moe_layer_count == 76
        assert t.resident_tail_layer is None
        assert t.slots == 2

    def test_odd_count_keeps_last_layer_resident(self):
        # GLM-5.2: 77 MoE layers, 2 slots -> 76 rotating + tail at global 77
        t = _mgr_mod.compute_folded_topology(offset=1, count=77, slots=2)
        assert t.rotating_moe_layer_count == 76
        assert t.resident_tail_layer == 1 + 77 - 1

    def test_ds_43_layers_2_slots(self):
        # DeepSeek-V4-Flash: 43 MoE layers -> 42 rotating + tail at 42
        t = _mgr_mod.compute_folded_topology(offset=0, count=43, slots=2)
        assert t.rotating_moe_layer_count == 42
        assert t.resident_tail_layer == 42

    def test_divisible_by_three_slots(self):
        t = _mgr_mod.compute_folded_topology(offset=2, count=75, slots=3)
        assert t.rotating_moe_layer_count == 75
        assert t.resident_tail_layer is None


# --------------------------------------------------------------------- #
# layer-prefix parsing                                                   #
# --------------------------------------------------------------------- #
class TestParseLayerIdx:
    def test_standard_prefix(self, mgr_glm):
        assert mgr_glm._parse_layer_idx("model.layers.5.mlp") == 5

    def test_nested_expert_prefix(self, mgr_glm):
        assert mgr_glm._parse_layer_idx("model.layers.12.mlp.experts") == 12

    def test_no_layers_token_raises(self, mgr_glm):
        # MTP-style prefix (outside MoE) must raise, not mis-parse
        with pytest.raises(ValueError):
            mgr_glm._parse_layer_idx("mtp.0.self_attn")


# --------------------------------------------------------------------- #
# slot assignment                                                         #
# --------------------------------------------------------------------- #
class TestSlotIdx:
    def test_rotation_2_slots(self, mgr_glm):
        # rel 0 -> slot0, rel 1 -> slot1, rel 2 -> slot0 ...
        assert mgr_glm._slot_idx("model.layers.1.mlp") == 0
        assert mgr_glm._slot_idx("model.layers.2.mlp") == 1
        assert mgr_glm._slot_idx("model.layers.3.mlp") == 0

    def test_resident_tail_forced_to_slot0(self, mgr_glm):
        assert mgr_glm._slot_idx("model.layers.77.mlp") == 0

    def test_out_of_range_layer_raises(self, mgr_glm):
        # layer beyond the rotating range (e.g. an MTP head at 78) must fail
        with pytest.raises(RuntimeError):
            mgr_glm._slot_idx("model.layers.78.mlp")

    def test_ds_shape(self, mgr_ds):
        assert mgr_ds._slot_idx("model.layers.0.mlp") == 0
        assert mgr_ds._slot_idx("model.layers.41.mlp") == 1
        assert mgr_ds._slot_idx("model.layers.42.mlp") == 0  # resident tail


# --------------------------------------------------------------------- #
# same-slot rotation                                                      #
# --------------------------------------------------------------------- #
class TestNextLayerInSlot:
    def test_advance_within_rotation(self, mgr_glm):
        assert mgr_glm._next_layer_in_slot("model.layers.1.mlp") == "model.layers.3.mlp"
        assert mgr_glm._next_layer_in_slot("model.layers.2.mlp") == "model.layers.4.mlp"

    def test_wrap_to_first_occurrence(self, mgr_glm):
        # last rotating slot-0 layer (global 75) wraps to the first slot-0
        # layer (global 1)
        assert mgr_glm._next_layer_in_slot("model.layers.75.mlp") == "model.layers.1.mlp"

    def test_slot3_advance_and_wrap(self, mgr_slot3):
        # 77 layers / 3 slots -> rotating 76, tail 77
        assert mgr_slot3._next_layer_in_slot("model.layers.1.mlp") == "model.layers.4.mlp"
        # rel 75 (global 76) is the last rotating slot-0 layer -> wraps to rel 0
        assert mgr_slot3._next_layer_in_slot("model.layers.76.mlp") == "model.layers.1.mlp"

    def test_resident_tail_next_wraps_to_slot_head(self, mgr_glm):
        # Not reached in production (preload hooks run on rotating layers
        # only); the wrap-to-head behavior is recorded as the current
        # contract rather than left untested.
        assert mgr_glm._next_layer_in_slot("model.layers.77.mlp") == "model.layers.1.mlp"

    def test_unregistered_candidate_raises(self, mgr_glm):
        mgr_glm.layer_prefixes = []  # nothing registered
        with pytest.raises(RuntimeError):
            mgr_glm._next_layer_in_slot("model.layers.1.mlp")


# --------------------------------------------------------------------- #
# managed-range boundary                                                  #
# --------------------------------------------------------------------- #
class TestIsOffloadManaged:
    def test_inside_range(self, mgr_glm):
        assert mgr_glm.is_offload_managed("model.layers.1.mlp")
        assert mgr_glm.is_offload_managed("model.layers.77.mlp")

    def test_outside_range_is_false(self, mgr_glm):
        # dense head (layer 0) and MTP head (layer 78) stay resident
        assert not mgr_glm.is_offload_managed("model.layers.0.mlp")
        assert not mgr_glm.is_offload_managed("model.layers.78.mlp")

    def test_ds_all_layers_managed(self, mgr_ds):
        assert mgr_ds.is_offload_managed("model.layers.0.mlp")
        assert not mgr_ds.is_offload_managed("model.layers.43.mlp")
