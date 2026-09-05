#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#
"""Unit tests for vllm_ascend.ops.fused_moe.moe_stage_params.

The module is a thin layer of frozen-slots dataclasses that gate quant-path
selection across the fused MoE pipeline. The existing
``test_moe_runtime_args.py`` only covers them indirectly via the builders, so
the ``@property`` branches and dataclass invariants are tested directly here.
"""

from __future__ import annotations

import dataclasses
import unittest

import torch

from vllm_ascend.ops.fused_moe.moe_stage_params import (
    MoEMxfpParams,
    MoEQuantParams,
    MoERoutingParams,
)
from vllm_ascend.quantization.quant_type import QuantType


def _make_routing(**overrides) -> MoERoutingParams:
    defaults: dict = {
        "expert_map": None,
        "global_redundant_expert_num": 0,
        "mc2_mask": None,
        "apply_router_weight_on_input": False,
    }
    defaults.update(overrides)
    return MoERoutingParams(**defaults)


class TestMoERoutingParams(unittest.TestCase):
    def test_defaults_for_optional_fields(self):
        routing = _make_routing()
        self.assertIsNone(routing.expert_map)
        self.assertEqual(routing.global_redundant_expert_num, 0)
        self.assertIsNone(routing.mc2_mask)
        self.assertFalse(routing.apply_router_weight_on_input)
        self.assertIsNone(routing.log2phy)
        self.assertIsNone(routing.pertoken_scale)

    def test_carries_optional_tensors_by_identity(self):
        expert_map = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        mc2_mask = torch.tensor([True, False, True, False])
        log2phy = torch.tensor([3, 2, 1, 0], dtype=torch.int32)
        pertoken_scale = torch.randn(8)

        routing = _make_routing(
            expert_map=expert_map,
            mc2_mask=mc2_mask,
            log2phy=log2phy,
            pertoken_scale=pertoken_scale,
            global_redundant_expert_num=2,
            apply_router_weight_on_input=True,
        )

        self.assertIs(routing.expert_map, expert_map)
        self.assertIs(routing.mc2_mask, mc2_mask)
        self.assertIs(routing.log2phy, log2phy)
        self.assertIs(routing.pertoken_scale, pertoken_scale)
        self.assertEqual(routing.global_redundant_expert_num, 2)
        self.assertTrue(routing.apply_router_weight_on_input)

    def test_frozen_dataclass_rejects_reassignment(self):
        routing = _make_routing()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            routing.global_redundant_expert_num = 1  # type: ignore[misc]

    def test_slots_dataclass_rejects_new_attributes(self):
        # frozen+slots dataclasses raise FrozenInstanceError in canonical CPython
        # but observed builds vary: AttributeError (slots only) or TypeError
        # (when the synthesised __setattr__ chains through super()). Any of
        # those is acceptable — the invariant is "no attribute pinning".
        routing = _make_routing()
        with self.assertRaises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
            routing.unexpected_attr = "x"  # type: ignore[attr-defined]


class TestMoEMxfpParams(unittest.TestCase):
    def test_all_fields_default(self):
        mxfp = MoEMxfpParams()
        self.assertIsNone(mxfp.act_quant_type)
        self.assertIsNone(mxfp.weight_quant_type)
        self.assertIsNone(mxfp.scale_dtype)
        self.assertIsNone(mxfp.per_token_scale_dtype)
        self.assertTrue(mxfp.use_bf16)

    def test_carries_explicit_dtypes(self):
        mxfp = MoEMxfpParams(
            act_quant_type=torch.float8_e4m3fn,
            weight_quant_type=torch.float8_e4m3fn,
            scale_dtype=torch.float32,
            per_token_scale_dtype=torch.float16,
            use_bf16=False,
        )
        self.assertEqual(mxfp.act_quant_type, torch.float8_e4m3fn)
        self.assertEqual(mxfp.weight_quant_type, torch.float8_e4m3fn)
        self.assertEqual(mxfp.scale_dtype, torch.float32)
        self.assertEqual(mxfp.per_token_scale_dtype, torch.float16)
        self.assertFalse(mxfp.use_bf16)

    def test_frozen_dataclass_rejects_reassignment(self):
        mxfp = MoEMxfpParams()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            mxfp.use_bf16 = False  # type: ignore[misc]


class TestMoEQuantParamsDefaults(unittest.TestCase):
    def test_default_is_dense_path(self):
        quant = MoEQuantParams()
        self.assertEqual(quant.quant_type, QuantType.NONE)
        self.assertIsNone(quant.comm_quant_mode)
        self.assertIsNone(quant.mxfp)
        self.assertFalse(quant.is_quant)
        self.assertFalse(quant.is_mxfp)
        self.assertFalse(quant.is_int_quant)
        self.assertFalse(quant.dispatch_with_quant)
        # is_per_channel_weight and use_w4a8_per_channel_gmm_swiglu were added
        # alongside W4A8 per-channel kernels; guard for older builds.
        if hasattr(quant, "is_per_channel_weight"):
            self.assertFalse(quant.is_per_channel_weight)
        if hasattr(quant, "use_w4a8_per_channel_gmm_swiglu"):
            self.assertFalse(quant.use_w4a8_per_channel_gmm_swiglu)

    def test_frozen_dataclass_rejects_reassignment(self):
        quant = MoEQuantParams()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            quant.quant_type = QuantType.W8A8  # type: ignore[misc]


def _resolve_quant_types(*names: str) -> set:
    """Return the subset of QuantType members that currently exist by name.

    QuantType has grown over releases (e.g. ``MXFP4`` and ``W4A8MXFP`` were
    added later). Resolving by name keeps the truth tables forward-compatible
    with future additions and tolerant of running against older installs that
    have fewer members.
    """

    resolved = set()
    for name in names:
        member = getattr(QuantType, name, None)
        if member is not None:
            resolved.add(member)
    return resolved


class TestMoEQuantParamsProperties(unittest.TestCase):
    """Truth table for the five ``@property`` branches.

    The MoE runtime gates dispatch vs. dense paths on these booleans, so each
    QuantType enumerator must land in exactly the right buckets.
    """

    IS_QUANT = {qt for qt in QuantType if qt != QuantType.NONE}
    IS_MXFP = _resolve_quant_types("MXFP8", "MXFP4", "W4A8MXFP")
    IS_INT_QUANT = _resolve_quant_types("W8A8", "W4A8")
    DISPATCH_WITH_QUANT = _resolve_quant_types(
        "W8A8",
        "W4A8",
        "MXFP8",
        "MXFP4",
        "W4A8MXFP",
    )

    def test_is_quant_matches_truth_table(self):
        for qt in QuantType:
            with self.subTest(quant_type=qt):
                quant = MoEQuantParams(quant_type=qt)
                self.assertEqual(quant.is_quant, qt in self.IS_QUANT)

    def test_is_mxfp_matches_truth_table(self):
        for qt in QuantType:
            with self.subTest(quant_type=qt):
                quant = MoEQuantParams(quant_type=qt)
                self.assertEqual(quant.is_mxfp, qt in self.IS_MXFP)

    def test_is_int_quant_matches_truth_table(self):
        for qt in QuantType:
            with self.subTest(quant_type=qt):
                quant = MoEQuantParams(quant_type=qt)
                self.assertEqual(quant.is_int_quant, qt in self.IS_INT_QUANT)

    def test_dispatch_with_quant_matches_truth_table(self):
        for qt in QuantType:
            with self.subTest(quant_type=qt):
                quant = MoEQuantParams(quant_type=qt)
                self.assertEqual(quant.dispatch_with_quant, qt in self.DISPATCH_WITH_QUANT)

    def test_use_w4a8_per_channel_gmm_swiglu_requires_both_flags(self):
        # Only W4A8 + per-channel weight should toggle the SWIGLU GMM kernel.
        # The is_per_channel_weight field arrived with the W4A8 per-channel
        # kernels; skip if running against an older build that lacks it.
        if "is_per_channel_weight" not in MoEQuantParams.__dataclass_fields__:
            self.skipTest("MoEQuantParams.is_per_channel_weight not available in this build")

        quant_on = MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=True)
        self.assertTrue(quant_on.use_w4a8_per_channel_gmm_swiglu)

        quant_off_no_flag = MoEQuantParams(quant_type=QuantType.W4A8, is_per_channel_weight=False)
        self.assertFalse(quant_off_no_flag.use_w4a8_per_channel_gmm_swiglu)

        # Per-channel flag alone is not enough; the quant type must be W4A8.
        for qt in QuantType:
            if qt == QuantType.W4A8:
                continue
            with self.subTest(quant_type=qt):
                quant = MoEQuantParams(quant_type=qt, is_per_channel_weight=True)
                self.assertFalse(quant.use_w4a8_per_channel_gmm_swiglu)

    def test_int_quant_and_mxfp_are_disjoint(self):
        # Sanity invariant: a quant type is never simultaneously int-quant and mxfp.
        for qt in QuantType:
            with self.subTest(quant_type=qt):
                quant = MoEQuantParams(quant_type=qt)
                self.assertFalse(quant.is_int_quant and quant.is_mxfp)

    def test_dispatch_with_quant_is_superset_of_int_and_mxfp(self):
        for qt in QuantType:
            with self.subTest(quant_type=qt):
                quant = MoEQuantParams(quant_type=qt)
                if quant.is_int_quant or quant.is_mxfp:
                    self.assertTrue(quant.dispatch_with_quant)

    def test_w4a16_is_quant_but_not_dispatch_with_quant(self):
        # W4A16 quantises weights only; activation stays bf16/fp16 so dispatch
        # carries dense activations, not per-token scales.
        quant = MoEQuantParams(quant_type=QuantType.W4A16)
        self.assertTrue(quant.is_quant)
        self.assertFalse(quant.is_int_quant)
        self.assertFalse(quant.is_mxfp)
        self.assertFalse(quant.dispatch_with_quant)

    def test_carries_optional_mxfp_leaf_by_identity(self):
        if not hasattr(QuantType, "MXFP8"):
            self.skipTest("QuantType.MXFP8 not available in this build")
        mxfp_leaf = MoEMxfpParams(
            act_quant_type=torch.float8_e4m3fn,
            weight_quant_type=torch.float8_e4m3fn,
        )
        quant = MoEQuantParams(quant_type=QuantType.MXFP8, mxfp=mxfp_leaf)
        self.assertIs(quant.mxfp, mxfp_leaf)
        self.assertTrue(quant.is_mxfp)


class TestModuleExports(unittest.TestCase):
    def test_all_lists_only_public_dataclasses(self):
        import vllm_ascend.ops.fused_moe.moe_stage_params as stage_params

        self.assertEqual(
            set(stage_params.__all__),
            {"MoERoutingParams", "MoEMxfpParams", "MoEQuantParams"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
