# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

from vllm_ascend.core.kv_cache_pool_plan import (
    CachePoolPlan,
    CachePoolSpec,
    RequestStateMemoryBudget,
    RequestStatePoolPlan,
    plan_request_state_pools,
)


class TestRequestStatePoolPlan(unittest.TestCase):
    def setUp(self):
        # Two history groups share page IDs but have separate tensor backings.
        self.history = CachePoolSpec("history", (0, 1), "token_block", 64)
        self.swa = CachePoolPlan(CachePoolSpec("swa", (2,), "request_slot", 32), 5)
        self.compressor = CachePoolPlan(CachePoolSpec("compressor", (3,), "request_slot", 16), 5)
        self.states = (self.swa, self.compressor)

    def plan(self, available=1000, **kwargs):
        return plan_request_state_pools(
            self.history, self.states, available, min_history_blocks=2, workspace_bytes=128, **kwargs
        )

    def test_fixed_state_reservation_and_page_rounding(self):
        plan = self.plan()
        # 5 slots include the null slot: 240 fixed state + 128 workspace.
        self.assertEqual(plan.budget.fixed_bytes, 368)
        self.assertEqual(plan.history.capacity, 9)
        self.assertEqual(plan.size_bytes, 944)
        self.assertEqual(1000 - plan.size_bytes, 56)
        self.assertEqual(plan.pool_for_group(0), plan.pool_for_group(1))
        self.assertEqual(plan.pool_for_group(2).capacity, 5)
        plan.validate_group_coverage(4)

    def test_exact_fit_and_one_byte_short(self):
        self.assertEqual(self.plan(496).history.capacity, 2)
        with self.assertRaisesRegex(ValueError, "below required minimum"):
            self.plan(495)

    def test_fixed_pools_do_not_fit(self):
        with self.assertRaises(ValueError):
            self.plan(367)
        # An override must not bypass fixed state/workspace accounting.
        with self.assertRaisesRegex(ValueError, "requires 496 bytes"):
            self.plan(367, num_history_blocks=2)

    def test_override_uses_full_budget(self):
        self.assertEqual(self.plan(num_history_blocks=4).size_bytes, 624)
        with self.assertRaisesRegex(ValueError, "requires 1008 bytes"):
            self.plan(num_history_blocks=10)
        with self.assertRaisesRegex(ValueError, "below required minimum"):
            self.plan(num_history_blocks=1)

    def test_budget_sweep_fits_and_maximizes_capacity(self):
        # Check the affine bound independently rather than duplicating the
        # planner's integer-division implementation.
        for available in range(496, 1100):
            with self.subTest(available=available):
                plan = self.plan(available)
                self.assertLessEqual(plan.size_bytes, available)
                self.assertGreater(plan.size_bytes + 64, available)
                self.assertEqual(plan.request_states, self.states)

    def test_concurrency_reserves_state_without_growing_history_padding(self):
        larger_states = tuple(replace(pool, capacity=9) for pool in self.states)
        plan = plan_request_state_pools(self.history, larger_states, 1000, min_history_blocks=2, workspace_bytes=128)
        self.assertEqual(plan.history.capacity, 6)
        self.assertEqual(plan.budget.request_state_bytes, 432)
        self.assertEqual(plan.budget.history_page_bytes, 64)

    def test_rank_minimum_is_revalidated_per_worker(self):
        plans = [self.plan(1000), self.plan(800)]
        agreed = min(plan.history.capacity for plan in plans)
        self.assertEqual(agreed, 6)
        for available in (1000, 800):
            plan = self.plan(available, num_history_blocks=agreed)
            self.assertEqual(plan.size_bytes, 752)
        with self.assertRaises(ValueError):
            self.plan(800, num_history_blocks=plans[0].history.capacity)

    def test_failure_does_not_mutate_inputs(self):
        before = deepcopy((self.history, self.states))
        with self.assertRaises(ValueError):
            self.plan(0)
        self.assertEqual(deepcopy((self.history, self.states)), before)
        self.assertEqual(self.plan().history.capacity, 9)

    def test_plan_copy_and_immutability(self):
        plan = self.plan()
        self.assertEqual(deepcopy(plan), plan)
        with self.assertRaises(FrozenInstanceError):
            plan.workspace_bytes = 0

    def test_invalid_specs(self):
        for kwargs in (
            {"pool_id": ""},
            {"pool_id": " "},
            {"group_ids": ()},
            {"group_ids": [0]},
            {"group_ids": (0, 0)},
            {"group_ids": (-1,)},
            {"group_ids": (True,)},
            {"allocation_unit": "bytes"},
            {"bytes_per_unit": 0},
            {"bytes_per_unit": -1},
            {"bytes_per_unit": 1.5},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.history, **kwargs)

    def test_invalid_capacities(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.swa, capacity=value)

    def test_invalid_pool_ownership(self):
        for pool in (
            replace(self.swa, spec=replace(self.swa.spec, group_ids=(1,))),
            replace(self.swa, spec=replace(self.swa.spec, pool_id="history")),
            replace(self.swa, spec=replace(self.swa.spec, allocation_unit="token_block")),
        ):
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                plan_request_state_pools(self.history, (pool,), 1000, min_history_blocks=2)
        with self.assertRaises(ValueError):
            plan_request_state_pools(self.swa.spec, self.states, 1000, min_history_blocks=2)
        with self.assertRaises(ValueError):
            plan_request_state_pools(self.history, (), 1000, min_history_blocks=2)
        with self.assertRaises(ValueError):
            plan_request_state_pools(self.history, list(self.states), 1000, min_history_blocks=2)

    def test_group_coverage_and_lookup_fail_closed(self):
        plan = self.plan()
        for count in (0, 3, 5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "coverage mismatch"):
                plan.validate_group_coverage(count)
        for group_id in (-1, 4, True):
            with self.subTest(group_id=group_id), self.assertRaises(ValueError):
                plan.pool_for_group(group_id)

    def test_independent_state_capacities(self):
        plan = RequestStatePoolPlan(CachePoolPlan(self.history, 2), (self.swa, replace(self.compressor, capacity=7)))
        self.assertEqual(plan.budget.request_state_bytes, 272)
        self.assertEqual(plan.size_bytes, 400)

    def test_large_integer_budget(self):
        available = 2**63 + 123
        plan = self.plan(available)
        self.assertLessEqual(plan.size_bytes, available)
        self.assertLess(available - plan.size_bytes, 64)

    def test_invalid_budget_values(self):
        for args in ((0, 0, 0), (1, -1, 0), (1, 0, -1), (1.5, 0, 0), (True, 0, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                RequestStateMemoryBudget(*args)
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.plan(value)
                with self.assertRaises(ValueError):
                    self.plan(num_history_blocks=value)
                with self.assertRaises(ValueError):
                    self.plan().budget.bytes_for_history_blocks(value)
        with self.assertRaises(ValueError):
            plan_request_state_pools(self.history, self.states, 1000, min_history_blocks=0)

    def test_zero_capacity_is_not_a_successful_plan(self):
        budget = RequestStateMemoryBudget(64, 240, 128)
        self.assertEqual(budget.history_capacity(0), 0)
        self.assertEqual(budget.history_capacity(368), 0)
        self.assertEqual(budget.bytes_for_history_blocks(0), 368)
        with self.assertRaises(ValueError):
            budget.resolve_history_capacity(368, min_history_blocks=1)


if __name__ == "__main__":
    unittest.main()
