# SPDX-License-Identifier: Apache-2.0
import unittest

import numpy as np

from vllm_ascend.eplb.core.policy.layer_placement import _residual_placement_is_feasible
from vllm_ascend.eplb.core.policy.policy_layer import LayerPlanner


class TestLayerPlanner(unittest.TestCase):
    def test_scores_replay_source_rank_routing(self):
        table = np.array([[[0, 1], [0, 2]]])
        heat = np.array([[8.0, 2.0, 6.0]])
        # The replicated expert contributes four tokens per rank: loads 6, 10.
        np.testing.assert_allclose(LayerPlanner._scores(table, heat, 3), [0.8])

    def test_zero_heat_keeps_current_placement(self):
        table = np.array([[[0, 1], [0, 2]]])
        planner = LayerPlanner(1)
        for _ in range(3):
            decision = planner.plan(table, np.zeros_like(table))
            self.assertFalse(decision.should_apply)
            np.testing.assert_array_equal(decision.placement, table)

    def test_residual_placement_respects_existing_copy(self):
        self.assertTrue(_residual_placement_is_feasible(np.array([1, 1]), np.array([1, 1]), [{0}, set()]))
        self.assertFalse(_residual_placement_is_feasible(np.array([1, 1]), np.array([2, 0]), [{0}, set()]))


if __name__ == "__main__":
    unittest.main()
