# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import copy
import unittest
from types import SimpleNamespace

from vllm_ascend.distributed.fasteep.policy import replica_donors, validate_config, validate_target


class TestPlacement(unittest.TestCase):
    def setUp(self):
        self.maps = [[[0, 1, -1, -1], [-1, -1, 0, 1], [0, 1, -1, -1]]]

    def test_replica_and_full_size(self):
        self.assertEqual(replica_donors(self.maps * 3, 2), {2: 0})
        self.assertEqual(replica_donors(self.maps, 3), {})

    def test_non_prefix_replica(self):
        maps = [[[-1, 0, -1, 1], [0, -1, 1, -1], [0, -1, 1, -1]]]
        self.assertEqual(replica_donors(maps, 2), {2: 1})

    def test_reject_missing_logical_expert(self):
        maps = [[[0, 1, -1, -1], [0, 1, -1, -1], [-1, -1, 0, 1]]]
        with self.assertRaisesRegex(ValueError, "cover"):
            replica_donors(maps, 2)

    def test_one_donor_must_match_every_layer(self):
        layer = copy.deepcopy(self.maps[0])
        layer[2] = layer[1]
        with self.assertRaisesRegex(ValueError, "complete replica"):
            replica_donors(self.maps + [layer], 2)

    def test_equal_capacity_is_not_replica(self):
        maps = copy.deepcopy(self.maps)
        maps[0][2] = [1, 0, -1, -1]
        with self.assertRaisesRegex(ValueError, "complete replica"):
            replica_donors(maps, 2)

    def test_bad_slots_shapes_and_empty(self):
        for maps in ([], [[]], [[[0], [0], [0, 1]]], [[[0, 0], [0, 1], [0, 1]]], [[[0, -2]] * 3]):
            with self.subTest(maps=maps), self.assertRaises(ValueError):
                replica_donors(maps, 2)

    def test_target_boundaries(self):
        for target in (True, False, 0, 1, 4, 2.0, "2", None):
            with self.subTest(target=target), self.assertRaises(ValueError):
                validate_target(target, 3)


class TestConfig(unittest.TestCase):
    def config(self):
        parallel = SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_size=3,
            data_parallel_size_local=3,
            enable_expert_parallel=True,
            enable_elastic_ep=False,
            enable_eplb=False,
        )
        return SimpleNamespace(
            parallel_config=parallel,
            use_v2_model_runner=False,
            model_config=SimpleNamespace(enforce_eager=True, enable_sleep_mode=True, dtype="torch.bfloat16"),
            quant_config=None,
            cache_config=SimpleNamespace(enable_prefix_caching=False),
        )

    def test_valid(self):
        validate_config(self.config())

    def test_incompatible_configurations(self):
        for field, value in (
            ("tensor_parallel_size", 2),
            ("data_parallel_size_local", 2),
            ("enable_elastic_ep", True),
            ("enable_eplb", True),
            ("enable_expert_parallel", False),
            ("data_parallel_external_lb", True),
            ("data_parallel_hybrid_lb", True),
        ):
            config = self.config()
            setattr(config.parallel_config, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(config)
        for field in ("lora_config", "kv_transfer_config", "speculative_config", "engram_config"):
            config = self.config()
            setattr(config, field, object())
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(config)
