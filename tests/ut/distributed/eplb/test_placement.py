# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from vllm_ascend.distributed.eplb.placement import (
    EplbPlacement,
    build_placement_metadata,
    load_placement,
    plan_placement_sources,
)


def write_placement(path, placement):
    Path(path).write_text(json.dumps(placement.to_dict()), encoding="utf-8")


def make_metadata():
    return {
        "model": "test-model",
        "revision": "test-revision",
        "architecture": "TestMoE",
        "hf_config_sha256": "test-config-hash",
        "quantization": "ascend",
        "dtype": "torch.bfloat16",
        "ep_size": 2,
        "tp_size": 2,
        "dp_size": 1,
        "pp_size": 1,
        "num_logical_experts": 4,
        "num_redundant_experts": 2,
        "layer_names": ["layers.0.moe"],
        "placement_scope": "mixed",
    }


class TestPlacement(unittest.TestCase):
    def setUp(self):
        self.placement = EplbPlacement(make_metadata(), ((0, 1, 2, 0, 1, 3),), {"workload": "unit-test"})

    def test_round_trip_and_relative_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment.json"
            write_placement(path, self.placement)
            loaded = load_placement(path)
            self.assertEqual(loaded, self.placement)
            self.assertEqual(loaded.digest(), self.placement.digest())

    def test_draft_placement_round_trip(self):
        draft_metadata = make_metadata()
        draft_metadata.update(
            model="draft-model",
            num_logical_experts=2,
            num_redundant_experts=0,
            layer_names=["layers.0.draft.moe"],
        )
        draft = EplbPlacement(draft_metadata, ((0, 1),), {"workload": "draft"})
        placement = EplbPlacement(
            self.placement.metadata,
            self.placement.physical_to_logical,
            self.placement.provenance,
            draft=draft,
        )

        data = placement.to_dict()
        self.assertEqual(data["schema_version"], 3)
        loaded = EplbPlacement.from_dict(data)
        self.assertEqual(loaded, placement)
        self.assertEqual(loaded.digest(), placement.digest())
        self.assertIsNotNone(loaded.draft)
        self.assertEqual(loaded.draft.metadata["model"], "draft-model")

        target_only = self.placement.to_dict()
        self.assertEqual(target_only["schema_version"], 3)
        self.assertNotIn("draft", target_only)
        self.assertEqual(EplbPlacement.from_dict(target_only), self.placement)

        legacy_v2 = dict(target_only, schema_version=2)
        self.assertEqual(EplbPlacement.from_dict(legacy_v2), self.placement)

    def test_draft_metadata_mismatch_is_reported(self):
        draft_metadata = make_metadata()
        draft_metadata.update(model="draft-model", layer_names=["layers.0.draft.moe"])
        placement = EplbPlacement(
            self.placement.metadata,
            self.placement.physical_to_logical,
            self.placement.provenance,
            draft=EplbPlacement(draft_metadata, ((0, 1),), {}),
        )
        expected = dict(draft_metadata)
        expected["model"] = "other-draft"
        with self.assertRaisesRegex(ValueError, "model mismatch"):
            placement.validate(expected_metadata=self.placement.metadata, expected_draft_metadata=expected)

    def test_missing_draft_is_reported_when_expected(self):
        with self.assertRaisesRegex(ValueError, "missing the expected draft"):
            self.placement.validate(expected_metadata=self.placement.metadata, expected_draft_metadata=make_metadata())

    def test_invalid_placements(self):
        cases = [
            ("missing logical expert", ((0, 1, 2, 0, 1, 2),), "cover every"),
            ("invalid ID", ((0, 1, 2, 0, 1, 4),), "invalid logical"),
            ("boolean ID", ((False, 1, 2, 0, 1, 3),), "invalid logical"),
            ("wrong slot count", ((0, 1, 2, 3),), "slot count"),
            ("missing layer", (), "layer count"),
        ]
        for label, mapping, error in cases:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, error):
                EplbPlacement(make_metadata(), mapping, {}).validate()

    def test_rejects_schema_and_unknown_fields(self):
        for version in (True, 0, 1, "3"):
            data = self.placement.to_dict()
            data["schema_version"] = version
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "schema_version"):
                EplbPlacement.from_dict(data)
        data = self.placement.to_dict()
        data["logical_to_physical"] = []
        with self.assertRaisesRegex(ValueError, "exactly"):
            EplbPlacement.from_dict(data)

    def test_metadata_mismatch(self):
        for name, value in (
            ("model", "other"),
            ("revision", "other"),
            ("quantization", None),
            ("hf_config_sha256", "other"),
            ("placement_scope", "prefill"),
            ("ep_size", 3),
            ("tp_size", 1),
            ("layer_names", ["layers.1.moe"]),
        ):
            expected = make_metadata()
            expected[name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"{name} mismatch"):
                self.placement.validate(expected)

    def test_rejects_invalid_parallel_metadata(self):
        for name, value in (("ep_size", True), ("ep_size", 4), ("pp_size", 2), ("layer_names", [""])):
            metadata = make_metadata()
            metadata[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                EplbPlacement(metadata, self.placement.physical_to_logical, {}).validate()

    def test_source_plan_handles_local_swaps_and_replication(self):
        old = (0, 1, 2, 3, 0, 1)
        target = (1, 0, 3, 2, 0, 1)
        ranks, slots = plan_placement_sources(old, target, 2)
        self.assertEqual((ranks[0][0], slots[0][0]), (0, 1))
        self.assertEqual((ranks[0][1], slots[0][1]), (0, 0))
        self.assertEqual((ranks[0][2], slots[0][2]), (1, 0))
        for rank in range(2):
            for slot in range(3):
                self.assertEqual(old[ranks[rank][slot] * 3 + slots[rank][slot]], target[rank * 3 + slot])

    def test_source_plan_rejects_missing_sources(self):
        with self.assertRaisesRegex(ValueError, "No source"):
            plan_placement_sources((0, 1), (0, 2), 2)

    def test_metadata_fingerprint_ignores_local_hf_path(self):
        layer = object()
        model = SimpleNamespace(
            named_modules=lambda: [("layers.0.moe", layer)],
            moe_layers=[layer],
            num_logical_experts=4,
            num_redundant_experts=2,
        )
        hf = {"_name_or_path": "/old", "transformers_version": "old", "hidden_size": 64}
        config = SimpleNamespace(
            model="test-model",
            revision="rev",
            architecture="TestMoE",
            quantization="ascend",
            dtype="bfloat16",
            hf_config=SimpleNamespace(to_dict=lambda: copy.deepcopy(hf)),
        )
        parallel = SimpleNamespace(tensor_parallel_size=2, data_parallel_size=1, pipeline_parallel_size=1)
        first = build_placement_metadata(model, config, parallel, 2, "mixed")
        hf.update(_name_or_path="/new", transformers_version="new")
        self.assertEqual(first, build_placement_metadata(model, config, parallel, 2, "mixed"))
        hf["hidden_size"] = 128
        self.assertNotEqual(
            first["hf_config_sha256"],
            build_placement_metadata(model, config, parallel, 2, "mixed")["hf_config_sha256"],
        )

    def test_invalid_json_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load_placement(path)
