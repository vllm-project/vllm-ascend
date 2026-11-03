# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU tests for the numerical oracle using small synthetic fixtures."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.e2e.common.multi_node.glm53_logits.config import CASES, build
from tests.e2e.common.multi_node.glm53_logits.gate import (
    check_capture,
    load_golden,
    sha256,
    validate_features,
    validate_model,
)
from tests.e2e.common.multi_node.glm53_logits.metrics import compare
from tests.ut.glm53_logits_helpers import make_fixture


class TestLogitsGate(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        make_fixture(self.root)

    def test_configurations_match_their_own_golden(self):
        for case in CASES:
            manifest, _ = load_golden(case, root=self.root)
            for entry in manifest["cases"][case]:
                with self.subTest(case=case, sample=entry["sample"], role=entry["role"]):
                    result = check_capture(self.root / entry["file"], entry, manifest, root=self.root)
                    self.assertTrue(result["passed"])
                    self.assertEqual(result["max_absolute_error"], 0)

    def test_wrong_history_and_nonfinite_fail(self):
        reference = dict(
            logits=np.arange(40, dtype=float).reshape(2, 20), prompt_ids=np.array([1, 2]), token_ids=np.array([3, 4])
        )
        candidate = {k: v.copy() for k, v in reference.items()}
        candidate["token_ids"][1] = 5
        with self.assertRaisesRegex(ValueError, "Unaligned"):
            compare(reference, candidate)
        candidate["token_ids"] = reference["token_ids"]
        candidate["logits"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            compare(reference, candidate)

    def test_same_forced_tokens_do_not_hide_corrupt_scores(self):
        manifest, _ = load_golden("disable_sfac8", root=self.root)
        entry = manifest["cases"]["disable_sfac8"][0]
        with np.load(self.root / entry["file"]) as original:
            arrays = {key: original[key].copy() for key in original.files}
        arrays["logits"] *= -1
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "corrupt.npz"
            np.savez(candidate, **arrays)
            self.assertFalse(check_capture(candidate, entry, manifest, root=self.root)["passed"])

    def test_missing_and_corrupt_baseline_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_golden("disable_pp", root)
            (root / "manifest.json").write_bytes((self.root / "manifest.json").read_bytes())
            (root / "corpus.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_golden("disable_pp", root)

    def test_provisioned_arrays_use_separate_root(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory)
            for name in ("manifest.json", "corpus.json"):
                (metadata / name).write_bytes((self.root / name).read_bytes())
            manifest, _ = load_golden("disable_pp", metadata, artifact_root=self.root)
            self.assertEqual(len(manifest["cases"]["disable_pp"]), 8)
            with self.assertRaises(FileNotFoundError):
                load_golden("disable_pp", metadata, artifact_root=metadata)

    def test_pp_off_preserves_pcp_and_removes_partition(self):
        p = build("disable_pp", "P", "/model")
        self.assertEqual(p["prefill_context_parallel_size"], 2)
        self.assertEqual(p["pipeline_parallel_size"], 1)
        self.assertNotIn("pp_layer_partition", p["kv_transfer_config"]["kv_connector_extra_config"]["prefill"])
        for case in CASES:
            for role in ("P", "D"):
                self.assertNotIn("enable_dsa_cp", json.dumps(build(case, role, "/model")))

    def test_empty_or_incomplete_worker_evidence_fails(self):
        with self.assertRaisesRegex(ValueError, "worker probes"):
            validate_features({"features": []}, "disable_pp", "P")

    def test_model_identity_and_incomplete_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            config = model / "config.json"
            config.write_text("{}")
            manifest = {"model_hashes": [{"file": "config.json", "sha256": sha256(config)}]}
            validate_model(model, manifest)
            config.write_text('{"num_hidden_layers": 78}')
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                validate_model(model, manifest)
            (model / "INCOMPLETE").touch()
            with self.assertRaisesRegex(ValueError, "completed"):
                validate_model(model, manifest)


if __name__ == "__main__":
    unittest.main()
