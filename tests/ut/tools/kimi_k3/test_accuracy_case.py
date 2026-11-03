# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.kimi_k3.accuracy_case import service_arguments, validate_inputs, validate_results


class AccuracyCaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "train.jsonl").write_text("")
        self.model = self.root / "model"
        self.model.mkdir()
        self.config = {"text_config": {"num_experts": 256, "num_hidden_layers": 93, "num_experts_per_tok": 16}}
        self.write_config()
        rows = [{"question": f"Unique question {i}?", "answer": f"Explanation #### {i}"} for i in range(500)]
        (self.dataset / "test.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
        self.work = self.root / "work"
        self.result = self.work / "results/model/gsm8k.json"
        self.pred = self.work / "predictions/model/gsm8k.jsonl"
        self.result.parent.mkdir(parents=True)
        self.pred.parent.mkdir(parents=True)
        self.records = [
            {"id": i, "origin_prompt": row["question"], "prediction": f"The answer is {i}", "success": True}
            for i, row in enumerate(rows)
        ]
        self.write_results()

    def write_config(self):
        (self.model / "config.json").write_text(json.dumps(self.config))

    def write_results(self, accuracy=100):
        self.pred.write_text("\n".join(json.dumps(x) for x in self.records))
        self.result.write_text(json.dumps({"accuracy": accuracy, "details": {str(i): {} for i in range(500)}}))

    def test_input_identity_and_threshold(self):
        digest = hashlib.sha256((self.dataset / "test.jsonl").read_bytes()).hexdigest()
        with patch("tools.kimi_k3.accuracy_case.TEST_SHA256", digest):
            self.assertEqual(validate_inputs(self.model, self.dataset, "60", "5"), (60, 5))
            self.config["text_config"]["num_experts"] = 896
            self.write_config()
            with self.assertRaisesRegex(ValueError, "num_experts"):
                validate_inputs(self.model, self.dataset, 60, 5)
        for baseline, tolerance in [(float("nan"), 5), (60, -1), (60, float("inf"))]:
            with self.assertRaises(ValueError):
                validate_inputs(self.model, self.dataset, baseline, tolerance)

    def test_dataset_changed(self):
        with self.assertRaisesRegex(ValueError, "test.jsonl"):
            validate_inputs(self.model, self.dataset, 60, 5)

    def test_independent_score_includes_invalid(self):
        self.records[0]["prediction"] = ""
        self.write_results(99.8)
        result = validate_results(self.work, self.dataset)
        self.assertEqual(result["correct"], 499)
        self.assertEqual(result["invalid_final_answers"], 1)

    def test_duplicate_or_partial_records_rejected(self):
        self.records[-1] = self.records[0]
        self.write_results()
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_results(self.work, self.dataset)
        self.records.pop()
        self.write_results()
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_results(self.work, self.dataset)

    def test_prompt_and_score_mismatch_rejected(self):
        self.records[0]["origin_prompt"] = "different question"
        self.write_results()
        with self.assertRaisesRegex(ValueError, "original question"):
            validate_results(self.work, self.dataset)
        self.records[0]["origin_prompt"] = "Unique question 0?"
        self.write_results(50)
        with self.assertRaisesRegex(ValueError, "Independent accuracy"):
            validate_results(self.work, self.dataset)

    def test_real_service_flags(self):
        script = Path("tools/kimi_k3/serve_ep256.sh").read_text()
        args = service_arguments(script)
        self.assertNotIn("--port", args)
        self.assertNotIn("exec", args)
        self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "8")
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "100")
        self.assertIn("--enable-expert-parallel", args)
        self.assertIn("--enforce-eager", args)
        self.assertFalse(any("$" in value for value in args))
