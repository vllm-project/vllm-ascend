# SPDX-License-Identifier: Apache-2.0
"""Source-only integration tests: python test_run.py --engine-root PATH."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qa_markdown import change_evidence
from run import cache_key, git


class InterfaceTests(unittest.TestCase):
    engine: Path

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.up = self.root / "up"
        self.down = self.root / "down"
        for root in (self.up, self.down):
            root.mkdir()
            git(root, "init")
            git(root, "config", "user.name", "Interface Test")
            git(root, "config", "user.email", "interface@example.invalid")
            git(root, "config", "core.autocrlf", "false")
        self.write(self.up, "vllm/__init__.py", "")
        self.write(self.up, "vllm/api.py", "def run(value):\n    return value\n")
        self.old = self.commit(self.up)
        self.write(self.up, "vllm/api.py", "def run(value, required):\n    return value\n")
        self.new = self.commit(self.up)
        self.write(self.down, "vllm_ascend/__init__.py", "")
        self.write(self.down, "vllm_ascend/client.py", "from vllm.api import run\n\ndef invoke():\n    return run(1)\n")
        self.write(self.down, ".github/vllm-main-verified.commit", self.old + "\n")
        self.ascend = self.commit(self.down)
        self.output = self.root / "result"
        self.cache = self.root / "cache"

    def tearDown(self):
        # Windows Git object files are read-only; TemporaryDirectory handles them.
        self.temp.cleanup()

    @staticmethod
    def write(root, name, contents):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    @staticmethod
    def commit(root):
        git(root, "add", ".")
        git(root, "commit", "-qm", "fixture")
        return git(root, "rev-parse", "HEAD")

    def invoke(self, *args, success=True):
        result = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                "-B",
                str(Path(__file__).with_name("run.py")),
                *map(str, args),
                "--output",
                str(self.output),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)

    def prepare(self, success=True):
        self.invoke(
            "prepare",
            "--engine-root",
            self.engine,
            "--engine-sha",
            git(self.engine, "rev-parse", "HEAD"),
            "--vllm-root",
            self.up,
            "--ascend-root",
            self.down,
            "--old",
            self.old,
            "--new",
            self.new,
            "--ascend-sha",
            self.ascend,
            success=success,
        )

    def status(self):
        return json.loads((self.output / "run-status.json").read_text(encoding="utf-8"))

    def test_break_cache_corruption_and_force(self):
        self.prepare()
        self.invoke("scan", "--cache-dir", self.cache)
        self.assertEqual(self.status()["cache_status"], "miss")
        self.assertGreater(self.status()["summary"]["actionable_introduced_break"], 0)
        log = self.output / "scan.log"
        log.unlink()
        self.invoke("scan", "--cache-dir", self.cache)
        self.assertEqual(self.status()["cache_status"], "hit")
        self.assertFalse(log.exists(), "Cache reuse must not start the analyzer")
        (self.cache / "main2main-range-report.json").write_text("{}", encoding="utf-8")
        self.invoke("scan", "--cache-dir", self.cache)
        self.assertEqual(self.status()["cache_status"], "invalid")
        self.invoke("scan", "--cache-dir", self.cache, "--force-rescan")
        self.assertEqual(self.status()["cache_status"], "bypassed")
        self.assertEqual(self.status()["model_calls"], 0)
        qa = (self.output / "qa-review.md").read_text(encoding="utf-8")
        self.assertIn("prepared_not_executed", qa)
        self.assertIn("required", qa)
        self.assertIn(f"/blob/{self.old}/vllm/api.py", qa)
        self.assertIn(f"/blob/{self.new}/vllm/api.py", qa)
        self.assertIn("QA 结论", qa)
        self.assertFalse((self.output / "qa-input.json").exists())

    def test_fixed_snapshot_invalidates_cache(self):
        self.prepare()
        self.invoke("scan", "--cache-dir", self.cache)
        original_key = json.loads((self.output / "run-metadata.json").read_text())["cache_key"]
        self.write(
            self.down, "vllm_ascend/client.py", "from vllm.api import run\n\ndef invoke():\n    return run(1, 2)\n"
        )
        self.ascend = self.commit(self.down)
        self.prepare()
        self.invoke("scan", "--cache-dir", self.cache)
        self.assertEqual(self.status()["cache_status"], "invalid")
        self.assertEqual(self.status()["summary"]["actionable_introduced_break"], 0)
        self.assertNotEqual(original_key, json.loads((self.output / "run-metadata.json").read_text())["cache_key"])

    def test_dirty_and_invalid_sha_fail(self):
        self.write(self.down, "vllm_ascend/untracked.py", "x = 1\n")
        self.prepare(success=False)
        self.assertEqual(self.status()["scan"], "failed")
        (self.down / "vllm_ascend/untracked.py").unlink()
        self.old = "not-a-sha"
        self.prepare(success=False)

    def test_removal_evidence_distinguishes_typing_guard(self):
        self.write(self.up, "vllm/constants.py", "VALUE = 1\n")
        old = self.commit(self.up)
        self.write(
            self.up, "vllm/constants.py", "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    VALUE: int\n"
        )
        new = self.commit(self.up)
        evidence = change_evidence(self.up, old, new, {"file": "vllm/constants.py", "name": "VALUE"})
        self.assertIn("-VALUE = 1", evidence)
        self.assertIn("if TYPE_CHECKING:", evidence)
        self.assertIn(f"/blob/{new}/vllm/constants.py#L3", evidence)
        self.assertIn("匹配 1 行", evidence)

    def test_fingerprint_changes(self):
        self.prepare()
        inputs = json.loads((self.output / "run-metadata.json").read_text())["inputs"]
        for name in ("engine_sha", "runner_sha256", "vllm_old_sha", "vllm_new_sha", "vllm_ascend_sha", "profile"):
            with self.subTest(name=name):
                changed = {**inputs, name: "changed"}
                self.assertNotEqual(cache_key(inputs), cache_key(changed))

    def test_equal_range_skips_engine(self):
        self.write(self.down, ".github/vllm-main-verified.commit", self.new + "\n")
        self.ascend = self.commit(self.down)
        self.old = self.new
        self.prepare()
        self.invoke("scan", "--cache-dir", self.cache)
        self.assertEqual(self.status()["scan"], "skipped_no_changes")
        self.assertFalse((self.output / "scan.log").exists())
        self.assertTrue((self.output / "qa-review.md").is_file())

    def test_marker_mismatch_and_reverse_range(self):
        self.old = self.new
        self.prepare(success=False)
        self.write(self.down, ".github/vllm-main-verified.commit", self.new + "\n")
        self.ascend = self.commit(self.down)
        self.new = git(self.up, "rev-parse", "HEAD~1")
        git(self.up, "checkout", "--detach", self.new)
        self.prepare(success=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    InterfaceTests.engine = args.engine_root.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
