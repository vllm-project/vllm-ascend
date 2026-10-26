# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.e2e.common.startup_diagnostics import StartupDiagnostics


class TestStartupDiagnostics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.diag = StartupDiagnostics(Path(self.tmp.name) / "evidence")

    def test_env_is_scoped_to_server(self):
        original = {"PYTHONPATH": "existing", "OMP_NUM_THREADS": "1"}
        env = self.diag.server_env(original)
        self.assertEqual(original["PYTHONPATH"], "existing")
        self.assertEqual(env["PYTHONPATH"], str(self.diag.root) + os.pathsep + "existing")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertTrue((self.diag.root / "sitecustomize.py").is_file())

    def test_throttled_samples_and_forced_final_snapshot(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("npu-smi not installed")):
            self.diag.sample(os.getpid())
            self.diag.sample(os.getpid())
            self.assertEqual(self.diag.sequence, 1)
            self.diag.stop(os.getpid(), "timeout-before-cleanup")
            self.assertEqual(self.diag.sequence, 2)
            self.diag.sample(os.getpid(), force=True)
            self.assertEqual(self.diag.sequence, 2)
        data = json.loads((self.diag.root / "sample-0002.json").read_text())
        self.assertEqual(data["reason"], "timeout-before-cleanup")
        self.assertIn("error", data["npu_smi"])
        self.assertTrue((self.diag.root / "STOP").exists())
        self.assertEqual(data["processes"][0]["pid"], os.getpid())

    def test_npu_command_timeout_does_not_fail_readiness(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("npu-smi", 10)) as run:
            self.diag.stop(os.getpid(), "ready")
        self.assertEqual(run.call_args.kwargs["timeout"], 10)
        self.assertTrue(self.diag.stopped)
        self.assertIn("error", json.loads((self.diag.root / "sample-0001.json").read_text())["npu_smi"])

    def test_sampling_error_still_stops_worker_timers(self):
        with patch.object(self.diag, "_sample", side_effect=OSError("cannot read proc")):
            self.diag.stop(os.getpid(), "ready")
        self.assertTrue((self.diag.root / "STOP").exists())
        self.assertTrue(self.diag.stopped)
