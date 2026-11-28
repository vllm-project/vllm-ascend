"""Exercise the model-owned Smoke interface used by Lite Actions."""

import json
import subprocess
import sys
import unittest


class SmokeContractTest(unittest.TestCase):
    def test_profile_exposes_independent_resource_requests(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.integration_tests.tools.lite_actions.entrypoint",
                "inspect",
                "--suite",
                "dispatch_smoke",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cases = json.loads(result.stdout)
        self.assertEqual(
            [(c["kind"], c["nnodes"], c["ngpu"]) for c in cases],
            [("worker_probe", 0, 0), ("npu_smoke", 1, 8), ("npu_smoke", 2, 8), ("npu_smoke", 1, 8)],
        )
        self.assertEqual(cases[1]["test_id"], cases[3]["test_id"])

    def test_unknown_test_does_not_resolve(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.integration_tests.tools.lite_actions.entrypoint",
                "inspect",
                "--test-id",
                "not_a_case",
            ],
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
