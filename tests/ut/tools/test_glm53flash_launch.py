# SPDX-License-Identifier: Apache-2.0
import json
import unittest

from tools.ci.glm53flash_launch import effective_settings, serve_command


class DeploymentTest(unittest.TestCase):
    def test_nightly_preserves_community(self):
        settings, changes, env = effective_settings("nightly_tp16")
        self.assertEqual(changes, {})
        self.assertEqual(settings["tensor_parallel_size"], 16)
        self.assertEqual(settings["speculative_config"]["num_speculative_tokens"], 3)
        self.assertTrue(settings["speculative_config"]["enforce_eager"])
        self.assertEqual(env["HCCL_BUFFSIZE"], "400")

    def test_reduction_is_explicit_and_isolated(self):
        settings, changes, _ = effective_settings("text_tp4")
        self.assertEqual(settings["tensor_parallel_size"], 4)
        self.assertNotIn("speculative_config", settings)
        self.assertNotIn("limit_mm_per_prompt", settings)
        self.assertEqual(changes["tensor_parallel_size"], {"community": 16, "effective": 4})
        settings["compilation_config"]["cudagraph_capture_sizes"].append(999)
        self.assertNotIn(999, effective_settings("text_tp4")[0]["compilation_config"]["cudagraph_capture_sizes"])

    def test_command_keeps_model_one_argument(self):
        settings, _, _ = effective_settings("nightly_tp16")
        command = serve_command("/models/path with spaces", settings, 18053)
        self.assertIn("/models/path with spaces", command)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertIn("--enable-expert-parallel", command)
        self.assertIn("--trust-remote-code", command)
        self.assertEqual(json.loads(command[command.index("--speculative-config") + 1]), settings["speculative_config"])


if __name__ == "__main__":
    unittest.main()
