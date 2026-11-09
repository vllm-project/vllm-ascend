# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise actual external-DP allocation and cache replay sequencing on CPU."""

import unittest
from pathlib import Path

import yaml

from tests.e2e.common.multi_node.external_dp.external_dp_config import ExternalDPConfigLoader, RankResolver
from tests.e2e.common.multi_node.external_dp.runtime import build_all_server_commands

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "tests/e2e/cases/models/configs/GLM"
NAMES = ("GLM-5.3-Flash-W8A8-A3-PD-ReplaySSM", "GLM-5.3-Flash-W8A8-A3-PD-Decode-Replay")


class FlashWeeklyTests(unittest.TestCase):
    def test_rendered_allocation_consumes_each_device_once(self):
        for name in NAMES:
            with self.subTest(case=name):
                config = ExternalDPConfigLoader.from_yaml(
                    str(CONFIG_DIR / (name + ".yaml")), cluster_ips=["192.0.2.1", "192.0.2.2"]
                )
                ranks = RankResolver(config).resolve()
                commands = build_all_server_commands(config, ranks)
                devices = {0: set(), 1: set()}
                for rank, command in zip(ranks, commands, strict=True):
                    allocated = set(map(int, rank.visible_devices.split(",")))
                    self.assertFalse(devices[rank.node_index] & allocated)
                    devices[rank.node_index].update(allocated)
                    self.assertNotIn("${", " ".join(command.cmd))
                    self.assertEqual(command.env["HCCL_IF_IP"], f"192.0.2.{rank.node_index + 1}")
                    self.assertEqual(command.env["VLLM_USE_V2_MODEL_RUNNER"], "1")
                self.assertTrue(all(value == set(range(config.npu_per_node)) for value in devices.values()))
                self.assertEqual(len([rank for rank in ranks if rank.role == "decoder"]), 8)

    def test_replay_uses_identical_requests_without_reset_between_runs(self):
        for name in NAMES:
            raw = yaml.safe_load((CONFIG_DIR / (name + ".yaml")).read_text())
            benches = raw["benchmarks"]
            keys = list(benches)
            cached = not name.endswith("ReplaySSM")
            for template in raw["templates"]:
                args = template["server_cmd_template"]
                self.assertEqual("--use-replayssm" in args, not cached)
                self.assertEqual("--enable-prefix-caching" in args, cached and template["node_index"] == 0)
            if not cached:
                self.assertFalse(any(key.startswith("decode_") for key in keys))
                continue
            for profile in ("16k", "128k"):
                measured = f"decode_{profile}"
                index = keys.index(measured)
                self.assertEqual(keys[index - 1], measured + "_warmup")
                self.assertEqual(keys[index - 2], "reset_before_" + measured)
                self.assertEqual(benches[measured], benches[measured + "_warmup"])
                commands = benches[keys[index - 2]]["cmds"]
                self.assertTrue(all("--fail" in command and "/reset_prefix_cache" in command for command in commands))
                expected_ports = [8000] if "A5" in name else [9081, 9082]
                self.assertEqual(len(commands), len(expected_ports))
                for port in expected_ports:
                    self.assertTrue(any(f":{port}/" in command for command in commands))


if __name__ == "__main__":
    unittest.main()
