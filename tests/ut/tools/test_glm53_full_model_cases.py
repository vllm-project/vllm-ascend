# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for full-model GLM PD allocation, role recipes and Weekly dispatch."""

import base64
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from tests.e2e.common.multi_node.external_dp.external_dp_config import ExternalDPConfigLoader, RankResolver
from tests.e2e.common.multi_node.external_dp.runtime import build_all_server_commands

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "tests/e2e/cases/models/configs/GLM/full_model"


class FullModelConfigTests(unittest.TestCase):
    def test_resolved_commands_fit_allocation_and_agree_on_transfer_topology(self):
        paths = sorted(CONFIG_DIR.glob("*.yaml"))
        self.assertEqual(len(paths), 12)
        for path in paths:
            with self.subTest(case=path.stem):
                config = ExternalDPConfigLoader.from_yaml(
                    str(path), cluster_ips=[f"192.0.2.{index + 1}" for index in range(4)]
                )
                ranks = RankResolver(config).resolve()
                commands = build_all_server_commands(config, ranks)
                devices_by_node = {index: set() for index in range(config.num_nodes)}
                decode_ranks = []
                transfer_topologies = []
                for rank, command in zip(ranks, commands, strict=True):
                    devices = set(map(int, rank.visible_devices.split(",")))
                    self.assertFalse(devices & devices_by_node[rank.node_index])
                    self.assertLess(max(devices), config.npu_per_node)
                    devices_by_node[rank.node_index].update(devices)
                    self.assertEqual(command.env["VLLM_USE_V2_MODEL_RUNNER"], "1")
                    self.assertNotIn("${", " ".join(command.cmd))
                    for forbidden in ("--hf-overrides", "--load-format", "--dsa-context-parallel-size"):
                        self.assertNotIn(forbidden, command.cmd)
                    additional = json.loads(command.cmd[command.cmd.index("--additional-config") + 1])
                    self.assertFalse(additional.get("enable_dsa_cp", False))
                    connector = json.loads(command.cmd[command.cmd.index("--kv-transfer-config") + 1])
                    self.assertEqual(connector["kv_connector"], "MooncakeConnectorV2")
                    transfer_topologies.append(connector["kv_connector_extra_config"])
                    if rank.role == "prefiller":
                        self.assertEqual(connector["kv_role"], "kv_producer")
                        self.assertGreater(rank.cp_size, 1)
                        self.assertEqual(command.env["VLLM_PP_LAYER_PARTITION"], "42,36")
                        self.assertEqual("--headless" in command.cmd, rank.node_index == 1)
                        caching = "--enable-prefix-caching" in command.cmd
                        self.assertEqual(caching, path.stem.endswith("DECODE"))
                    else:
                        decode_ranks.append(rank.dp_rank)
                        self.assertEqual(connector["kv_role"], "kv_consumer")
                        self.assertNotIn("--prefill-context-parallel-size", command.cmd)
                        self.assertEqual(connector["kv_connector_extra_config"]["decode"]["tp_size"], rank.tp_size)
                self.assertEqual(sorted(decode_ranks), list(range(16)))
                self.assertTrue(all(topology == transfer_topologies[0] for topology in transfer_topologies))
                self.assertTrue(all(len(devices) == config.npu_per_node for devices in devices_by_node.values()))

    def test_role_recipes_and_workloads_remain_isolated(self):
        for path in sorted(CONFIG_DIR.glob("*.yaml")):
            with self.subTest(case=path.stem):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                a5 = "-A5-" in path.stem
                prefill_case = path.stem.endswith("PREFILL")
                for template in config["templates"]:
                    args = template["server_cmd_template"]
                    additional = json.loads(args[args.index("--additional-config") + 1])
                    max_len = int(args[args.index("--max-model-len") + 1])
                    self.assertEqual(max_len, 135000 if a5 else 200000)
                    self.assertNotIn("--enable-batch-sharded-sampling", args)
                    if template["node_index"] < 2:
                        self.assertEqual(additional["enable_fused_mc2"], 2)
                        self.assertTrue(additional["enable_flashcomm1"])
                        self.assertIn("--enforce-eager", args)
                    else:
                        self.assertIn("--no-enable-prefix-caching", args)
                        self.assertTrue(additional["recompute_scheduler_enable"])
                        if prefill_case:
                            self.assertEqual(additional["enable_fused_mc2"], 1)
                            self.assertNotIn("multistream_overlap_shared_expert", additional)
                        else:
                            self.assertNotIn("enable_fused_mc2", additional)
                            self.assertTrue(additional["multistream_overlap_shared_expert"])
                            compilation = additional["ascend_compilation_config"]
                            self.assertTrue(compilation["enable_npugraph_ex"])
                            self.assertEqual(compilation["enable_static_kernel"], not a5)
                benchmarks = config["benchmarks"]
                if path.stem.endswith("ACCURACY"):
                    self.assertEqual(list(benchmarks), ["acc_gpqa"])
                    continue
                self.assertEqual(len(benchmarks), 8)
                for name, bench in benchmarks.items():
                    self.assertEqual(bench["max_out_len"], 1 if prefill_case else 1024)
                    self.assertEqual(bench["request_rate"], 0)
                    if name.endswith("_warmup"):
                        measured = benchmarks[name.removesuffix("_warmup")]
                        for key in ("dataset_path", "num_prompts", "batch_size", "max_out_len"):
                            self.assertEqual(bench[key], measured[key])
                    if prefill_case:
                        self.assertNotIn("tpot_threshold", bench)

    def test_matrix_entries_resolve_to_correct_hardware(self):
        resolver = ROOT / ".github/workflows/scripts/resolve_nightly_tests.py"
        for frequency, expected_counts in (("weekly", {"a3": 4, "a5": 8}),):
            path = ROOT / f".github/workflows/configs/{frequency}_config.yaml"
            matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
            for soc, expected_count in expected_counts.items():
                entries = [
                    entry
                    for entry in matrix[soc]["multi_node"]["test_config"]
                    if entry.get("config_base_path", "").endswith("/GLM/full_model")
                ]
                self.assertEqual(len(entries), expected_count)
                self.assertEqual(len({entry["name"] for entry in entries}), expected_count)
                for entry in entries:
                    with self.subTest(frequency=frequency, case=entry["name"]):
                        config_path = ROOT / entry["config_base_path"] / entry["config_file_path"]
                        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                        self.assertEqual(entry["size"], config["num_nodes"])
                        self.assertEqual(config["npu_per_node"], 8 if soc == "a5" else 16)
                        with tempfile.TemporaryDirectory() as temp:
                            output = Path(temp) / "output"
                            env = {
                                **os.environ,
                                "NIGHTLY_MATRIX": "",
                                "WEEKLY_MATRIX": base64.b64encode(path.read_bytes()).decode(),
                                "TEST_CASES": entry["name"],
                                "IS_A3_560T": "false",
                                "GITHUB_OUTPUT": str(output),
                            }
                            subprocess.run([sys.executable, str(resolver)], env=env, check=True, capture_output=True)
                            flags = dict(line.split("=", 1) for line in output.read_text().splitlines())
                            self.assertEqual(
                                [key for key, value in flags.items() if value == "true"], [f"dispatch_{soc}"]
                            )

        nightly = (ROOT / ".github/workflows/configs/nightly_config.yaml").read_text(encoding="utf-8")
        self.assertNotIn("/GLM/full_model", nightly)

    def test_weekly_a5_workflow_and_dispatch_are_connected(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/schedule_weekly_test_a5.yaml").read_text())
        jobs = workflow["jobs"]
        self.assertEqual(jobs["multi-node-tests"]["with"]["soc_version"], "a5")
        self.assertEqual(jobs["multi-node-tests"]["with"]["test_frequency"], "weekly")
        self.assertIn("KUBECONFIG_A5_B64", jobs["multi-node-tests"]["secrets"]["KUBECONFIG_B64"])
        dispatch = yaml.safe_load((ROOT / ".github/workflows/pr_nightly_command.yml").read_text())
        dispatch_job = dispatch["jobs"]["dispatch-weekly-a5"]
        self.assertIn("command == 'weekly'", dispatch_job["if"])
        self.assertIn("schedule_weekly_test_a5.yaml", str(dispatch_job["steps"]))
        self.assertIn("dispatch-weekly-a5", dispatch["jobs"]["post-links"]["needs"])


if __name__ == "__main__":
    unittest.main()
