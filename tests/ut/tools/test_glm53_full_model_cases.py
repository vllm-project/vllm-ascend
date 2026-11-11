# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for full-model GLM PD allocation, role recipes and Weekly dispatch."""

import base64
import json
import math
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
CONFIG_DIR = ROOT / "tests/SIT/cases/models/configs/GLM/full_model"


class FullModelConfigTests(unittest.TestCase):
    def test_resolved_commands_fit_allocation_and_agree_on_transfer_topology(self):
        paths = sorted(CONFIG_DIR.glob("*.yaml"))
        self.assertEqual(len(paths), 3)
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
                        self.assertEqual("--headless" in command.cmd, rank.node_rank == 1)
                        caching = "--enable-prefix-caching" in command.cmd
                        self.assertTrue(caching)
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
                for template in config["templates"]:
                    args = template["server_cmd_template"]
                    additional = json.loads(args[args.index("--additional-config") + 1])
                    max_len = int(args[args.index("--max-model-len") + 1])
                    self.assertEqual(max_len, 135000 if a5 else 131072)
                    self.assertNotIn("--enable-batch-sharded-sampling", args)
                    if template["node_index"] in config["routing"]["groups"]["prefiller"]:
                        self.assertEqual(additional["enable_fused_mc2"], 1)
                        self.assertTrue(additional["enable_flashcomm1"])
                        self.assertIn("--enforce-eager", args)
                    else:
                        self.assertIn("--no-enable-prefix-caching", args)
                        self.assertTrue(additional["scheduler_config"]["recompute_scheduler_enable"])
                        self.assertNotIn("enable_fused_mc2", additional)
                        self.assertTrue(additional["multistream_overlap_shared_expert"])
                        compilation = additional["ascend_compilation_config"]
                        self.assertTrue(compilation["enable_npugraph_ex"])
                        self.assertEqual(compilation["enable_static_kernel"], not a5)
                        self.assertEqual(compilation["enable_super_kernel"], not a5)
                benchmarks = {
                    name: bench for name, bench in config["benchmarks"].items() if bench["case_type"] == "performance"
                }
                self.assertEqual(len(benchmarks), 24 if a5 else 6)
                for name, bench in benchmarks.items():
                    prefill_case = name.startswith("prefill_")
                    self.assertEqual(bench["max_out_len"], 1 if prefill_case else 1024)
                    expected_rate = 1.8 if not a5 and name.startswith("pd_") else 0
                    self.assertEqual(bench["request_rate"], expected_rate)
                    if name.endswith("_warmup"):
                        measured = benchmarks[name.removesuffix("_warmup")]
                        for key in ("dataset_path", "batch_size", "max_out_len", "request_rate"):
                            self.assertEqual(bench[key], measured[key])
                        if a5 or name.startswith("decode_"):
                            self.assertEqual(bench["num_prompts"], measured["num_prompts"])
                    if prefill_case:
                        self.assertNotIn("tpot_threshold", bench)

    def test_loads_and_cache_boundaries_share_one_deployment(self):
        expected = {"prefill": (16, 28, 0, 1, 32), "pd": (16, 160, 1.8, 1024, 80), "decode": (16, 320, 0, 1024, 320)}
        path = CONFIG_DIR / "GLM-5.3-W8A8C8-A3-ALL-E2E.yaml"
        benches = yaml.safe_load(path.read_text(encoding="utf-8"))["benchmarks"]
        for mode, (concurrency, requests, rate, output, warmup) in expected.items():
            case = benches[f"{mode}_perf_16k"]
            self.assertEqual(
                (case["batch_size"], case["num_prompts"], case["request_rate"], case["max_out_len"]),
                (concurrency, requests, rate, output),
            )
            self.assertEqual(benches[f"{mode}_perf_16k_warmup"]["num_prompts"], warmup)
            self.assertEqual(case.get("tpot_threshold"), None if mode == "prefill" else 20)
        for path in sorted(CONFIG_DIR.glob("*.yaml")):
            config = ExternalDPConfigLoader.from_yaml(str(path), cluster_ips=[f"192.0.2.{i + 1}" for i in range(4)])
            master_p = next(
                rank for rank in RankResolver(config).resolve() if rank.role == "prefiller" and not rank.is_headless
            )
            self.assertEqual(master_p.role, "prefiller")
            self.assertFalse(master_p.is_headless)
            self.assertEqual(master_p.port, 7100)
            cases = config.benchmark_cases
            names = [case["benchmark_name"] for case in cases]
            for index, case in enumerate(cases):
                name = case["benchmark_name"]
                if name.startswith("reset_before_"):
                    self.assertEqual(case["case_type"], "curl_cmd")
                    self.assertIn("--fail", case["cmds"][0])
                    reset_host = "192.0.2.3" if "-A3-" in path.stem else "127.0.0.1"
                    self.assertIn(f"http://{reset_host}:7100/reset_prefix_cache", case["cmds"][0])
                elif name.startswith(("prefill_", "pd_")) and not name.endswith("_warmup"):
                    self.assertEqual(names[index - 1], "reset_before_" + name)
                elif name.startswith("decode_") and not name.endswith("_warmup"):
                    self.assertEqual(names[index - 1], name + "_warmup")
                    self.assertEqual(names[index - 2], "reset_before_" + name)
                    for key in ("dataset_path", "num_prompts", "max_out_len", "batch_size", "request_rate"):
                        self.assertEqual(cases[index - 1][key], case[key])

    def test_matrix_entries_resolve_to_correct_hardware(self):
        resolver = ROOT / ".github/workflows/scripts/resolve_nightly_tests.py"
        for frequency, expected_counts in (("weekly", {"a3": 1, "a5": 2}),):
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
                self.assertFalse(any(entry["name"].endswith("ACCURACY") for entry in entries))
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

    def test_gpqa_runs_once_per_family_after_e2e_performance(self):
        for family in ("W8A8C8-A3", "W8A8C8-A5", "W4A4C8-A5"):
            with self.subTest(family=family):
                paths = sorted(CONFIG_DIR.glob(f"GLM-5.3-{family}-*.yaml"))
                self.assertEqual(len(paths), 1)
                accuracy_cases = []
                for path in paths:
                    config = ExternalDPConfigLoader.from_yaml(
                        str(path), cluster_ips=[f"192.0.2.{index + 1}" for index in range(4)]
                    )
                    cases = config.benchmark_cases
                    accuracy = [case for case in cases if case["case_type"] == "accuracy"]
                    if path.stem.endswith("E2E"):
                        self.assertEqual(len(accuracy), 1)
                        self.assertEqual(cases[-1]["benchmark_name"], "acc_gpqa")
                        self.assertTrue(all(case["case_type"] in ("performance", "curl_cmd") for case in cases[:-1]))
                        self.assertEqual(cases[-1]["dataset_path"], "vllm-ascend/gpqa")
                        self.assertEqual(cases[-1]["max_out_len"], 65536)
                        self.assertEqual(cases[-1]["batch_size"], 16 if family.endswith("A3") else 32)
                        self.assertEqual(cases[-1]["baseline"], 86.92)
                        self.assertEqual(cases[-1]["verify_mode"], "ge")
                    else:
                        self.assertEqual(accuracy, [])
                    accuracy_cases.extend(accuracy)
                self.assertEqual(len(accuracy_cases), 1)

    def test_scheduler_and_graph_budgets_cover_peak_client_load(self):
        for path in sorted(CONFIG_DIR.glob("*.yaml")):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            peak_clients = max(
                case["batch_size"]
                for case in config["benchmarks"].values()
                if case["case_type"] in ("performance", "accuracy")
            )
            decode_nodes = [
                node for node in config["config"] if node["node_index"] in config["routing"]["groups"]["decoder"]
            ]
            global_dp = decode_nodes[0]["dp_size"]
            self.assertEqual(sum(node["dp_size_local"] for node in decode_nodes), global_dp)
            balanced_peak = math.ceil(peak_clients / global_dp)
            for template in config["templates"]:
                args = template["server_cmd_template"]
                seqs = int(args[args.index("--max-num-seqs") + 1])
                tokens = int(args[args.index("--max-num-batched-tokens") + 1])
                if template["node_index"] in config["routing"]["groups"]["prefiller"]:
                    # The P pipeline has DP1; PP and PCP do not divide its request load.
                    self.assertGreaterEqual(seqs, peak_clients)
                    self.assertGreaterEqual(tokens, seqs)
                    continue
                self.assertGreaterEqual(seqs, balanced_peak)
                self.assertLessEqual(seqs, 2 * balanced_peak)
                spec = json.loads(args[args.index("--speculative-config") + 1])
                query_length = spec["num_speculative_tokens"] + 1
                compilation = json.loads(args[args.index("--compilation-config") + 1])
                self.assertGreaterEqual(tokens, seqs * query_length)
                # Every possible uniform decode batch has an exact capture, no
                # oversized padding gap between common per-instance loads.
                self.assertEqual(
                    compilation["cudagraph_capture_sizes"],
                    [batch * query_length for batch in range(1, seqs + 1)],
                )
                self.assertEqual(compilation["max_cudagraph_capture_size"], seqs * query_length)

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
