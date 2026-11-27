#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from subprocess import CompletedProcess, TimeoutExpired
from types import SimpleNamespace
from unittest.mock import call, patch

from tests.ut.base import TestBase
from vllm_ascend.worker.worker import _NPU_SMI_TIMEOUT_SECONDS, NPUWorker, resolve_card_id_for_health

WORKER_MODULE = "vllm_ascend.worker.worker"
NPU_SMI_MAP = """\
NPU ID  Chip ID  Chip Logic ID  Chip Phy-ID  Chip Name
1  0  0  2  Ascend910
1  1  1  3  Ascend910
1  2  -  -  Mcu
4  0  2  8  Ascend910
4  1  3  9  Ascend910
0  0  4  0  Ascend910
"""


class TestNPUWorkerCheckHealth(TestBase):
    def setUp(self):
        self.worker = NPUWorker.__new__(NPUWorker)
        self.worker.local_rank = 0
        self.worker.device = SimpleNamespace(index=0)
        command_module = self.mock_worker_attribute("subprocess")
        command_module.TimeoutExpired = TimeoutExpired
        self.command = command_module.run
        self.current_platform = self.mock_worker_attribute("current_platform")
        self.logger = self.mock_worker_attribute("logger")
        self.current_platform.visible_device_id_to_physical_device_id.return_value = 2
        self.mapping_result = CompletedProcess(["npu-smi", "info", "-m"], 0, NPU_SMI_MAP, "")
        self.healthy_result = CompletedProcess(["npu-smi"], 0, "Health: OK\n", "")

    def mock_worker_attribute(self, name):
        patcher = patch(f"{WORKER_MODULE}.{name}")
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_queries_card_owning_bound_device_including_multi_chip_and_remapping(self):
        for local_rank, visible_id, phy_id, card_id in ((3, 0, 2, 1), (1, 1, 3, 1), (0, 5, 8, 4), (0, 0, 0, 0)):
            with self.subTest(local_rank=local_rank, visible_id=visible_id, phy_id=phy_id):
                self.command.reset_mock()
                self.current_platform.reset_mock()
                self.worker.local_rank = local_rank
                self.worker.device.index = visible_id
                self.current_platform.visible_device_id_to_physical_device_id.return_value = phy_id
                self.command.side_effect = [self.mapping_result, self.healthy_result]
                self.worker.check_health()
                self.current_platform.visible_device_id_to_physical_device_id.assert_called_once_with(visible_id)
                self.command.assert_has_calls(
                    [
                        call(
                            ["npu-smi", "info", "-m"],
                            capture_output=True,
                            text=True,
                            timeout=_NPU_SMI_TIMEOUT_SECONDS,
                        ),
                        call(
                            ["npu-smi", "info", "-i", str(card_id), "-t", "health"],
                            capture_output=True,
                            text=True,
                            timeout=_NPU_SMI_TIMEOUT_SECONDS,
                        ),
                    ]
                )
                self.assertEqual(self.command.call_count, 2)
                self.assertEqual(
                    self.logger.debug.call_args.args,
                    ("check_health success for rank %s, NPU card %s.", local_rank, card_id),
                )
        self.current_platform.logical_device_id_to_visible_device_id.assert_not_called()
        self.current_platform.device_id_to_physical_device_id.assert_not_called()
        self.logger.warning.assert_not_called()
        self.logger.error.assert_not_called()

    def test_unavailable_mapping_warns_without_querying_a_guessed_card(self):
        for output in (
            "",
            "  \n",
            "NPU ID  Chip ID  Chip Logic ID\n2 0 4",
            "Chip ID  Chip Phy-ID\n0 2",
            "NPU ID  Chip Phy-ID\n",
            "NPU ID  Chip Phy-ID\n2 8",
            "NPU ID  Chip Phy-ID\n1",
            "NPU ID  Chip Phy-ID\n- 2\nMcu -",
        ):
            with self.subTest(output=output):
                self.command.reset_mock()
                self.logger.reset_mock()
                self.command.side_effect = [CompletedProcess(["npu-smi"], 0, output, "")]
                self.worker.check_health()
                self.command.assert_called_once()
                self.assertEqual(self.command.call_args.args[0], ["npu-smi", "info", "-m"])
                self.logger.warning.assert_called_once()
                self.assertEqual(self.logger.warning.call_args.args[1], self.worker.local_rank)
                self.assertTrue(self.logger.warning.call_args.args[2])
                self.logger.error.assert_not_called()
        self.current_platform.device_id_to_physical_device_id.assert_not_called()

    def test_mapping_command_failures_warn_and_skip_health_query(self):
        for result, expected in (
            (CompletedProcess(["npu-smi"], 215, "Invalid card id.", ""), "return_code=215"),
            (CompletedProcess(["npu-smi"], 1, "", "driver unavailable"), "driver unavailable"),
            (TimeoutExpired(["npu-smi", "info", "-m"], _NPU_SMI_TIMEOUT_SECONDS), "timed out"),
            (FileNotFoundError("npu-smi not found"), "npu-smi not found"),
        ):
            with self.subTest(result=result):
                self.command.reset_mock()
                self.logger.reset_mock()
                self.command.side_effect = [result]
                self.worker.check_health()
                self.command.assert_called_once()
                self.logger.warning.assert_called_once()
                self.assertIn(expected, self.logger.warning.call_args.args[2])
                if isinstance(result, CompletedProcess) and result.returncode == 215:
                    self.assertIn(result.stdout, self.logger.warning.call_args.args[2])
                self.logger.error.assert_not_called()

    def test_runtime_resolution_failure_warns_without_running_any_command(self):
        self.current_platform.visible_device_id_to_physical_device_id.side_effect = RuntimeError("CANN lookup failed")
        self.worker.check_health()
        self.command.assert_not_called()
        self.logger.warning.assert_called_once()
        self.assertIn("CANN lookup failed", self.logger.warning.call_args.args[2])

    def test_reordered_columns_skip_malformed_and_non_numeric_cards(self):
        output = "Chip Phy-ID   Chip Name    NPU ID     Chip ID\n2\n2 Ascend - 0\n- Mcu 1 2\n2 Ascend 4 0"
        self.command.return_value = CompletedProcess(["npu-smi"], 0, output, "")
        self.assertEqual(resolve_card_id_for_health(0), (4, None))

    def test_health_command_failures_keep_existing_non_raising_behavior(self):
        for result, log_level, expected in (
            (CompletedProcess(["npu-smi"], 215, "Invalid card id.", ""), "warning", "fail"),
            (CompletedProcess(["npu-smi"], 1, "", "health query failed"), "warning", "fail"),
            (TimeoutExpired(["npu-smi"], _NPU_SMI_TIMEOUT_SECONDS), "warning", "timeout"),
            (FileNotFoundError("npu-smi not found"), "warning", "tool not found"),
            (RuntimeError("unexpected error"), "error", "fail"),
        ):
            with self.subTest(result=result):
                self.logger.reset_mock()
                self.command.side_effect = [self.mapping_result, result]
                self.worker.check_health()
                log = getattr(self.logger, log_level)
                log.assert_called_once()
                self.assertIn(expected, log.call_args.args[0])
                self.assertIn(1, log.call_args.args[1:])
                self.assertIn(0, log.call_args.args[1:])
                if isinstance(result, CompletedProcess):
                    self.assertIn(result.returncode, log.call_args.args)
                    self.assertEqual(log.call_args.args[-1], result.stderr)

    def test_unhealthy_status_is_logged_without_changing_rpc_behavior(self):
        self.command.side_effect = [self.mapping_result, CompletedProcess(["npu-smi"], 0, "Health: ALARM", "")]
        self.worker.check_health()
        self.logger.error.assert_called_once()
        self.assertIn("not OK", str(self.logger.error.call_args.args[-1]))
        self.assertEqual(self.logger.debug.call_count, 1)

    def test_missing_health_status_preserves_existing_parser_behavior(self):
        self.command.side_effect = [self.mapping_result, CompletedProcess(["npu-smi"], 0, "NPU ID: 1", "")]
        self.worker.check_health()
        self.logger.error.assert_not_called()
        self.logger.warning.assert_not_called()
        self.assertEqual(self.logger.debug.call_args.args, ("check_health success for rank %s, NPU card %s.", 0, 1))
