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
# This file is a part of the vllm-ascend project.
#

import subprocess
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, call, mock_open, patch

import vllm_ascend.cpu_binding as cpu_binding_module
from vllm_ascend.cpu_binding import CpuAlloc, DeviceInfo, bind_cpus, is_arm_cpu
from vllm_ascend.utils import AscendDeviceType


def make_cpu_alloc(rank_id=0):
    cpu_alloc = object.__new__(CpuAlloc)
    cpu_alloc.rank_id = rank_id
    cpu_alloc.device_info = SimpleNamespace(
        running_npu_list=[0],
        allowed_cpus=[],
        npu_affinity={},
        total_logic_npus=0,
    )
    cpu_alloc.cpu_node = {}
    cpu_alloc.numa_to_cpu_map = defaultdict(list)
    cpu_alloc.npu_cpu_pool = {}
    cpu_alloc.assign_main = {}
    cpu_alloc.assign_acl = {}
    cpu_alloc.assign_rel = {}
    return cpu_alloc


class TestDeviceInfo(unittest.TestCase):

    @patch('vllm_ascend.cpu_binding.subprocess.Popen')
    def test_execute_command(self, mock_popen):
        process = MagicMock()
        process.communicate.return_value = (b'command-output', b'')
        process.returncode = 7
        mock_popen.return_value.__enter__.return_value = process

        output, return_code = cpu_binding_module.execute_command(['dummy', 'cmd'])

        self.assertEqual(output, 'command-output')
        self.assertEqual(return_code, 7)
        mock_popen.assert_called_once_with(
            ['dummy', 'cmd'],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    @patch('vllm_ascend.cpu_binding.execute_command')
    def setUp(self, mock_execute_command):
        mock_execute_command.side_effect = [
            ("NPU ID  Chip ID  Chip Logic ID  Chip Name\n0 0 0 Ascend\n0 1 - Mcu\n1 0 1 Ascend", 0),
            ("| NPU Chip | Process id |\n| 0 0 | 1234 | vllm | 56000 |\n| 1 0 | 1235 | vllm | 56000 |", 0),
            ("", 0),
        ]
        self.device_info = DeviceInfo()

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_get_npu_map_info(self, mock_execute_command):
        execute_result_list = [
            ("NPU ID  Chip ID  Chip Logic ID  Chip Phy-ID Chip Name\n0 0 0 0 Ascend\n0 1 1 1 Ascend\n0 2 - - Mcu", 0),
            ("NPU ID  Chip ID  Chip Logic ID  Chip Name\n8 0 0 Ascend\n8 1 - Mcu\n9 0 1 Ascend", 0),
        ]
        result_list = [
            {'0': {'0': '0', '1': '1'}},
            {'8': {'0': '0'}, '9': {'0': '1'}},
        ]
        for result in execute_result_list:
            mock_execute_command.return_value = result
            npu_map_info = self.device_info.get_npu_map_info()
            expected = result_list.pop(0)
            self.assertEqual(npu_map_info, expected)

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_get_running_npus(self, mock_execute_command):
        mock_execute_command.side_effect = [
            ("| NPU Chip | Process id |\n| 0 1 | 1236 | vllm | 56000 |", 0),
            ("", 0),
            ("| NPU Chip | Process id |\n| 1 0 | 1236 | vllm | 56000 |", 0),
        ]
        with self.assertRaises(RuntimeError):
            self.device_info.get_running_npus()
        with self.assertRaises(RuntimeError):
            self.device_info.get_running_npus()
        running_npus = self.device_info.get_running_npus()
        self.assertEqual(len(running_npus), 1)

    @patch('vllm_ascend.cpu_binding.ASCEND_RT_VISIBLE_DEVICES', '1,5')
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_get_running_npus_filters_invalid_rows_and_visible_devices(self, mock_execute_command):
        device_info = object.__new__(DeviceInfo)
        device_info.npu_map_info = {'0': {'0': '0', '1': '1'}}
        mock_execute_command.return_value = (
            "ignored before header\n"
            "| NPU Chip | Process id |\n"
            "| malformed |\n"
            "| xx yy | 1001 | vllm |\n"
            "| 0 0 | 1234 | vllm |\n"
            "| 0 1 | 2345 | vllm |",
            0,
        )

        self.assertEqual(device_info.get_running_npus(), [1])

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_get_running_npus_skips_non_pipe_rows_inside_process_section(self, mock_execute_command):
        device_info = object.__new__(DeviceInfo)
        device_info.npu_map_info = {'0': {'0': '0'}}
        mock_execute_command.return_value = (
            "| NPU Chip | Process id |\n"
            "separator row\n"
            "| 0 0 | 1234 | vllm |",
            0,
        )

        self.assertEqual(device_info.get_running_npus(), [0])

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_parse_topo_affinity(self, mock_execute_command):
        mock_execute_command.return_value = ("NPU0 X HCCS HCCS HCCS HCCS HCCS HCCS HCCS 0-3", 0)
        affinity = self.device_info.parse_topo_affinity()
        expected = {0: [0, 1, 2, 3]}
        self.assertEqual(affinity, expected)

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_parse_topo_affinity_skips_affinity_header_and_non_npu_rows(self, mock_execute_command):
        device_info = object.__new__(DeviceInfo)
        mock_execute_command.return_value = (
            "HEADER\n"
            "NPU Chip Affinity\n"
            "not-an-npu row\n"
            "NPU0 x x x 2-3",
            0,
        )

        self.assertEqual(device_info.parse_topo_affinity(), {1: [2, 3]})

    def test_expand_cpu_list(self):
        result = self.device_info.expand_cpu_list("0-2, 4, 6-8")
        self.assertEqual(result, [0, 1, 2, 4, 6, 7, 8])

    def test_get_all_logic_npus(self):
        self.assertEqual(self.device_info.all_logic_npus, [0, 1])
        self.assertEqual(self.device_info.total_logic_npus, 2)

    def test_get_all_logic_npus_filters_invalid_values(self):
        device_info = object.__new__(DeviceInfo)
        device_info.npu_map_info = {
            '0': {'0': '0', '1': '', '2': 'abc'},
            '1': {'0': '2'},
        }

        self.assertEqual(device_info.get_all_logic_npus(), [0, 2])

    @patch('vllm_ascend.cpu_binding.os.path.exists', return_value=False)
    def test_parse_allowed_cpus_returns_empty_when_status_file_missing(self, _mock_exists):
        device_info = object.__new__(DeviceInfo)

        self.assertEqual(device_info.parse_allowed_cpus(), [])

    @patch('vllm_ascend.cpu_binding.os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data='Name:\tpython\nState:\tR\n')
    def test_parse_allowed_cpus_raises_when_field_missing(self, _mock_open, _mock_exists):
        device_info = object.__new__(DeviceInfo)

        with self.assertRaises(RuntimeError):
            device_info.parse_allowed_cpus()


class TestCpuAlloc(unittest.TestCase):

    @patch('vllm_ascend.cpu_binding.execute_command')
    def setUp(self, mock_execute_command):
        mock_execute_command.side_effect = [
            ("NPU ID  Chip ID  Chip Logic ID  Chip Name\n0 0 0 Ascend\n0 1 - Mcu\n1 0 1 Ascend", 0),
            ("| NPU Chip | Process id |\n| 0 0 | 1234 | vllm | 56000 |\n| 1 0 | 1235 | vllm | 56000 |", 0),
            ("", 0),
        ]
        self.cpu_alloc = CpuAlloc(0)

    def test_average_distribute(self):
        self.cpu_alloc.npu_cpu_pool = {0: [10, 11, 12, 13], 1: [10, 11, 12, 13]}
        groups = {'[10, 11, 12, 13]': [0, 1]}
        result = self.cpu_alloc.average_distribute(groups)
        self.assertEqual(result, {0: [10, 11], 1: [12, 13]})

        self.cpu_alloc.npu_cpu_pool = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        }
        groups = {'[0, 1, 2, 3, 4, 5]': [0, 1, 2]}
        result = self.cpu_alloc.average_distribute(groups)
        self.assertEqual(result, {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7],
            2: [8, 9, 10, 11, 12, 13],
        })

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type')
    def test_binding_mode_table(self, mock_get_device_type):
        mock_get_device_type.return_value = AscendDeviceType.A2
        self.assertEqual(self.cpu_alloc._binding_mode(), 'topo_affinity')
        mock_get_device_type.return_value = AscendDeviceType.A3
        self.assertEqual(self.cpu_alloc._binding_mode(), 'global_slice')

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type')
    def test_build_cpu_pools_fallback_to_global_slice(self, mock_get_device_type):
        mock_get_device_type.return_value = AscendDeviceType.A2
        self.cpu_alloc.device_info.npu_affinity = {}
        with patch.object(self.cpu_alloc, 'build_cpu_node_map') as mock_build_cpu_node_map, \
                patch.object(self.cpu_alloc, 'build_global_slice_cpu_pool') as mock_build_global_slice_cpu_pool:
            self.cpu_alloc.build_cpu_pools()
        mock_build_cpu_node_map.assert_called_once()
        mock_build_global_slice_cpu_pool.assert_called_once()

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type')
    def test_build_cpu_pools_global_slice_mode(self, mock_get_device_type):
        mock_get_device_type.return_value = AscendDeviceType.A3
        with patch.object(self.cpu_alloc, 'build_cpu_node_map') as mock_build_cpu_node_map, \
                patch.object(self.cpu_alloc, 'build_global_slice_cpu_pool') as mock_build_global_slice_cpu_pool:
            self.cpu_alloc.build_cpu_pools()
        mock_build_cpu_node_map.assert_called_once()
        mock_build_global_slice_cpu_pool.assert_called_once()

    def test_extend_numa(self):
        result = self.cpu_alloc.extend_numa([])
        self.assertEqual(result, [])
        self.cpu_alloc.cpu_node = {0: 0, 1: 0, 2: 1, 3: 1}
        self.cpu_alloc.numa_to_cpu_map = {0: [0, 1], 1: [2, 3]}
        self.cpu_alloc.device_info.allowed_cpus = [0, 1, 2, 3]
        result = self.cpu_alloc.extend_numa([0, 1])
        self.assertEqual(result, [0, 1, 2, 3])
        self.cpu_alloc.device_info.allowed_cpus = [0, 1, 3]
        result = self.cpu_alloc.extend_numa([0, 1])
        self.assertEqual(result, [0, 1, 3])

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_build_cpu_node_map(self, mock_execute_command):
        mock_execute_command.return_value = ('', 0)
        with self.assertRaises(RuntimeError):
            self.cpu_alloc.build_cpu_node_map()
        mock_execute_command.return_value = ('0 0\n1 1\n2 0\n3 1', 0)
        self.cpu_alloc.build_cpu_node_map()
        expected_cpu_node = {0: 0, 1: 1, 2: 0, 3: 1}
        expected_numa_to_cpu_map = {0: [0, 2], 1: [1, 3]}
        self.assertEqual(self.cpu_alloc.cpu_node, expected_cpu_node)
        self.assertEqual(self.cpu_alloc.numa_to_cpu_map, expected_numa_to_cpu_map)

    def test_build_global_slice_cpu_pool_uses_total_logic_npus(self):
        self.cpu_alloc.device_info.running_npu_list = [1]
        self.cpu_alloc.device_info.allowed_cpus = list(range(20))
        self.cpu_alloc.device_info.total_logic_npus = 2
        self.cpu_alloc.device_info.npu_affinity = {0: [0, 1], 1: [2, 3]}

        self.cpu_alloc.build_global_slice_cpu_pool()
        self.assertEqual(self.cpu_alloc.npu_cpu_pool[1], list(range(10, 20)))

    def test_build_global_slice_cpu_pool_fallback_to_affinity_len(self):
        self.cpu_alloc.device_info.running_npu_list = [0, 1]
        self.cpu_alloc.device_info.allowed_cpus = list(range(12))
        self.cpu_alloc.device_info.total_logic_npus = 0
        self.cpu_alloc.device_info.npu_affinity = {0: [0, 1], 1: [2, 3]}

        self.cpu_alloc.build_global_slice_cpu_pool()
        self.assertEqual(self.cpu_alloc.npu_cpu_pool[0], [0, 1, 2, 3, 4, 5])
        self.assertEqual(self.cpu_alloc.npu_cpu_pool[1], [6, 7, 8, 9, 10, 11])

    def test_build_global_slice_cpu_pool_fallback_to_running_len(self):
        self.cpu_alloc.device_info.running_npu_list = [0, 1]
        self.cpu_alloc.device_info.allowed_cpus = list(range(12))
        self.cpu_alloc.device_info.total_logic_npus = 0
        self.cpu_alloc.device_info.npu_affinity = {}

        self.cpu_alloc.build_global_slice_cpu_pool()
        self.assertEqual(self.cpu_alloc.npu_cpu_pool[0], [0, 1, 2, 3, 4, 5])
        self.assertEqual(self.cpu_alloc.npu_cpu_pool[1], [6, 7, 8, 9, 10, 11])

    def test_build_global_slice_cpu_pool_raises_when_cpu_insufficient(self):
        self.cpu_alloc.device_info.running_npu_list = [0, 1]
        self.cpu_alloc.device_info.allowed_cpus = list(range(8))
        self.cpu_alloc.device_info.total_logic_npus = 2

        with self.assertRaises(RuntimeError):
            self.cpu_alloc.build_global_slice_cpu_pool()

    def test_build_global_slice_cpu_pool_raises_invalid_npu_id(self):
        self.cpu_alloc.device_info.running_npu_list = [2]
        self.cpu_alloc.device_info.allowed_cpus = list(range(12))
        self.cpu_alloc.device_info.total_logic_npus = 2

        with self.assertRaises(RuntimeError):
            self.cpu_alloc.build_global_slice_cpu_pool()

    def test_build_global_slice_cpu_pool_returns_when_running_or_allowed_empty(self):
        self.cpu_alloc.device_info.running_npu_list = []
        self.cpu_alloc.device_info.allowed_cpus = list(range(12))
        self.cpu_alloc.build_global_slice_cpu_pool()
        self.assertEqual(self.cpu_alloc.npu_cpu_pool, {})

        self.cpu_alloc.device_info.running_npu_list = [0]
        self.cpu_alloc.device_info.allowed_cpus = []
        self.cpu_alloc.build_global_slice_cpu_pool()
        self.assertEqual(self.cpu_alloc.npu_cpu_pool, {})

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_allocate(self, _mock_execute_command):
        self.cpu_alloc.device_info.running_npu_list = [0]
        self.cpu_alloc.npu_cpu_pool = {0: [0, 1, 2, 3, 4]}
        self.cpu_alloc.allocate()
        self.assertEqual(self.cpu_alloc.assign_main[0], [2])
        self.assertEqual(self.cpu_alloc.assign_acl[0], [3])
        self.assertEqual(self.cpu_alloc.assign_rel[0], [4])
        self.cpu_alloc.npu_cpu_pool = {0: [0, 1]}
        with self.assertRaises(RuntimeError):
            self.cpu_alloc.allocate()

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_threads(self, mock_execute_command):
        thread_message = '1234 1234 ? 00:00:03 acl_thread\n4567 4567 ? 00:00:03 release_thread'
        mock_execute_command.return_value = (thread_message, 0)
        self.cpu_alloc.device_info.running_npu_list = [0]
        self.cpu_alloc.assign_main = {0: [0, 1]}
        self.cpu_alloc.assign_acl = {0: [2]}
        self.cpu_alloc.assign_rel = {0: [3]}
        self.cpu_alloc.bind_threads()
        mock_execute_command.assert_called()

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type')
    @patch('vllm_ascend.cpu_binding.os.listdir')
    @patch('builtins.open', new_callable=mock_open, read_data='123: 0 0 0 0 sq_send_trigger_irq\n')
    @patch('vllm_ascend.cpu_binding.shutil.which')
    @patch('vllm_ascend.cpu_binding.os.access')
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_npu_irq_a3_uses_card_chip_mapping(self, mock_execute_command, mock_access,
                                                     mock_which, _mock_open, mock_listdir,
                                                     mock_get_device_type):
        mock_access.return_value = True
        mock_which.return_value = None
        mock_listdir.side_effect = FileNotFoundError
        mock_get_device_type.return_value = AscendDeviceType.A3
        mock_execute_command.return_value = ('PCIe Bus Info 0000:03:00.0', 0)
        self.cpu_alloc.rank_id = 0
        self.cpu_alloc.device_info.running_npu_list = [3]
        self.cpu_alloc.npu_cpu_pool = {3: [0, 1, 2, 3, 4]}

        self.cpu_alloc.bind_npu_irq()

        mock_execute_command.assert_any_call(['npu-smi', 'info', '-t', 'board', '-i', '1', '-c', '1'])


class TestCpuBindingSupplemental(unittest.TestCase):

    def test_cpu_to_mask_handles_single_and_multi_group_masks(self):
        self.assertEqual(CpuAlloc.cpu_to_mask(3), '00000008')
        self.assertEqual(CpuAlloc.cpu_to_mask(35), '00000008,00000000')

    def test_get_threads_map_skips_irrelevant_lines(self):
        thread_message = (
            'bad-line\n'
            '123 456 ? 00:00:01 acl_thread\n'
            '123 789 ? 00:00:01 release_thread\n'
            '123 999 ? 00:00:01 worker_thread\n'
            '555 666 ? 00:00:01 acl_thread'
        )

        self.assertEqual(
            CpuAlloc.get_threads_map(thread_message),
            {
                '123': {'acl_thread': ['456'], 'release_thread': ['789']},
                '555': {'acl_thread': ['666'], 'release_thread': []},
            },
        )

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_skips_empty_cpu_list(self, mock_execute_command):
        CpuAlloc.bind('123', [], False)

        mock_execute_command.assert_not_called()

    @patch('vllm_ascend.cpu_binding.execute_command', return_value=('ok', 0))
    def test_bind_uses_sub_thread_flag(self, mock_execute_command):
        CpuAlloc.bind('123', [1, 2], True)

        mock_execute_command.assert_called_once_with(['taskset', '-acp', '1,2', '123'])

    @patch('vllm_ascend.cpu_binding.execute_command', return_value=('failed', 1))
    def test_bind_raises_for_failed_taskset(self, mock_execute_command):
        with self.assertRaises(RuntimeError):
            CpuAlloc.bind('123', [1, 2], False)

        mock_execute_command.assert_called_once_with(['taskset', '-cp', '1,2', '123'])

    def test_extend_numa_returns_original_list_when_multiple_nodes_present(self):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.cpu_node = {0: 0, 1: 1}

        self.assertEqual(cpu_alloc.extend_numa([0, 1]), [0, 1])

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_build_cpu_node_map_skips_blank_and_header_rows(self, mock_execute_command):
        cpu_alloc = make_cpu_alloc()
        mock_execute_command.return_value = ('CPU NODE\n\n0 0\n1 1', 0)

        cpu_alloc.build_cpu_node_map()

        self.assertEqual(cpu_alloc.cpu_node, {0: 0, 1: 1})
        self.assertEqual(cpu_alloc.numa_to_cpu_map, {0: [0], 1: [1]})

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type', return_value='unknown')
    def test_binding_mode_defaults_to_topo_affinity_for_unknown_device(self, _mock_get_device_type):
        self.assertEqual(CpuAlloc._binding_mode(), 'topo_affinity')

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type', return_value=AscendDeviceType.A2)
    def test_build_cpu_pools_raises_on_affinity_conflict(self, _mock_get_device_type):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [0]
        cpu_alloc.device_info.allowed_cpus = [8, 9]
        cpu_alloc.device_info.npu_affinity = {0: [0, 1]}

        with patch.object(cpu_alloc, 'build_cpu_node_map'):
            with self.assertRaises(RuntimeError):
                cpu_alloc.build_cpu_pools()

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type', return_value=AscendDeviceType.A2)
    def test_build_cpu_pools_topo_mode_builds_and_splits_duplicate_groups(self, _mock_get_device_type):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [0, 1, 2]
        cpu_alloc.device_info.allowed_cpus = [0, 1, 2, 3]
        cpu_alloc.device_info.npu_affinity = {0: [0, 1], 1: [2, 3], 2: [2, 3]}

        with patch.object(cpu_alloc, 'build_cpu_node_map'), \
                patch.object(cpu_alloc, 'extend_numa', side_effect=lambda cpus: cpus):
            cpu_alloc.build_cpu_pools()

        self.assertEqual(cpu_alloc.npu_cpu_pool, {0: [0, 1], 1: [2], 2: [3]})

    @patch('vllm_ascend.cpu_binding.logger.info')
    def test_print_plan_handles_empty_release_assignment(self, mock_logger_info):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [1]
        cpu_alloc.rank_id = 0
        cpu_alloc.assign_main = {1: [2, 3]}
        cpu_alloc.assign_acl = {1: [4]}
        cpu_alloc.assign_rel = {1: []}

        cpu_alloc.print_plan()

        self.assertEqual(mock_logger_info.call_count, 2)

    @patch('vllm_ascend.cpu_binding.logger.info')
    def test_print_plan_handles_non_empty_release_assignment(self, mock_logger_info):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [1]
        cpu_alloc.rank_id = 0
        cpu_alloc.assign_main = {1: [2, 3]}
        cpu_alloc.assign_acl = {1: [4]}
        cpu_alloc.assign_rel = {1: [5]}

        cpu_alloc.print_plan()

        self.assertEqual(mock_logger_info.call_count, 2)

    @patch('vllm_ascend.cpu_binding.shutil.which', return_value=None)
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_memory_skips_when_migratepages_missing(self, mock_execute_command, _mock_which):
        cpu_alloc = make_cpu_alloc()

        cpu_alloc.bind_memory('999', 0)

        mock_execute_command.assert_not_called()

    @patch('vllm_ascend.cpu_binding.shutil.which', return_value='/usr/bin/migratepages')
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_memory_skips_when_cpu_pool_or_numa_invalid(self, mock_execute_command, _mock_which):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.numa_to_cpu_map = {0: [0], 1: [1]}
        cpu_alloc.bind_memory('1000', 0)
        mock_execute_command.assert_not_called()

        cpu_alloc.npu_cpu_pool = {0: [8]}
        cpu_alloc.cpu_node = {8: 3}
        cpu_alloc.bind_memory('1000', 0)
        mock_execute_command.assert_not_called()

    @patch('vllm_ascend.cpu_binding.shutil.which', return_value='/usr/bin/migratepages')
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_memory_executes_on_valid_numa_target(self, mock_execute_command, _mock_which):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.npu_cpu_pool = {0: [8, 9]}
        cpu_alloc.cpu_node = {8: 1}
        cpu_alloc.numa_to_cpu_map = {0: [0], 1: [8, 9]}

        cpu_alloc.bind_memory('1000', 0)

        mock_execute_command.assert_called_once_with(['migratepages', '1000', '0,1', '1'])

    @patch('vllm_ascend.cpu_binding.psutil.Process')
    @patch('vllm_ascend.cpu_binding.execute_command', return_value=(
        '1000 2000 ? 00:00:01 acl_thread\n1000 3000 ? 00:00:01 release_thread',
        0,
    ))
    def test_bind_threads_binds_main_acl_and_release_threads(self, _mock_execute_command, mock_process):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [0]
        cpu_alloc.assign_main = {0: [1, 2]}
        cpu_alloc.assign_acl = {0: [3]}
        cpu_alloc.assign_rel = {0: [4]}
        mock_process.return_value.pid = 1000

        with patch.object(cpu_alloc, 'bind') as mock_bind, patch.object(cpu_alloc, 'bind_memory') as mock_bind_memory:
            cpu_alloc.bind_threads()

        self.assertEqual(
            mock_bind.call_args_list,
            [
                call('1000', [1, 2], True),
                call('2000', [3], False),
                call('3000', [4], False),
            ],
        )
        mock_bind_memory.assert_called_once_with('1000', 0)

    @patch('vllm_ascend.cpu_binding.os.access', return_value=False)
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_npu_irq_returns_when_irq_path_not_writable(self, mock_execute_command, _mock_access):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.bind_npu_irq()

        mock_execute_command.assert_not_called()

    @patch('vllm_ascend.cpu_binding.os.access', return_value=True)
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_npu_irq_returns_when_current_npu_has_no_cpu_pool(self, mock_execute_command, _mock_access):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [0]
        cpu_alloc.npu_cpu_pool = {}

        cpu_alloc.bind_npu_irq()

        mock_execute_command.assert_not_called()

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type', return_value=AscendDeviceType.A2)
    @patch('builtins.open', new_callable=mock_open, read_data='123: 0 0 0 sq_send_trigger_irq\n')
    @patch('vllm_ascend.cpu_binding.shutil.which', return_value=None)
    @patch('vllm_ascend.cpu_binding.os.access', return_value=True)
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_npu_irq_skips_when_cpu_pool_too_small(self, mock_execute_command, _mock_access,
                                                        _mock_which, _mock_open, _mock_get_device_type):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [0]
        cpu_alloc.npu_cpu_pool = {0: [7]}

        cpu_alloc.bind_npu_irq()

        mock_execute_command.assert_not_called()

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type', return_value=AscendDeviceType.A2)
    @patch('builtins.open', new_callable=mock_open, read_data='123: 0 0 0 sq_send_trigger_irq\n')
    @patch('vllm_ascend.cpu_binding.shutil.which', return_value=None)
    @patch('vllm_ascend.cpu_binding.os.access', return_value=True)
    @patch('vllm_ascend.cpu_binding.execute_command', return_value=('board info without pci', 0))
    def test_bind_npu_irq_skips_when_pci_address_missing(self, mock_execute_command, _mock_access,
                                                         _mock_which, _mock_open, _mock_get_device_type):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [0]
        cpu_alloc.npu_cpu_pool = {0: [7, 8]}

        cpu_alloc.bind_npu_irq()

        mock_execute_command.assert_called_once_with(['npu-smi', 'info', '-t', 'board', '-i', '0'])

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type', return_value=AscendDeviceType.A2)
    @patch('vllm_ascend.cpu_binding.os.listdir', return_value=['456', '457'])
    @patch('builtins.open', new_callable=mock_open, read_data='123: 0 0 0 sq_send_trigger_irq\n')
    @patch('vllm_ascend.cpu_binding.shutil.which', return_value=None)
    @patch('vllm_ascend.cpu_binding.os.access', return_value=True)
    @patch('vllm_ascend.cpu_binding.execute_command', return_value=('prefix\nPCIe Bus Info 0000:03:00.0', 0))
    def test_bind_npu_irq_skips_when_sq_irq_not_found(self, _mock_execute_command, _mock_access,
                                                      _mock_which, _mock_open, _mock_listdir,
                                                      _mock_get_device_type):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [0]
        cpu_alloc.npu_cpu_pool = {0: [7, 8]}

        cpu_alloc.bind_npu_irq()

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type', return_value=AscendDeviceType.A2)
    @patch('vllm_ascend.cpu_binding.os.listdir', return_value=['123', '124'])
    @patch('builtins.open', new_callable=mock_open, read_data='123: 0 0 0 sq_send_trigger_irq\n')
    @patch('vllm_ascend.cpu_binding.shutil.which', return_value='/bin/systemctl')
    @patch('vllm_ascend.cpu_binding.os.access', return_value=True)
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_npu_irq_stops_irqbalance_and_writes_affinity_masks(self, mock_execute_command,
                                                                     _mock_access, _mock_which,
                                                                     mock_file, _mock_listdir,
                                                                     _mock_get_device_type):
        mock_execute_command.side_effect = [
            ('irqbalance.service enabled\n', 0),
            ('', 0),
            ('stopped', 0),
            ('prefix\nPCIe Bus Info 0000:03:00.0', 0),
        ]
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [0]
        cpu_alloc.npu_cpu_pool = {0: [8, 9, 10]}

        cpu_alloc.bind_npu_irq()

        self.assertIn(call(['systemctl', 'stop', 'irqbalance']), mock_execute_command.call_args_list)
        self.assertIn(call(['npu-smi', 'info', '-t', 'board', '-i', '0']), mock_execute_command.call_args_list)
        handle = mock_file()
        self.assertEqual(handle.write.call_args_list, [call('00000100'), call('00000200')])

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type', return_value=AscendDeviceType.A2)
    @patch('vllm_ascend.cpu_binding.os.listdir', return_value=['123', '124'])
    @patch('builtins.open', new_callable=mock_open, read_data='123: 0 0 0 sq_send_trigger_irq\n')
    @patch('vllm_ascend.cpu_binding.shutil.which', return_value='/bin/systemctl')
    @patch('vllm_ascend.cpu_binding.os.access', return_value=True)
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_npu_irq_keeps_irqbalance_when_inactive(self, mock_execute_command, _mock_access,
                                                         _mock_which, _mock_open, _mock_listdir,
                                                         _mock_get_device_type):
        mock_execute_command.side_effect = [
            ('irqbalance.service enabled\n', 0),
            ('', 3),
            ('prefix\nPCIe Bus Info 0000:03:00.0', 0),
        ]
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [0]
        cpu_alloc.npu_cpu_pool = {0: [8, 9, 10]}

        cpu_alloc.bind_npu_irq()

        self.assertNotIn(call(['systemctl', 'stop', 'irqbalance']), mock_execute_command.call_args_list)

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type', return_value=AscendDeviceType.A2)
    @patch('vllm_ascend.cpu_binding.os.listdir', return_value=['123', '124'])
    @patch('builtins.open', new_callable=mock_open, read_data='123: 0 0 0 sq_send_trigger_irq\n')
    @patch('vllm_ascend.cpu_binding.shutil.which', return_value='/bin/systemctl')
    @patch('vllm_ascend.cpu_binding.os.access', return_value=True)
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_npu_irq_skips_irqbalance_handling_when_service_absent(self, mock_execute_command, _mock_access,
                                                                        _mock_which, _mock_open, _mock_listdir,
                                                                        _mock_get_device_type):
        mock_execute_command.side_effect = [
            ('another.service enabled\n', 0),
            ('prefix\nPCIe Bus Info 0000:03:00.0', 0),
        ]
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [0]
        cpu_alloc.npu_cpu_pool = {0: [8, 9, 10]}

        cpu_alloc.bind_npu_irq()

        self.assertNotIn(call(['systemctl', 'is-active', '--quiet', 'irqbalance']), mock_execute_command.call_args_list)

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type', return_value=AscendDeviceType.A2)
    @patch('vllm_ascend.cpu_binding.os.listdir', return_value=['123', '124'])
    @patch(
        'builtins.open',
        new_callable=mock_open,
        read_data='100: 0 0 0 other_irq\n123: 0 0 0 sq_send_trigger_irq\n',
    )
    @patch('vllm_ascend.cpu_binding.shutil.which', return_value=None)
    @patch('vllm_ascend.cpu_binding.os.access', return_value=True)
    @patch('vllm_ascend.cpu_binding.execute_command', return_value=('prefix\nPCIe Bus Info 0000:03:00.0', 0))
    def test_bind_npu_irq_scans_multiple_interrupt_lines(self, _mock_execute_command, _mock_access,
                                                         _mock_which, mock_file, _mock_listdir,
                                                         _mock_get_device_type):
        cpu_alloc = make_cpu_alloc()
        cpu_alloc.device_info.running_npu_list = [0]
        cpu_alloc.npu_cpu_pool = {0: [8, 9, 10]}

        cpu_alloc.bind_npu_irq()

        handle = mock_file()
        self.assertEqual(handle.write.call_args_list, [call('00000100'), call('00000200')])

    def test_run_all_invokes_steps_in_order(self):
        cpu_alloc = make_cpu_alloc()
        calls = []

        with patch.object(cpu_alloc, 'build_cpu_pools', side_effect=lambda: calls.append('build_cpu_pools')), \
                patch.object(cpu_alloc, 'allocate', side_effect=lambda: calls.append('allocate')), \
                patch.object(cpu_alloc, 'print_plan', side_effect=lambda: calls.append('print_plan')), \
                patch.object(cpu_alloc, 'bind_threads', side_effect=lambda: calls.append('bind_threads')), \
                patch.object(cpu_alloc, 'bind_npu_irq', side_effect=lambda: calls.append('bind_npu_irq')):
            cpu_alloc.run_all()

        self.assertEqual(calls, ['build_cpu_pools', 'allocate', 'print_plan', 'bind_threads', 'bind_npu_irq'])


class TestBindingSwitch(unittest.TestCase):

    @patch('vllm_ascend.cpu_binding.platform.machine')
    def test_is_arm_cpu(self, mock_machine):
        mock_machine.return_value = 'x86_64'
        self.assertFalse(is_arm_cpu())
        mock_machine.return_value = 'aarch64'
        self.assertTrue(is_arm_cpu())
        mock_machine.return_value = 'armv8'
        self.assertTrue(is_arm_cpu())
        mock_machine.return_value = 'mips64'
        self.assertFalse(is_arm_cpu())

    @patch('vllm_ascend.cpu_binding.CpuAlloc')
    @patch('vllm_ascend.cpu_binding.is_arm_cpu')
    def test_bind_cpus_skip_non_arm(self, mock_is_arm_cpu, mock_cpu_alloc):
        mock_is_arm_cpu.return_value = False
        bind_cpus(0)
        mock_cpu_alloc.assert_not_called()

    @patch('vllm_ascend.cpu_binding.CpuAlloc')
    @patch('vllm_ascend.cpu_binding.is_arm_cpu', return_value=True)
    def test_bind_cpus_runs_allocator_on_arm(self, _mock_is_arm_cpu, mock_cpu_alloc):
        bind_cpus(1)

        mock_cpu_alloc.assert_called_once_with(1)
        mock_cpu_alloc.return_value.run_all.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
