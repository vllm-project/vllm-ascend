import unittest
from unittest.mock import mock_open, patch

from vllm_ascend.cpu_binding import CpuAlloc, DeviceInfo, bind_cpus, is_arm_cpu
from vllm_ascend.utils import AscendDeviceType


class TestDeviceInfo(unittest.TestCase):

    @patch('vllm_ascend.cpu_binding.execute_command')
    def setUp(self, mock_execute_command):
        mock_execute_command.side_effect = [
            ("NPU ID  Chip ID  Chip Logic ID  Chip Name\n0 0 0 Ascend\n0 1 - Mcu\n1 0 1 Ascend",
             0),
            ("| NPU Chip | Process id |\n| 0 0 | 1234 | vllm | 56000 |\n| 1 0 | 1235 | vllm | 56000 |",
             0), ("", 0)
        ]
        self.device_info = DeviceInfo()

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_get_npu_map_info(self, mock_execute_command):
        execute_result_list = [
            ("NPU ID  Chip ID  Chip Logic ID  Chip Phy-ID Chip Name\n0 0 0 0 Ascend\n0 1 1 1 Ascend\n0 2 - - Mcu",
             0),
            ("NPU ID  Chip ID  Chip Logic ID  Chip Name\n8 0 0 Ascend\n8 1 - Mcu\n9 0 1 Ascend",
             0),
        ]
        result_list = [{
            '0': {
                '0': '0',
                '1': '1'
            }
        }, {
            '8': {
                '0': '0'
            },
            '9': {
                '0': '1'
            }
        }]
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
            ("| NPU Chip | Process id |\n| 1 0 | 1236 | vllm | 56000 |", 0)
        ]
        with self.assertRaises(RuntimeError):
            self.device_info.get_running_npus()
        with self.assertRaises(RuntimeError):
            self.device_info.get_running_npus()
        running_npus = self.device_info.get_running_npus()
        self.assertEqual(len(running_npus), 1)

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_parse_topo_affinity(self, mock_execute_command):
        mock_execute_command.return_value = (
            "NPU0 X HCCS HCCS HCCS HCCS HCCS HCCS HCCS 0-3", 0)
        affinity = self.device_info.parse_topo_affinity()
        expected = {0: [0, 1, 2, 3]}
        self.assertEqual(affinity, expected)

    def test_expand_cpu_list(self):
        result = self.device_info.expand_cpu_list("0-2, 4, 6-8")
        self.assertEqual(result, [0, 1, 2, 4, 6, 7, 8])

    def test_get_all_logic_npus(self):
        self.assertEqual(self.device_info.all_logic_npus, [0, 1])
        self.assertEqual(self.device_info.total_logic_npus, 2)


class TestCpuAlloc(unittest.TestCase):

    @patch('vllm_ascend.cpu_binding.execute_command')
    def setUp(self, mock_execute_command):
        mock_execute_command.side_effect = [
            ("NPU ID  Chip ID  Chip Logic ID  Chip Name\n0 0 0 Ascend\n0 1 - Mcu\n1 0 1 Ascend",
             0),
            ("| NPU Chip | Process id |\n| 0 0 | 1234 | vllm | 56000 |\n| 1 0 | 1235 | vllm | 56000 |",
             0), ("", 0)
        ]
        self.cpu_alloc = CpuAlloc(0)

    def test_average_distribute(self):
        self.cpu_alloc.npu_cpu_pool = {
            0: [10, 11, 12, 13],
            1: [10, 11, 12, 13]
        }
        groups = {"[10, 11, 12, 13]": [0, 1]}
        result = self.cpu_alloc.average_distribute(groups)
        self.assertEqual(result, {0: [10, 11], 1: [12, 13]})
        self.cpu_alloc.npu_cpu_pool = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        }
        groups = {"[0, 1, 2, 3, 4, 5]": [0, 1, 2]}
        result = self.cpu_alloc.average_distribute(groups)
        self.assertEqual(result, {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7],
            2: [8, 9, 10, 11, 12, 13]
        })

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type')
    def test_binding_mode_table(self, mock_get_device_type):
        mock_get_device_type.return_value = AscendDeviceType.A2
        self.assertEqual(self.cpu_alloc._binding_mode(), "topo_affinity")
        mock_get_device_type.return_value = AscendDeviceType.A3
        self.assertEqual(self.cpu_alloc._binding_mode(), "global_slice")

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type')
    def test_build_cpu_pools_fallback_to_global_slice(self, mock_get_device_type):
        mock_get_device_type.return_value = AscendDeviceType.A2
        self.cpu_alloc.device_info.npu_affinity = {}
        with patch.object(self.cpu_alloc, "build_cpu_node_map") as mock_build_cpu_node_map, \
                patch.object(self.cpu_alloc, "build_global_slice_cpu_pool") as mock_build_global_slice_cpu_pool:
            self.cpu_alloc.build_cpu_pools()
        mock_build_cpu_node_map.assert_called_once()
        mock_build_global_slice_cpu_pool.assert_called_once()

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type')
    def test_build_cpu_pools_global_slice_mode(self, mock_get_device_type):
        mock_get_device_type.return_value = AscendDeviceType.A3
        with patch.object(self.cpu_alloc, "build_cpu_node_map") as mock_build_cpu_node_map, \
                patch.object(self.cpu_alloc, "build_global_slice_cpu_pool") as mock_build_global_slice_cpu_pool:
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
        mock_execute_command.return_value = ("", 0)
        with self.assertRaises(RuntimeError):
            self.cpu_alloc.build_cpu_node_map()
        mock_execute_command.return_value = ("0 0\n1 1\n2 0\n3 1", 0)
        self.cpu_alloc.build_cpu_node_map()
        expected_cpu_node = {0: 0, 1: 1, 2: 0, 3: 1}
        expected_numa_to_cpu_map = {0: [0, 2], 1: [1, 3]}
        self.assertEqual(self.cpu_alloc.cpu_node, expected_cpu_node)
        self.assertEqual(self.cpu_alloc.numa_to_cpu_map,
                         expected_numa_to_cpu_map)

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

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_allocate(self, mock_execute_command):
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
        thread_message = "1234 1234 ? 00:00:03 acl_thread\n4567 4567 ? 00:00:03 release_thread"
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
        mock_execute_command.return_value = ("PCIe Bus Info 0000:03:00.0", 0)
        self.cpu_alloc.rank_id = 0
        self.cpu_alloc.device_info.running_npu_list = [3]
        self.cpu_alloc.npu_cpu_pool = {3: [0, 1, 2, 3, 4]}

        self.cpu_alloc.bind_npu_irq()

        mock_execute_command.assert_any_call(["npu-smi", "info", "-t", "board", "-i", "1", "-c", "1"])


class TestBindingSwitch(unittest.TestCase):

    @patch('vllm_ascend.cpu_binding.platform.machine')
    def test_is_arm_cpu(self, mock_machine):
        mock_machine.return_value = "x86_64"
        self.assertFalse(is_arm_cpu())
        mock_machine.return_value = "aarch64"
        self.assertTrue(is_arm_cpu())
        mock_machine.return_value = "armv8"
        self.assertTrue(is_arm_cpu())
        mock_machine.return_value = "mips64"
        self.assertFalse(is_arm_cpu())

    @patch('vllm_ascend.cpu_binding.CpuAlloc')
    @patch('vllm_ascend.cpu_binding.is_arm_cpu')
    def test_bind_cpus_skip_non_arm(self, mock_is_arm_cpu, mock_cpu_alloc):
        mock_is_arm_cpu.return_value = False
        bind_cpus(0)
        mock_cpu_alloc.assert_not_called()


if __name__ == '__main__':
    unittest.main()
