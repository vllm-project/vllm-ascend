from unittest.mock import patch

from vllm_ascend.cpu_binding import CpuAlloc, DeviceInfo


def _device(mapping):
    device = DeviceInfo.__new__(DeviceInfo)
    device.npu_map_info = mapping
    return device


def test_physical_topology_expands_to_logical_chips():
    device = _device({"32768": {"0": "0", "1": "1"}, "32896": {"0": "2", "1": "3"}})
    with patch("vllm_ascend.cpu_binding.execute_command", return_value=("NPU32768 48-71\nNPU32896 72-95", 0)):
        assert device.parse_topo_affinity() == {
            0: list(range(48, 72)),
            1: list(range(48, 72)),
            2: list(range(72, 96)),
            3: list(range(72, 96)),
        }


def test_overlapping_physical_and_logical_ids_keep_logical_rows():
    device = _device({"0": {"0": "0", "1": "1"}, "1": {"0": "2", "1": "3"}})
    with patch("vllm_ascend.cpu_binding.execute_command", return_value=("NPU0 0-3\nNPU1 4-7", 0)):
        assert device.parse_topo_affinity() == {0: list(range(4)), 1: list(range(4, 8))}


def test_irq_query_uses_physical_chip_mapping():
    binder = CpuAlloc.__new__(CpuAlloc)
    binder.device_info = _device({"32768": {"0": "0", "1": "1"}})
    with patch("vllm_ascend.cpu_binding.execute_command", return_value=("PCIe Bus Info : 0000:01:00.0", 0)) as command:
        info = binder._get_board_info(1)
    command.assert_called_once_with(["npu-smi", "info", "-t", "board", "-i", "32768", "-c", "1"])
    assert binder._parse_pci_addr(info) == "0000:01:00.0"
