from __future__ import annotations

import faulthandler
import os
import sys
import unittest
from pathlib import Path
import time

import torch


MODEL_DIR = Path(__file__).resolve().parent
DAEMON_DIR = MODEL_DIR.parent / "daemon"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(DAEMON_DIR) not in sys.path:
    sys.path.insert(0, str(DAEMON_DIR))

from camem import CaMemAllocator, camem_available  # noqa: E402


MiB = 1024 * 1024
GiB = 1024 * 1024 * 1024

faulthandler.enable(all_threads=True)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _log(msg: str) -> None:
    print(f"[test_camem_allocator] {msg}", flush=True)


def _safe_sync(where: str) -> None:
    if hasattr(torch, "npu") and torch.npu.is_available():
        _log(f"sync enter: {where}")
        torch.npu.synchronize()
        _log(f"sync done : {where}")


class TestCaMemAllocator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _log("setUpClass enter")
        if not camem_available:
            raise unittest.SkipTest("camem extension is unavailable")
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise unittest.SkipTest("torch.npu is unavailable")

        os.environ.setdefault("MDAEMON_KV_INIT_MB", "64")

        default_device = int(torch.npu.current_device())
        cls.device_id = _env_int("TEST_NPU_DEVICE_ID", default_device)
        device_count = int(torch.npu.device_count()) if hasattr(torch.npu, "device_count") else 1
        if cls.device_id < 0 or cls.device_id >= device_count:
            raise unittest.SkipTest(
                f"Invalid TEST_NPU_DEVICE_ID={cls.device_id}, available range is [0, {device_count - 1}]"
            )
        torch.npu.set_device(cls.device_id)
        cls.device = f"npu:{cls.device_id}"
        _log(
            f"current device={cls.device_id}, device={cls.device}, "
            f"MDAEMON_KV_INIT_MB={os.environ.get('MDAEMON_KV_INIT_MB')}"
        )

        CaMemAllocator.instance = None
        try:
            cls.alloc = CaMemAllocator.get_instance()
            _log("CaMemAllocator.get_instance done")
        except Exception as exc:
            raise unittest.SkipTest(
                "Failed to initialize CaMemAllocator. Ensure daemon is launched "
                "externally and model-side shm/semaphore envs are configured. "
                f"Root error: {exc}"
            )
        _log("setUpClass done")

    @classmethod
    def tearDownClass(cls) -> None:
        _log("tearDownClass enter")
        try:
            if hasattr(cls, "alloc") and cls.alloc is not None:
                _log("allocator close begin")
                cls.alloc.close()
                _log("allocator close done")
        finally:
            CaMemAllocator.instance = None
            _log("tearDownClass done")

    def _dump_pointer_state(self, where: str) -> None:
        size = len(self.alloc.pointer_to_data)
        _log(f"{where}: pointer_to_data size={size}")
        for ptr, data in list(self.alloc.pointer_to_data.items()):
            _log(
                f"  ptr=0x{ptr:x} kind={data.kind} tag={data.tag} "
                f"mapped={data.mapped_bytes} handle_size={data.handle[1]} "
                f"cpu_backup={'yes' if data.cpu_backup_tensor is not None else 'no'}"
            )

    def _alloc_weight_tensor(self, size_bytes: int, tag: str) -> torch.Tensor:
        with self.alloc.use_weight_memory_pool(tag=tag):
            tensor = torch.empty(size_bytes, dtype=torch.bfloat16, device=self.device)
        return tensor

    def _alloc_kvcache_tensor(self, size_bytes: int, tag: str) -> torch.Tensor:
        with self.alloc.use_kvcache_memory_pool(tag=tag):
            tensor = torch.empty(size_bytes, dtype=torch.bfloat16, device=self.device)
        return tensor

    # def test_weight_kvcache_data_integrity_after_sleep_wakeup(self) -> None:
    #     _log("test_weight_kvcache_data_integrity_after_sleep_wakeup enter")
    #     weight_tag = "weight_integrity"
    #     kv_tag = "kv_integrity"

    #     with self.alloc.use_weight_memory_pool(tag=weight_tag):
    #         weight_tensor = torch.arange(4 * MiB,
    #                                      dtype=torch.bfloat16,
    #                                      device=self.device)
    #     with self.alloc.use_kvcache_memory_pool(tag=kv_tag):
    #         kvcache_tensor = torch.arange(6 * MiB,
    #                                       dtype=torch.bfloat16,
    #                                       device=self.device)

    #     _safe_sync("after integrity tensor alloc+write")

    #     expected_weight = weight_tensor.cpu().clone()
    #     expected_kv = kvcache_tensor.cpu().clone()

    #     _log("sleep for 10 seconds to allow manual inspection if needed")
    #     time.sleep(10)

    #     self.alloc.sleep(offload_tags=(weight_tag, kv_tag))
    #     _safe_sync("after integrity sleep")

    #     _log("sleep for 10 seconds to allow manual inspection if needed")
    #     time.sleep(10)

    #     self.alloc.wake_up(tags=[weight_tag, kv_tag])
    #     _safe_sync("after integrity wake_up")

    #     got_weight = weight_tensor.cpu()
    #     got_kv = kvcache_tensor.cpu()

    #     self.assertTrue(torch.equal(got_weight, expected_weight),
    #                     "weight tensor data mismatch after sleep/wake")
    #     self.assertTrue(torch.equal(got_kv, expected_kv),
    #                     "kvcache tensor data mismatch after sleep/wake")
    #     _log("test_weight_kvcache_data_integrity_after_sleep_wakeup done")

    def test_weight_kvcache_sleep_wakeup_and_free(self) -> None:
        _log("test_weight_kvcache_sleep_wakeup_and_free enter")
        weight_tag = "weight_tag"
        kv_tag = "kv_tag"

        _log("alloc weight tensor begin")
        weight_tensor = self._alloc_weight_tensor(2411 * MiB, tag=weight_tag)
        _log("alloc weight tensor done")
        _safe_sync("after weight alloc")

        _log("alloc kvcache tensor begin")
        kvcache_tensor = self._alloc_kvcache_tensor(1963 * MiB, tag=kv_tag)
        _log("alloc kvcache tensor done")
        _safe_sync("after kvcache alloc")

        weight_ptr = int(weight_tensor.data_ptr())
        kv_ptr = int(kvcache_tensor.data_ptr())
        _log(f"weight_ptr=0x{weight_ptr:x}, kv_ptr=0x{kv_ptr:x}")
        self._dump_pointer_state("after allocations")

        self.assertIn(weight_ptr, self.alloc.pointer_to_data)
        self.assertIn(kv_ptr, self.alloc.pointer_to_data)

        weight_data = self.alloc.pointer_to_data[weight_ptr]
        kv_data = self.alloc.pointer_to_data[kv_ptr]

        self.assertEqual(weight_data.kind, "weight")
        self.assertEqual(kv_data.kind, "kvcache")
        self.assertGreater(weight_data.mapped_bytes, 0)
        self.assertGreaterEqual(weight_data.mapped_bytes, weight_data.handle[1])

        kv_expected_mapped = min(kv_data.handle[1], _env_int("MDAEMON_KV_INIT_MB", 128) * MiB)
        self.assertEqual(kv_data.mapped_bytes, kv_expected_mapped)
        _log(f"kv_expected_mapped={kv_expected_mapped}")

        _log("sleep for 10 seconds to allow manual inspection if needed")
        time.sleep(10)

        _log("sleep begin")
        self.alloc.sleep(offload_tags=(weight_tag,))
        _log("sleep done")
        _safe_sync("after sleep")
        self._dump_pointer_state("after sleep")

        weight_data = self.alloc.pointer_to_data[weight_ptr]
        kv_data = self.alloc.pointer_to_data[kv_ptr]

        self.assertIsNotNone(weight_data.cpu_backup_tensor)
        self.assertIsNone(kv_data.cpu_backup_tensor)
        self.assertEqual(weight_data.mapped_bytes, 0)
        self.assertEqual(kv_data.mapped_bytes, 0)

        _log("sleep for 10 seconds to allow manual inspection if needed")
        time.sleep(10)

        _log("wake_up begin")
        self.alloc.wake_up(tags=[weight_tag, kv_tag])
        _log("wake_up done")
        _safe_sync("after wake_up")
        self._dump_pointer_state("after wake_up")

        weight_data = self.alloc.pointer_to_data[weight_ptr]
        kv_data = self.alloc.pointer_to_data[kv_ptr]

        self.assertIsNone(weight_data.cpu_backup_tensor)
        self.assertGreater(weight_data.mapped_bytes, 0)
        self.assertEqual(kv_data.mapped_bytes, kv_expected_mapped)

        _log("test_weight_kvcache_sleep_wakeup_and_free done")

    def test_double_pool_contexts_exist(self) -> None:
        self.assertTrue(hasattr(self.alloc, "use_weight_memory_pool"))
        self.assertTrue(hasattr(self.alloc, "use_kvcache_memory_pool"))
        self.assertTrue(hasattr(self.alloc, "use_memory_pool"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
