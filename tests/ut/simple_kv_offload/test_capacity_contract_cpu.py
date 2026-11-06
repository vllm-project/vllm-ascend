# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only reproduction of lost simple-offload block-capacity settings.

Run: python3 tests/ut/simple_kv_offload/test_capacity_contract_cpu.py
The base worker methods are copied verbatim from pinned upstream commit
24420b7a341af37a9bc3186e6fc6062512e0aa58 and release tag v0.32.0 (Apache-2.0).
Tensor view construction,
allocation and distributed reduction boundaries are mocked; production
configuration forwarding and integer capacity calculations execute unchanged.
"""

import __future__

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "vllm_ascend").is_dir())
SIMPLE = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/kv_offload/simple"
UPSTREAM_WORKER = r"""
class SimpleCPUOffloadWorker:
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        kv_offload_backend: str = "cpu",
        disk_path: str | None = None,
        disk_capacity_bytes: int = 0,
        disk_buffer_slots: int = 2,
        use_page_cache: bool = False,
        num_cpu_blocks: int | None = None,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        self._num_cpu_blocks_override = num_cpu_blocks
        self.disk_path = disk_path
        self.disk_capacity_bytes = disk_capacity_bytes
        self.disk_buffer_slots = disk_buffer_slots
        self.use_page_cache = use_page_cache
        self.disk_mode = kv_offload_backend == "disk"

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        self._backend: DmaCopyBackend | DiskBackend | None = None

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        self._load_events: list[tuple[int, torch.Event]] = []
        self._store_events: list[tuple[int, torch.Event]] = []
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Compute-done event recorded before each store; reused across steps
        # (get_finished runs once per step, copy queue is FIFO).
        self._store_compute_done: torch.Event | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        self._completed_store_events: dict[int, int] = {}

    def _resolve_num_cpu_blocks(self, capacity_bytes: int, bytes_per_block: int) -> int:
        local_capacity = capacity_bytes // bytes_per_block
        requested = self._num_cpu_blocks_override
        if requested is None:
            return sync_num_offload_blocks_across_workers(max(1, local_capacity))
        if type(requested) is not int or not 0 < requested <= local_capacity:
            raise ValueError(
                f"num_cpu_blocks={requested!r} must be between 1 and "
                f"local offload capacity {local_capacity}"
            )
        return requested

"""


RELEASE_WORKER = r"""
class SimpleCPUOffloadWorker:
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        kv_offload_backend: str = "cpu",
        disk_path: str | None = None,
        disk_capacity_bytes: int = 0,
        disk_buffer_slots: int = 2,
        use_page_cache: bool = False,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        self.disk_path = disk_path
        self.disk_capacity_bytes = disk_capacity_bytes
        self.disk_buffer_slots = disk_buffer_slots
        self.use_page_cache = use_page_cache
        self.disk_mode = kv_offload_backend == "disk"

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        self._backend: DmaCopyBackend | DiskBackend | None = None

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        self._load_events: list[tuple[int, torch.Event]] = []
        self._store_events: list[tuple[int, torch.Event]] = []
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Compute-done event recorded before each store; reused across steps
        # (get_finished runs once per step, copy queue is FIFO).
        self._store_compute_done: torch.Event | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        self._completed_store_events: dict[int, int] = {}
"""


def load_class(path, name, methods, namespace):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    module = ast.Module(body=[cls], type_ignores=[])
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class SimpleOffloadSizingTest(unittest.TestCase):
    def setUp(self):
        self.sync_capacity = MagicMock(return_value=3)
        torch = SimpleNamespace(zeros=MagicMock(), npu=SimpleNamespace(Stream=MagicMock()))
        self.namespace = {
            "torch": torch,
            "logger": MagicMock(),
            "NPUDmaCopyBackend": MagicMock(),
            "sync_num_offload_blocks_across_workers": self.sync_capacity,
            "is_pin_memory_available": lambda: False,
            "_flatten_kv_value": lambda tensor: [tensor],
        }
        self.use_upstream_fixture(UPSTREAM_WORKER)
        hardware = patch.dict(sys.modules, {"torch_npu": MagicMock(), "acl": MagicMock()})
        hardware.start()
        self.addCleanup(hardware.stop)

    def use_upstream_fixture(self, fixture):
        exec(
            compile(fixture, "upstream-simple-worker", "exec", flags=__future__.annotations.compiler_flag),
            self.namespace,
        )
        self.worker_cls = load_class(
            SIMPLE / "worker.py", "SimpleCPUOffloadNPUWorker", {"__init__", "register_kv_caches"}, self.namespace
        )

    def register(self, requested):
        worker = self.worker_cls(None, SimpleNamespace(num_blocks=4), 96)
        # The configuration-forwarding test below separately checks transport
        # of this field. Here isolate the production sizing calculation.
        worker._num_cpu_blocks_override = requested
        tensor = SimpleNamespace(
            device="cpu",
            shape=(4, 6),
            dtype="int8",
            stride=lambda axis: 6,
            element_size=lambda: 1,
            untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 100),
        )
        worker._build_block_views = MagicMock(return_value={"layer": tensor})
        worker.register_kv_caches({"layer": tensor})
        return worker

    def test_explicit_block_count_is_preserved_at_registration(self):
        self.assertEqual(self.register(2).num_cpu_blocks, 2)
        self.sync_capacity.assert_not_called()

    def test_automatic_capacity_uses_all_rank_minimum(self):
        self.assertEqual(self.register(None).num_cpu_blocks, 3)
        self.sync_capacity.assert_called_once_with(16)

    def test_explicit_capacity_cannot_exceed_byte_budget(self):
        with self.assertRaisesRegex(ValueError, "local offload capacity"):
            self.register(17)

    def test_connector_forwards_parsed_block_count_to_replacement(self):
        upstream_handler = SimpleNamespace(cpu_capacity_bytes=96, _num_cpu_blocks_override=2)

        class UpstreamConnector:
            def __init__(self, *args):
                self.worker_handler = upstream_handler

        make_worker = MagicMock()
        role = object()
        connector = load_class(
            SIMPLE / "simple_cpu_offload_connector.py",
            "AscendSimpleCPUOffloadConnector",
            {"__init__"},
            {
                "SimpleCPUOffloadConnector": UpstreamConnector,
                "SimpleCPUOffloadNPUWorker": make_worker,
                "KVConnectorRole": SimpleNamespace(WORKER=role),
                "logger": MagicMock(),
            },
        )
        config, cache = object(), object()
        connector(config, role, cache)
        make_worker.assert_called_once_with(config, cache, 96, num_cpu_blocks=2)

    def test_release_constructor_keeps_legacy_signature(self):
        self.use_upstream_fixture(RELEASE_WORKER)
        worker = self.worker_cls(None, SimpleNamespace(num_blocks=4), 96)
        self.assertEqual(worker.cpu_capacity_bytes, 96)
        self.assertFalse(hasattr(worker, "_num_cpu_blocks_override"))
        self.assertFalse(hasattr(worker, "_resolve_num_cpu_blocks"))

    def test_release_registration_keeps_legacy_local_capacity(self):
        self.use_upstream_fixture(RELEASE_WORKER)
        self.assertEqual(self.register(None).num_cpu_blocks, 16)
        self.sync_capacity.assert_not_called()

    def test_release_connector_accepts_handler_without_override(self):
        self.use_upstream_fixture(RELEASE_WORKER)
        upstream_handler = SimpleNamespace(cpu_capacity_bytes=96)

        class UpstreamConnector:
            def __init__(self, *args):
                self.worker_handler = upstream_handler

        role = object()
        connector = load_class(
            SIMPLE / "simple_cpu_offload_connector.py",
            "AscendSimpleCPUOffloadConnector",
            {"__init__"},
            {
                "SimpleCPUOffloadConnector": UpstreamConnector,
                "SimpleCPUOffloadNPUWorker": self.worker_cls,
                "KVConnectorRole": SimpleNamespace(WORKER=role),
                "logger": MagicMock(),
            },
        )
        result = connector(object(), role, SimpleNamespace(num_blocks=4))
        self.assertIsInstance(result.worker_handler, self.worker_cls)
        self.assertEqual(result.worker_handler.cpu_capacity_bytes, 96)
        self.assertFalse(hasattr(result.worker_handler, "_num_cpu_blocks_override"))


if __name__ == "__main__":
    unittest.main()
