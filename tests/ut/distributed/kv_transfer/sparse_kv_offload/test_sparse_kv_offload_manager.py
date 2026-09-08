import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
    HostMemoryRegion,
    HostPoolTopology,
    MooncakeHostPool,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    get_sparse_kv_offload_cpu_pool_size_bytes,
    plan_sparse_kv_offload_memory,
)
from vllm_ascend.utils import AscendDeviceType


class _FakeKVCacheSpec:
    def __init__(
        self,
        *,
        page_size_bytes,
        max_blocks_per_request,
        store_on_host,
        block_size=128,
    ):
        self.page_size_bytes = page_size_bytes
        self.max_blocks_per_request = max_blocks_per_request
        self.store_on_host = store_on_host
        self.block_size = block_size

    def max_memory_usage_bytes(self, _vllm_config):
        return self.max_blocks_per_request * self.page_size_bytes


def _make_memory_plan_inputs(max_num_seqs=2):
    specs = {
        "host.0": _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        ),
        "host.1": _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        ),
        "device.0": _FakeKVCacheSpec(
            page_size_bytes=512,
            max_blocks_per_request=100,
            store_on_host=False,
        ),
    }
    vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs))
    alignment_reserve = 2 * manager_module._CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    return specs, vllm_config, alignment_reserve


def _make_manager_init_inputs(dram_size_per_dp_gb=1):
    spec = _FakeKVCacheSpec(
        page_size_bytes=1024,
        max_blocks_per_request=100,
        store_on_host=True,
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=200,
        kv_cache_groups=[SimpleNamespace(layer_names=["host.0"], kv_cache_spec=spec)],
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_layers=MagicMock(return_value=1),
            max_model_len=128,
        ),
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=1,
        ),
        speculative_config=None,
    )
    offload_config = SimpleNamespace(
        topk_buffer_size=1,
        topk=1,
        use_fused_overlap=False,
        host_backend="memfabric",
        dram_size_per_dp_GB=dram_size_per_dp_gb,
    )
    return vllm_config, kv_cache_config, offload_config


class TestSparseKVOffloadMemoryPlanning(unittest.TestCase):
    def test_non_a3_is_rejected(self):
        with (
            patch.object(manager_module, "_SPARSE_KV_OFFLOAD_MANAGER", None),
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A2),
            self.assertRaisesRegex(RuntimeError, "Sparse KV offload is only support on A3"),
        ):
            manager_module.init_sparse_kv_offload_manager(None, None, None)

    def test_memory_plan_is_limited_by_active_workload(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=1000 * 512,
            dram_limit_bytes=alignment_reserve + 1000 * 2048,
            keep_device_kv_cache=False,
        )

        self.assertEqual(budget.npu_limit_blocks, 1000)
        self.assertEqual(budget.dram_limit_blocks, 1000)
        self.assertEqual(budget.workload_limit_blocks, 201)
        self.assertEqual(budget.final_num_blocks, 201)
        self.assertEqual(budget.final_planner_bytes, 201 * (2048 + 512))
        self.assertEqual(budget.planned_host_bytes, 201 * 2048)
        self.assertEqual(budget.planned_device_bytes, 201 * 512)
        self.assertEqual(budget.limiting_factor, "workload")

    def test_memory_plan_is_limited_by_dram_capacity(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=1000 * 512,
            dram_limit_bytes=alignment_reserve + 150 * 2048,
            keep_device_kv_cache=False,
        )

        self.assertEqual(budget.dram_limit_blocks, 150)
        self.assertEqual(budget.final_num_blocks, 150)
        self.assertEqual(budget.limiting_factor, "dram")

    def test_warning_when_dram_budget_caps_npu_utilization(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        with self.assertLogs(manager_module.logger, level="WARNING") as logs:
            plan_sparse_kv_offload_memory(
                kv_cache_spec=specs,
                vllm_config=vllm_config,
                available_device_memory_bytes=1000 * 512,
                dram_limit_bytes=alignment_reserve + 500 * 2048,
                keep_device_kv_cache=False,
            )
        self.assertTrue(any("dram_size_per_dp_GB" in line for line in logs.output))

    def test_no_warning_when_dram_budget_not_below_npu(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        with patch.object(manager_module.logger, "warning_once") as mock_warn:
            plan_sparse_kv_offload_memory(
                kv_cache_spec=specs,
                vllm_config=vllm_config,
                available_device_memory_bytes=1000 * 512,
                dram_limit_bytes=alignment_reserve + 1000 * 2048,
                keep_device_kv_cache=False,
            )
        mock_warn.assert_not_called()

    def test_keep_device_cache_counts_full_npu_page(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()
        total_page_size = 2048 + 512

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=125 * total_page_size,
            dram_limit_bytes=alignment_reserve + 1000 * 2048,
            keep_device_kv_cache=True,
        )

        self.assertEqual(budget.npu_limit_blocks, 125)
        self.assertEqual(budget.final_num_blocks, 125)
        self.assertEqual(budget.planned_device_bytes, 125 * total_page_size)
        self.assertEqual(budget.limiting_factor, "npu")

    def test_non_positive_capacity_produces_zero_blocks(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        for available_device_memory, dram_limit, limiting_factor in (
            (-1, alignment_reserve + 1000 * 2048, "npu"),
            (1000 * 512, alignment_reserve - 1, "dram"),
        ):
            with self.subTest(limiting_factor=limiting_factor):
                budget = plan_sparse_kv_offload_memory(
                    kv_cache_spec=specs,
                    vllm_config=vllm_config,
                    available_device_memory_bytes=available_device_memory,
                    dram_limit_bytes=dram_limit,
                    keep_device_kv_cache=False,
                )
                self.assertEqual(budget.final_num_blocks, 0)
                self.assertEqual(budget.final_planner_bytes, 0)
                self.assertEqual(budget.limiting_factor, limiting_factor)

    def test_memory_plan_rejects_invalid_spec_layouts(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()
        invalid_cases = (
            ({"device.0": specs["device.0"]}, "at least one host"),
            ({"host.0": specs["host.0"]}, "at least one device"),
            (
                {
                    **specs,
                    "device.1": _FakeKVCacheSpec(
                        page_size_bytes=512,
                        max_blocks_per_request=100,
                        store_on_host=False,
                        block_size=64,
                    ),
                },
                "one shared block size",
            ),
        )

        for invalid_specs, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                plan_sparse_kv_offload_memory(
                    kv_cache_spec=invalid_specs,
                    vllm_config=vllm_config,
                    available_device_memory_bytes=1000 * 512,
                    dram_limit_bytes=alignment_reserve + 1000 * 2048,
                    keep_device_kv_cache=False,
                )

    def test_cpu_pool_size_includes_per_layer_alignment_reserve(self):
        specs, _, alignment_reserve = _make_memory_plan_inputs()
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[name],
                    kv_cache_spec=spec,
                )
                for name, spec in specs.items()
            ],
        )

        self.assertEqual(
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config),
            200 * 2048 + alignment_reserve,
        )

    def test_cpu_pool_size_supports_uniform_specs(self):
        specs, _, alignment_reserve = _make_memory_plan_inputs()
        uniform_specs = UniformTypeKVCacheSpecs(
            block_size=128,
            kv_cache_specs=specs,
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=list(specs),
                    kv_cache_spec=uniform_specs,
                )
            ],
        )

        self.assertEqual(
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config),
            200 * 2048 + alignment_reserve,
        )

    def test_cpu_pool_size_rejects_missing_host_specs(self):
        specs, _, _ = _make_memory_plan_inputs()
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["device.0"],
                    kv_cache_spec=specs["device.0"],
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "host-resident"):
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config)

    def test_manager_initializes_offload_with_planned_pool_size(self):
        vllm_config, kv_cache_config, offload_config = _make_manager_init_inputs()
        planned_pool_size = 4096
        for rank, expected_alloc_size in ((0, planned_pool_size), (1, 0)):
            with self.subTest(rank=rank):
                offload_backend = SimpleNamespace(
                    OffloadConfig=lambda: SimpleNamespace(),
                    Scene=SimpleNamespace(SHARED="shared"),
                    initialize=MagicMock(return_value=0),
                )
                tp_group = SimpleNamespace(barrier=MagicMock())

                with (
                    patch.object(
                        manager_module,
                        "get_tensor_model_parallel_rank",
                        return_value=rank,
                    ),
                    patch.object(
                        manager_module,
                        "get_tensor_model_parallel_world_size",
                        return_value=2,
                    ),
                    patch.object(
                        manager_module,
                        "get_tp_group",
                        return_value=tp_group,
                    ),
                    patch.object(
                        manager_module,
                        "get_sparse_kv_offload_cpu_pool_size_bytes",
                        return_value=planned_pool_size,
                    ),
                    patch.object(
                        manager_module,
                        "offload",
                        offload_backend,
                        create=True,
                    ),
                    patch.object(
                        manager_module.torch,
                        "zeros",
                        return_value=MagicMock(),
                    ),
                    patch.object(
                        manager_module.torch,
                        "empty",
                        return_value=MagicMock(),
                    ),
                    patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
                ):
                    manager = SparseKVOffloadManager(
                        vllm_config,
                        kv_cache_config,
                        offload_config,
                    )
                    manager.prepare_host_kv_allocation(
                        device_id=7,
                        dp_rank=0,
                    )

                initialized_config = offload_backend.initialize.call_args.args[0]
                self.assertEqual(
                    initialized_config.reserve_size,
                    planned_pool_size,
                )
                self.assertEqual(
                    initialized_config.alloc_size,
                    expected_alloc_size,
                )
                self.assertEqual(initialized_config.world_size, 2)
                self.assertEqual(initialized_config.rank_id, rank)
                tp_group.barrier.assert_called_once_with()

    def test_layer_allocation_uses_backend_callback_on_non_owner_rank(self):
        host_k = MagicMock()
        host_v = MagicMock()
        allocate_host = MagicMock(return_value=[host_k, host_v])

        result = manager_module.allocate_kv_cache_tensors_for_sparse_kv_offload(
            k_tensor_size=128,
            v_tensor_size=64,
            alignment=32,
            tp_rank=1,
            keep_device_kv_cache=False,
            npu_kv_cache_allocate_func=MagicMock(),
            host_kv_cache_allocate_func=allocate_host,
        )

        self.assertIs(result[2], host_k)
        self.assertIs(result[3], host_v)
        allocate_host.assert_called_once_with([128, 64], 32)

    def test_manager_prepares_mooncake_host_pool(self):
        vllm_config, kv_cache_config, offload_config = _make_manager_init_inputs()
        offload_config.host_backend = "mooncake"
        planned_pool_size = 4096
        allocator = MagicMock()
        tp_group = SimpleNamespace(barrier=MagicMock())

        with (
            patch.object(manager_module, "get_tensor_model_parallel_rank", return_value=1),
            patch.object(manager_module, "get_tensor_model_parallel_world_size", return_value=2),
            patch.object(manager_module, "get_tp_group", return_value=tp_group),
            patch.object(
                manager_module,
                "get_sparse_kv_offload_cpu_pool_size_bytes",
                return_value=planned_pool_size,
            ),
            patch.object(
                manager_module.MooncakeHostPool,
                "allocate",
                return_value=allocator,
            ) as allocate_pool,
            patch.object(manager_module.torch, "zeros", return_value=MagicMock()),
            patch.object(manager_module.torch, "empty", return_value=MagicMock()),
            patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
        ):
            manager = SparseKVOffloadManager(vllm_config, kv_cache_config, offload_config)
            manager.prepare_host_kv_allocation(device_id=7, dp_rank=3)

        self.assertIs(manager._host_kv_allocator, allocator)
        self.assertTrue(manager._host_allocation_prepared)
        allocate_pool.assert_called_once()
        call = allocate_pool.call_args
        self.assertEqual(call.kwargs["size_bytes"], planned_pool_size)
        self.assertEqual(call.kwargs["alignment"], manager_module._CPU_CACHE_ALIGNMENT)
        topology = call.kwargs["topology"]
        self.assertEqual(topology.tp_rank, 1)
        self.assertEqual(topology.tp_size, 2)
        self.assertEqual(topology.owner_rank, 0)
        self.assertEqual(topology.device_id, 7)
        self.assertEqual(topology.dp_rank, 3)
        self.assertIs(topology.tp_group, tp_group)
        tp_group.barrier.assert_not_called()

    def test_failed_mooncake_prepare_remains_retryable(self):
        manager = object.__new__(SparseKVOffloadManager)
        manager.host_backend = "mooncake"
        manager._host_allocation_prepared = False
        manager._host_kv_allocator = None
        manager.host_pool_size_bytes = 4096
        manager.tp_rank = 0
        manager.tp_size = 1
        manager.tp_group = MagicMock()

        with (
            patch.object(
                manager_module.MooncakeHostPool,
                "allocate",
                side_effect=RuntimeError("allocation failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "allocation failed"),
        ):
            manager.prepare_host_kv_allocation(device_id=0, dp_rank=0)

        self.assertFalse(manager._host_allocation_prepared)
        self.assertIsNone(manager._host_kv_allocator)

    def test_manager_close_releases_host_allocator_once(self):
        manager = object.__new__(SparseKVOffloadManager)
        allocator = MagicMock()
        manager._host_kv_allocator = allocator

        manager.close()
        manager.close()

        allocator.close.assert_called_once_with()
        self.assertIsNone(manager._host_kv_allocator)

    def test_manager_close_retains_host_allocator_when_close_fails(self):
        manager = object.__new__(SparseKVOffloadManager)
        allocator = MagicMock()
        allocator.close.side_effect = [RuntimeError("close failed"), None]
        manager._host_kv_allocator = allocator

        with self.assertRaisesRegex(RuntimeError, "close failed"):
            manager.close()

        self.assertIs(manager._host_kv_allocator, allocator)

        manager.close()

        self.assertIsNone(manager._host_kv_allocator)
        self.assertEqual(allocator.close.call_count, 2)

    def test_manager_rejects_pool_larger_than_dram_limit(self):
        vllm_config, kv_cache_config, offload_config = _make_manager_init_inputs()
        offload_backend = SimpleNamespace(
            OffloadConfig=MagicMock(),
            Scene=SimpleNamespace(SHARED="shared"),
            initialize=MagicMock(return_value=0),
        )

        with (
            patch.object(
                manager_module,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch.object(
                manager_module,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch.object(
                manager_module,
                "get_tp_group",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                manager_module,
                "get_sparse_kv_offload_cpu_pool_size_bytes",
                return_value=(1 << 30) + 1,
            ),
            patch.object(manager_module, "offload", offload_backend, create=True),
            patch.object(manager_module.torch, "zeros", return_value=MagicMock()),
            patch.object(manager_module.torch, "empty", return_value=MagicMock()),
            patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
            self.assertRaisesRegex(ValueError, "exceeds DRAM limit"),
        ):
            SparseKVOffloadManager(
                vllm_config,
                kv_cache_config,
                offload_config,
            )

        offload_backend.OffloadConfig.assert_not_called()
        offload_backend.initialize.assert_not_called()


class TestMemFabricHostKVAllocator(unittest.TestCase):
    def test_owner_rank_delegates_to_empty_aligned_helper(self):
        allocator = manager_module.MemFabricHostKVAllocator(tp_rank=0)
        k, v = MagicMock(), MagicMock()
        with patch.object(manager_module, "empty_aligned_int8_cpu_tensors", return_value=[k, v]) as helper:
            tensors = allocator.allocate_tensors([128, 64], alignment=32)
        helper.assert_called_once_with([128, 64], 32)
        self.assertEqual(tensors, [k, v])

    def test_non_owner_rank_returns_none_views_without_allocation(self):
        allocator = manager_module.MemFabricHostKVAllocator(tp_rank=1)
        with patch.object(manager_module, "empty_aligned_int8_cpu_tensors") as helper:
            tensors = allocator.allocate_tensors([128, 64], alignment=32)
        helper.assert_not_called()
        self.assertEqual(tensors, [None, None])

    def test_close_is_noop_and_idempotent(self):
        allocator = manager_module.MemFabricHostKVAllocator(tp_rank=0)
        allocator.close()
        allocator.close()  # 不抛异常即可；MemFabric 池由 offload 运行时托管


class TestHostBackendSelection(unittest.TestCase):
    def test_manager_prepares_memfabric_host_allocator(self):
        vllm_config, kv_cache_config, offload_config = _make_manager_init_inputs()
        planned_pool_size = 4096
        tp_group = SimpleNamespace(barrier=MagicMock())
        offload_backend = SimpleNamespace(
            OffloadConfig=lambda: SimpleNamespace(),
            Scene=SimpleNamespace(SHARED="shared"),
            initialize=MagicMock(return_value=0),
        )

        with (
            patch.object(manager_module, "get_tensor_model_parallel_rank", return_value=0),
            patch.object(manager_module, "get_tensor_model_parallel_world_size", return_value=2),
            patch.object(manager_module, "get_tp_group", return_value=tp_group),
            patch.object(
                manager_module,
                "get_sparse_kv_offload_cpu_pool_size_bytes",
                return_value=planned_pool_size,
            ),
            patch.object(manager_module, "offload", offload_backend, create=True),
            patch.object(manager_module.torch, "zeros", return_value=MagicMock()),
            patch.object(manager_module.torch, "empty", return_value=MagicMock()),
            patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
        ):
            manager = SparseKVOffloadManager(vllm_config, kv_cache_config, offload_config)
            manager.prepare_host_kv_allocation(device_id=7, dp_rank=0)

        self.assertIsInstance(manager._host_kv_allocator, manager_module.MemFabricHostKVAllocator)
        self.assertTrue(manager._host_allocation_prepared)
        tp_group.barrier.assert_called_once_with()

    def test_failed_memfabric_prepare_remains_retryable(self):
        manager = object.__new__(SparseKVOffloadManager)
        manager.host_backend = "memfabric"
        manager._host_allocation_prepared = False
        manager._host_kv_allocator = None
        manager.host_pool_size_bytes = 4096
        manager.tp_rank = 0
        manager.tp_size = 1
        manager.tp_group = MagicMock()

        offload_backend = SimpleNamespace(
            OffloadConfig=lambda: SimpleNamespace(),
            Scene=SimpleNamespace(SHARED="shared"),
            initialize=MagicMock(return_value=1),  # 触发 assert
        )
        with (
            patch.object(manager_module, "offload", offload_backend, create=True),
            self.assertRaisesRegex(AssertionError, "offload.initialize failed"),
        ):
            manager.prepare_host_kv_allocation(device_id=0, dp_rank=0)

        self.assertFalse(manager._host_allocation_prepared)
        self.assertIsNone(manager._host_kv_allocator)

    def test_prepare_is_idempotent_across_backends(self):
        # mooncake
        manager = object.__new__(SparseKVOffloadManager)
        manager.host_backend = "mooncake"
        manager._host_allocation_prepared = False
        manager._host_kv_allocator = None
        manager.host_pool_size_bytes = 4096
        manager.tp_rank = 0
        manager.tp_size = 1
        manager.tp_group = MagicMock()
        with patch.object(manager_module.MooncakeHostPool, "allocate", return_value=MagicMock()) as allocate_pool:
            manager.prepare_host_kv_allocation(device_id=0, dp_rank=0)
            manager.prepare_host_kv_allocation(device_id=0, dp_rank=0)
        allocate_pool.assert_called_once()

        # memfabric
        manager = object.__new__(SparseKVOffloadManager)
        manager.host_backend = "memfabric"
        manager._host_allocation_prepared = False
        manager._host_kv_allocator = None
        manager.host_pool_size_bytes = 4096
        manager.tp_rank = 0
        manager.tp_size = 1
        manager.tp_group = MagicMock()
        offload_backend = SimpleNamespace(
            OffloadConfig=lambda: SimpleNamespace(),
            Scene=SimpleNamespace(SHARED="shared"),
            initialize=MagicMock(return_value=0),
        )
        with patch.object(manager_module, "offload", offload_backend, create=True):
            manager.prepare_host_kv_allocation(device_id=0, dp_rank=0)
            manager.prepare_host_kv_allocation(device_id=0, dp_rank=0)
        offload_backend.initialize.assert_called_once()

    def test_prepare_rejects_unknown_host_backend(self):
        manager = object.__new__(SparseKVOffloadManager)
        manager.host_backend = "rdma"
        manager._host_allocation_prepared = False
        manager._host_kv_allocator = None
        manager.host_pool_size_bytes = 4096
        manager.tp_rank = 0
        manager.tp_size = 1
        manager.tp_group = MagicMock()

        with (
            patch.object(manager_module.MooncakeHostPool, "allocate") as allocate_pool,
            patch.object(manager_module, "offload", MagicMock(), create=True) as offload_backend,
            self.assertRaisesRegex(ValueError, "Unsupported sparse KV offload Host backend"),
        ):
            manager.prepare_host_kv_allocation(device_id=0, dp_rank=0)

        allocate_pool.assert_not_called()
        offload_backend.initialize.assert_not_called()
        self.assertIsNone(manager._host_kv_allocator)

    def test_allocate_host_kv_tensors_raises_before_prepare(self):
        manager = object.__new__(SparseKVOffloadManager)
        manager._host_kv_allocator = None

        with self.assertRaisesRegex(RuntimeError, "prepare_host_kv_allocation must run"):
            manager.allocate_host_kv_tensors([128, 64], 32)

    def test_allocate_host_kv_tensors_delegates_to_allocator(self):
        mooncake = MagicMock()
        mooncake.allocate_tensors.return_value = [MagicMock()]
        cases = [
            ("mooncake", mooncake),
            ("memfabric_rank0", manager_module.MemFabricHostKVAllocator(0)),
            ("memfabric_rank1", manager_module.MemFabricHostKVAllocator(1)),
        ]
        for name, allocator in cases:
            with self.subTest(name=name):
                manager = object.__new__(SparseKVOffloadManager)
                manager._host_kv_allocator = allocator
                with patch.object(
                    manager_module,
                    "empty_aligned_int8_cpu_tensors",
                    return_value=[MagicMock(), MagicMock()],
                ) as helper:
                    tensors = manager.allocate_host_kv_tensors([128, 64], 32)
                if name == "mooncake":
                    self.assertIs(tensors, mooncake.allocate_tensors.return_value)
                    helper.assert_not_called()
                elif name == "memfabric_rank0":
                    self.assertIs(tensors, helper.return_value)
                else:
                    self.assertEqual(tensors, [None, None])
                    helper.assert_not_called()

    def test_close_releases_mooncake_region_via_manager(self):
        release_callback = MagicMock()
        pool = MooncakeHostPool(
            HostMemoryRegion(
                torch.empty(64, dtype=torch.int8),
                handle="segment",
                release_callback=release_callback,
            ),
            HostPoolTopology(tp_rank=0, tp_size=1),
        )
        manager = object.__new__(SparseKVOffloadManager)
        manager._host_kv_allocator = pool

        manager.close()

        release_callback.assert_called_once_with("segment")
        self.assertIsNone(manager._host_kv_allocator)

    def test_close_memfabric_allocator_is_noop(self):
        manager = object.__new__(SparseKVOffloadManager)
        manager._host_kv_allocator = manager_module.MemFabricHostKVAllocator(0)

        manager.close()

        self.assertIsNone(manager._host_kv_allocator)

    def test_layer_allocation_none_cpu_on_memfabric_non_owner(self):
        with patch.object(manager_module, "empty_aligned_int8_cpu_tensors") as helper:
            result = manager_module.allocate_kv_cache_tensors_for_sparse_kv_offload(
                k_tensor_size=128,
                v_tensor_size=64,
                alignment=32,
                tp_rank=1,
                keep_device_kv_cache=False,
                npu_kv_cache_allocate_func=MagicMock(),
                host_kv_cache_allocate_func=manager_module.MemFabricHostKVAllocator(1).allocate_tensors,
            )
        helper.assert_not_called()
        self.assertIsNone(result[2])
        self.assertIsNone(result[3])

    def test_layer_allocation_cpu_on_memfabric_owner(self):
        k_cpu, v_cpu = MagicMock(), MagicMock()
        with patch.object(manager_module, "empty_aligned_int8_cpu_tensors", return_value=[k_cpu, v_cpu]) as helper:
            result = manager_module.allocate_kv_cache_tensors_for_sparse_kv_offload(
                k_tensor_size=128,
                v_tensor_size=64,
                alignment=32,
                tp_rank=0,
                keep_device_kv_cache=False,
                npu_kv_cache_allocate_func=MagicMock(),
                host_kv_cache_allocate_func=manager_module.MemFabricHostKVAllocator(0).allocate_tensors,
            )
        helper.assert_called_once_with([128, 64], 32)
        self.assertIs(result[2], k_cpu)
        self.assertIs(result[3], v_cpu)

    def test_both_allocators_satisfy_hostkvallocator_contract(self):
        for cls in (
            manager_module.MooncakeHostPool,
            manager_module.MemFabricHostKVAllocator,
        ):
            with self.subTest(cls=cls):
                self.assertTrue(hasattr(cls, "allocate_tensors"))
                self.assertTrue(hasattr(cls, "close"))


def _shape_aware_tensor_fake(*args, **kwargs):
    """最小 torch.Tensor 替身：切片/视图/形状断言全部自洽。"""
    fake = MagicMock()
    shape = args[0] if args else kwargs.get("shape", [])
    if isinstance(shape, int):  # torch.arange(n, ...) 场景
        shape = [shape]
    fake.shape = torch.Size(shape)
    fake._itemsize = getattr(kwargs.get("dtype", torch.int8), "itemsize", 1)

    def _view(*view_args):
        # .view(dtype): 按元素大小换算第一维；.view(shape...): 直接复用自身。
        dtype = view_args[0] if view_args else None
        itemsize = getattr(dtype, "itemsize", 0)
        if not itemsize:
            return fake
        new_shape = list(fake.shape)
        if new_shape:
            new_shape[0] = new_shape[0] * fake._itemsize // itemsize
        return _shape_aware_tensor_fake(new_shape, dtype=dtype)

    def _getitem(key):
        if not isinstance(key, slice):
            return fake
        start = key.start or 0
        stop = key.stop if key.stop is not None else (fake.shape[0] if fake.shape else 0)
        new_shape = list(fake.shape)
        if new_shape:
            new_shape[0] = stop - start
        return _shape_aware_tensor_fake(new_shape, dtype=fake.dtype)

    fake.view.side_effect = _view
    fake.__getitem__.side_effect = _getitem
    fake.repeat.return_value = fake
    fake.pin_memory.return_value = fake
    fake.copy_.return_value = fake
    fake.item.return_value = 0xFEED
    fake.data_ptr.return_value = 0x1000
    fake.numel.return_value = 4096
    fake.element_size.return_value = 1
    fake.size.side_effect = lambda dim: fake.shape[dim]
    fake.dtype = kwargs.get("dtype", torch.int8)
    fake.device = kwargs.get("device", "cpu")
    return fake


class TestRegisterKvCachesBackendBranch(unittest.TestCase):
    @staticmethod
    def _make_register_caches_manager(host_backend, k_cpu, v_cpu):
        topk = _shape_aware_tensor_fake([1, 1])  # size(-2)==1, size(-1)==1
        topk.dtype = torch.bfloat16
        topk.device = "npu"
        kv_caches = {"host.0": (MagicMock(), MagicMock(), k_cpu, v_cpu, topk, topk)}
        manager = object.__new__(SparseKVOffloadManager)
        manager.host_backend = host_backend
        manager._host_kv_allocator = None  # 分支只看配置，与 allocator 实例无关
        manager._register_offload_layers = MagicMock()
        manager.offload_layer_names = ["host.0"]
        manager.num_layers = 1
        manager.tp_size = 2
        manager.kv_cache_config = SimpleNamespace(num_blocks=8)
        manager.block_size = 128
        manager.topk_buffer_size = 256
        manager.topk = 8
        manager.max_num_tokens = 4
        manager.max_num_topk_rows = 4
        manager.max_model_len = 1024
        manager.use_fused_overlap = False
        manager.tp_rank = 1
        manager.tp_group = MagicMock()
        return manager, kv_caches

    @staticmethod
    def _patch_torch_factories() -> contextlib.ExitStack:
        stack = contextlib.ExitStack()
        for factory in ("zeros", "empty", "full", "arange"):
            stack.enter_context(
                patch.object(
                    manager_module.torch,
                    factory,
                    side_effect=_shape_aware_tensor_fake,
                )
            )
        return stack

    def test_register_kv_caches_uses_local_views_with_mooncake_backend(self):
        k_cpu = _shape_aware_tensor_fake([64])
        v_cpu = _shape_aware_tensor_fake([64])
        manager, kv_caches = self._make_register_caches_manager("mooncake", k_cpu, v_cpu)

        with self._patch_torch_factories():
            manager.register_kv_caches(kv_caches)

        manager.tp_group.broadcast.assert_not_called()
        self.assertEqual(len(manager.k_caches_cpu), 1)
        self.assertEqual(manager.gvas_k_bases, [k_cpu.data_ptr()])
        self.assertEqual(manager.gvas_v_bases, [v_cpu.data_ptr()])

    def test_register_kv_caches_broadcasts_with_memfabric_backend(self):
        manager, kv_caches = self._make_register_caches_manager("memfabric", None, None)

        with self._patch_torch_factories():
            manager.register_kv_caches(kv_caches)

        self.assertEqual(manager.tp_group.broadcast.call_count, 5)
        self.assertEqual(manager.k_caches_cpu, [])
        self.assertEqual(len(manager.gvas_k_bases), 1)


if __name__ == "__main__":
    unittest.main()
