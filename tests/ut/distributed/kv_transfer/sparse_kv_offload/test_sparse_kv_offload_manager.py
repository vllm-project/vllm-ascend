import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import Tensor
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.mooncake_host_pool import (
    HostMemoryRegion,
    HostPoolTopology,
    MooncakeHostPool,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.request_identity import RequestIdentityRegistry
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    get_sparse_kv_offload_cpu_pool_size_bytes,
    plan_sparse_kv_offload_memory,
    resolve_sparse_kv_offload_store_url,
)
from vllm_ascend.utils import AscendDeviceType


class TestSparseKVOffloadRequestMetadata(unittest.TestCase):
    def setUp(self):
        self.identity = RequestIdentityRegistry()
        self.req_ids = SimpleNamespace(np=np.full(4, 99, dtype=np.int64), copy_to_gpu=MagicMock())
        self.token_to_req = SimpleNamespace(np=np.full(8, 99, dtype=np.int32), copy_to_gpu=MagicMock())

    def update(self, ids, *, num_reqs=None, is_dummy=False, query_start=None):
        if num_reqs is None:
            num_reqs = len(ids)
        if query_start is None:
            query_start = np.arange(num_reqs + 1, dtype=np.int32)
        manager_module.update_sparse_kv_offload_metadata(
            int(query_start[-1]),
            num_reqs,
            8,
            4,
            ids,
            query_start,
            self.req_ids,
            self.token_to_req,
            request_identity=self.identity,
            is_dummy=is_dummy,
        )

    def test_colliding_ids_have_distinct_stable_owners_after_batch_reorder(self):
        ids = ["234054828099429543", "471754017250656876", "304399536143223879"]
        self.update(ids)
        owners = dict(zip(ids, self.req_ids.np[:3].tolist()))
        self.assertEqual(len(set(owners.values())), 3)
        self.assertTrue(all(owner > 0 for owner in owners.values()))
        self.update(ids[::-1])
        self.assertEqual(self.req_ids.np.tolist(), [owners[req_id] for req_id in ids[::-1]] + [0])
        self.assertEqual(self.req_ids.np.dtype, np.dtype(np.int64))
        self.assertEqual(self.token_to_req.np.tolist(), [0, 1, 2, 0, 0, 0, 0, 0])
        self.req_ids.copy_to_gpu.assert_called_with(4)
        self.token_to_req.copy_to_gpu.assert_called_with(8)

    def test_owner_change_after_finished_request_reuses_the_same_string(self):
        self.update(["same-id"])
        old_owner = int(self.req_ids.np[0])
        self.identity.remove("same-id")
        self.update(["same-id"])
        self.assertGreater(int(self.req_ids.np[0]), old_owner)

    def test_dummy_does_not_alias_live_owners_or_consume_real_ids(self):
        self.update(["live"])
        owner = int(self.req_ids.np[0])
        self.update([], num_reqs=2, is_dummy=True)
        self.assertEqual(self.req_ids.np.tolist(), [-2, -3, 0, 0])
        self.update(["live", "next"])
        self.assertEqual(self.req_ids.np.tolist(), [owner, owner + 1, 0, 0])

    def test_invalid_or_incomplete_active_ids_are_rejected_before_copy(self):
        for ids in ([], ["only-one"], ["valid", None], ["valid", 123]):
            with self.subTest(ids=ids):
                with self.assertRaises((ValueError, TypeError, RuntimeError)):
                    self.update(ids, num_reqs=2)
                self.req_ids.copy_to_gpu.assert_not_called()
                self.token_to_req.copy_to_gpu.assert_not_called()

    def test_zero_request_batch_publishes_only_padding(self):
        self.update([])
        self.assertEqual(self.req_ids.np.tolist(), [0] * 4)
        self.assertEqual(self.token_to_req.np.tolist(), [0] * 8)

    def test_int64_owner_is_published_without_uint32_truncation(self):
        self.identity._next_owner_id = 1 << 32
        self.update(["large-owner"])
        self.assertEqual(self.req_ids.np.tolist(), [1 << 32, 0, 0, 0])

    def test_int64_wrap_publishes_valid_owners_and_keeps_target_draft_identity(self):
        self.update(["retained"])
        retained_owner = int(self.req_ids.np[0])
        invalidate = MagicMock()
        self.identity._on_wrap = invalidate
        self.identity._next_owner_id = (1 << 63) - 1
        self.update(["last", "overflow", "retained"])
        self.assertEqual(self.req_ids.np.tolist(), [(1 << 63) - 1, 2, retained_owner, 0])
        invalidate.assert_called_once_with()
        self.update(["retained", "overflow", "last"])
        self.assertEqual(self.req_ids.np.tolist(), [retained_owner, 2, (1 << 63) - 1, 0])
        self.assertEqual(self.req_ids.np.dtype, np.dtype(np.int64))


class TestSparseKVOffloadOwnerReset(unittest.TestCase):
    def test_reset_drains_plans_and_invalidates_target_draft_and_fallback_in_place(self):
        target_owners = manager_module.torch.tensor([1, 2], dtype=manager_module.torch.int64)
        draft_owners = manager_module.torch.tensor([3, 4], dtype=manager_module.torch.int64)
        fallback_owners = manager_module.torch.tensor([5, 6], dtype=manager_module.torch.int64)
        buffers = (target_owners, draft_owners, fallback_owners)
        pointers = [buffer.data_ptr() for buffer in buffers]
        manager = SimpleNamespace(
            lru_last_req_ids_cpu_list=[target_owners, draft_owners],
            fused_overlap_plan_owner_layer_id=1,
            fused_overlap_plan_topk=2,
            fused_overlap_plan_num_tokens=2,
            fused_overlap_plan_membership_map=object(),
        )
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "target": SimpleNamespace(impl=SimpleNamespace(fused_overlap_last_req_ids=fallback_owners)),
                    "draft-unallocated": SimpleNamespace(impl=SimpleNamespace(fused_overlap_last_req_ids=None)),
                    "other": object(),
                }
            )
        )
        snapshots = []
        with (
            patch.object(manager_module, "get_sparse_kv_offload_manager", return_value=manager),
            patch.object(
                manager_module.torch_npu.npu,
                "synchronize",
                side_effect=lambda: snapshots.append([buffer.tolist() for buffer in buffers]),
            ),
        ):
            manager_module.reset_sparse_kv_offload_request_owners(config)

        self.assertEqual(snapshots, [[[1, 2], [3, 4], [5, 6]], [[-1, -1], [-1, -1], [-1, -1]]])
        self.assertEqual([buffer.data_ptr() for buffer in buffers], pointers)
        self.assertIsNone(manager.fused_overlap_plan_owner_layer_id)
        self.assertIsNone(manager.fused_overlap_plan_topk)
        self.assertEqual(manager.fused_overlap_plan_num_tokens, 0)
        self.assertIsNone(manager.fused_overlap_plan_membership_map)


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
        # vLLM main's UniformTypeKVCacheSpecs reads this from each member spec.
        self.dcp_sharded = False

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


def _make_manager_init_inputs(dram_size_per_dp_gb=1, store_port_base=8500):
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
        parallel_config=SimpleNamespace(data_parallel_index=0),
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
        use_fused_copy_sfa=False,
        host_backend="memfabric",
        dram_size_per_dp_GB=dram_size_per_dp_gb,
        store_port_base=store_port_base,
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

        self.assertEqual(
            get_sparse_kv_offload_cpu_pool_size_bytes(
                kv_cache_config,
                include_mooncake_writeback_sinks=True,
            ),
            200 * 2048 + alignment_reserve + 2 * (1024 // 128),
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
        for rank, expected_alloc_size, dp_index, store_port_base in (
            (0, planned_pool_size, 0, 8500),
            (1, 0, 0, 8500),
            (0, planned_pool_size, 1, 8500),
            (1, 0, 1, 8500),
            (0, planned_pool_size, 0, 9500),
            (1, 0, 0, 9500),
        ):
            with self.subTest(rank=rank, dp_index=dp_index, store_port_base=store_port_base):
                vllm_config.parallel_config.data_parallel_index = dp_index
                offload_config.store_port_base = store_port_base
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
                    offload_backend.initialize.assert_not_called()
                    manager.prepare_host_kv_allocation(device_id=7, dp_rank=dp_index)

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
                self.assertEqual(
                    initialized_config.store_url,
                    f"tcp://127.0.0.1:{offload_config.store_port_base + dp_index}",
                )
                tp_group.barrier.assert_called_once_with()

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

    def test_manager_close_drops_membership_tensor_references(self):
        manager = object.__new__(SparseKVOffloadManager)
        membership = MagicMock()
        planner = MagicMock()
        device_staging = MagicMock()
        manager.fused_overlap_membership_map = membership
        manager.fused_overlap_planner_membership_map = planner
        manager.fused_overlap_membership_plan_device_staging = device_staging
        manager.fused_plan_broadcast_buffer = MagicMock()
        manager.fused_plan_metadata_npu = MagicMock()
        manager.fused_plan_status_npu = MagicMock()
        manager.fused_plan_current_linear_slots_npu = MagicMock()
        manager._host_kv_allocator = None

        manager.close()

        self.assertIsNone(manager.fused_overlap_membership_map)
        self.assertIsNone(manager.fused_overlap_planner_membership_map)
        self.assertIsNone(manager.fused_overlap_membership_plan_device_staging)
        self.assertIsNone(manager.fused_plan_broadcast_buffer)
        self.assertIsNone(manager.fused_plan_metadata_npu)
        self.assertIsNone(manager.fused_plan_status_npu)
        self.assertIsNone(manager.fused_plan_current_linear_slots_npu)


if __name__ == "__main__":
    unittest.main()


class TestHostKVRegistration(unittest.TestCase):
    def test_topk_cache_components_exclude_only_registered_scratch_views(self):
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager.offload_layer_names = ["target", "draft"]

        self.assertEqual(
            manager.get_topk_cache_components(),
            {("target", 4), ("target", 5), ("draft", 4), ("draft", 5)},
        )
        manager.offload_layer_names = ["other"]
        self.assertEqual(manager.get_topk_cache_components(), {("other", 4), ("other", 5)})

    def test_copy_sfa_exports_block_byte_strides_on_all_tp_ranks(self):
        torch = manager_module.torch
        original_zeros = torch.zeros
        original_tensor = torch.tensor
        broadcasts: list[Tensor] = []
        for rank in (0, 1):
            with self.subTest(rank=rank):
                manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
                manager.tp_rank = rank
                manager.host_backend = "memfabric"
                manager.num_layers = 1
                manager.offload_layer_names = ["layer.0"]
                manager.kv_cache_config = SimpleNamespace(num_blocks=3)
                manager.topk_buffer_size = 256
                manager.block_size = 128
                manager.max_num_tokens = 2
                manager.use_fused_copy_sfa = True
                manager.use_fused_overlap = False
                manager._register_offload_layers = MagicMock()
                manager._bind_copy_sfa_bases = MagicMock()
                manager._init_fused_overlap_planner = MagicMock()
                manager._restore_bfloat16_tensor = MagicMock(
                    side_effect=lambda ptr, shape: torch.empty(shape, dtype=torch.bfloat16)
                )
                pending = iter(broadcasts)

                def broadcast(tensor, src, rank=rank, pending=pending):
                    if rank == 0:
                        broadcasts.append(tensor.clone())
                    else:
                        tensor.copy_(next(pending))

                manager.tp_group = SimpleNamespace(broadcast=broadcast)
                cache = (
                    torch.empty(0),
                    torch.empty(0),
                    torch.empty((3, 128, 1, 4), dtype=torch.bfloat16),
                    torch.empty((3, 128, 1, 2), dtype=torch.bfloat16),
                    torch.empty((2, 256, 1, 4), dtype=torch.bfloat16),
                    torch.empty((2, 256, 1, 2), dtype=torch.bfloat16),
                )

                def cpu_zeros(*args, **kwargs):
                    kwargs["device"] = "cpu"
                    return original_zeros(*args, **kwargs)

                def cpu_tensor(*args, **kwargs):
                    kwargs["device"] = "cpu"
                    return original_tensor(*args, **kwargs)

                with (
                    patch.object(torch, "zeros", side_effect=cpu_zeros),
                    patch.object(torch, "tensor", side_effect=cpu_tensor),
                ):
                    manager.register_kv_caches({"layer.0": cache})

                # SfaRemoteD2H consumes these strides on every TP rank.
                self.assertEqual(manager.cpu_block_lens, [(1024, 512)])
                manager._bind_copy_sfa_bases.assert_called_once()
                manager._init_fused_overlap_planner.assert_not_called()


class TestSparseKVOffloadStorePort(unittest.TestCase):
    def test_default_base_keeps_the_historical_endpoint(self):
        config = SimpleNamespace(store_port_base=8500)

        self.assertEqual(
            resolve_sparse_kv_offload_store_url(config, 0),
            "tcp://127.0.0.1:8500",
        )
        self.assertEqual(
            resolve_sparse_kv_offload_store_url(config, 3),
            "tcp://127.0.0.1:8503",
        )

    def test_independent_services_can_use_distinct_bases(self):
        service_a = SimpleNamespace(store_port_base=8500)
        service_b = SimpleNamespace(store_port_base=9500)

        # Two services in the same network namespace may reuse the same DP
        # index, so the configured base has to keep their endpoints apart.
        self.assertEqual(
            resolve_sparse_kv_offload_store_url(service_a, 0),
            "tcp://127.0.0.1:8500",
        )
        self.assertEqual(
            resolve_sparse_kv_offload_store_url(service_b, 0),
            "tcp://127.0.0.1:9500",
        )

    def test_tp_ranks_of_one_group_share_the_endpoint(self):
        config = SimpleNamespace(store_port_base=9000)

        # The endpoint only depends on the DP index, so every TP rank of the
        # same offload group agrees on it.
        endpoints = {resolve_sparse_kv_offload_store_url(config, 1) for _ in range(4)}

        self.assertEqual(endpoints, {"tcp://127.0.0.1:9001"})

    def test_port_out_of_range_is_rejected(self):
        config = SimpleNamespace(store_port_base=65535)

        with self.assertRaisesRegex(ValueError, "valid TCP port"):
            resolve_sparse_kv_offload_store_url(config, 1)


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
        manager.sparse_kv_offload_config = SimpleNamespace(store_port_base=8500)
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
        manager.sparse_kv_offload_config = SimpleNamespace(store_port_base=8500)
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
        cases = [
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
                if name == "memfabric_rank0":
                    self.assertIs(tensors, helper.return_value)
                else:
                    self.assertEqual(tensors, [None, None])
                    helper.assert_not_called()

    def test_mooncake_allocation_adds_one_private_sink_row(self):
        pool = MooncakeHostPool(
            HostMemoryRegion(torch.empty(128, dtype=torch.int8), handle=None),
            HostPoolTopology(tp_rank=0, tp_size=1),
        )
        manager = object.__new__(SparseKVOffloadManager)
        manager._host_kv_allocator = pool
        manager.kv_cache_config = SimpleNamespace(num_blocks=2)
        manager.block_size = 4
        manager._mooncake_writeback_raw_views = {}

        logical_k_raw, logical_v_raw = manager.allocate_host_kv_tensors(
            [32, 16],
            alignment=1,
        )

        self.assertEqual(logical_k_raw.numel(), 32)
        self.assertEqual(logical_v_raw.numel(), 16)
        physical_k = manager._mooncake_writeback_raw_views[logical_k_raw.data_ptr()]
        physical_v = manager._mooncake_writeback_raw_views[logical_v_raw.data_ptr()]
        self.assertEqual(physical_k.numel(), 36)
        self.assertEqual(physical_v.numel(), 18)

        logical_k = logical_k_raw.view(torch.bfloat16).view(8, 2)
        logical_v = logical_v_raw.view(torch.bfloat16).view(8, 1)
        self.assertEqual(
            manager._get_mooncake_writeback_view(logical_k, 2).shape,
            (9, 2),
        )
        self.assertEqual(
            manager._get_mooncake_writeback_view(logical_v, 1).shape,
            (9, 1),
        )

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
        manager.sparse_kv_offload_config = SimpleNamespace(lru_max_threads=8)
        manager._host_kv_allocator = None
        if host_backend == "mooncake":
            manager._host_kv_allocator = object.__new__(MooncakeHostPool)
            manager._get_mooncake_writeback_view = MagicMock()
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
        manager.use_fused_copy_sfa = False
        manager._restore_bfloat16_tensor = MagicMock(return_value=MagicMock())
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
        self.assertEqual(manager._get_mooncake_writeback_view.call_count, 2)

    def test_register_kv_caches_broadcasts_with_memfabric_backend(self):
        manager, kv_caches = self._make_register_caches_manager("memfabric", None, None)

        with self._patch_torch_factories():
            manager.register_kv_caches(kv_caches)

        self.assertEqual(manager.tp_group.broadcast.call_count, 5)
        self.assertEqual(len(manager.k_caches_cpu), 1)
        self.assertEqual(manager._restore_bfloat16_tensor.call_count, 2)
        self.assertEqual(len(manager.gvas_k_bases), 1)


class TestLRUPlannerThreadBudget(unittest.TestCase):
    @staticmethod
    def _make_planner(configured):
        manager = object.__new__(SparseKVOffloadManager)
        manager.sparse_kv_offload_config = SimpleNamespace(lru_max_threads=configured)
        manager.max_num_topk_rows = 2
        manager.topk = 4
        manager.topk_buffer_size = 8
        manager.max_model_len = 16
        manager.num_layers = 1
        manager._warmup_external_lru_planner_threads = MagicMock()
        return manager

    def test_workspace_budget_is_limited_by_affinity_and_at_least_one(self):
        for configured, cpus, expected in ((16, 4, 4), (2, 8, 2), (8, 0, 1)):
            with self.subTest(configured=configured, cpus=cpus):
                manager = self._make_planner(configured)
                with (
                    patch.object(manager_module.os, "sched_getaffinity", return_value=set(range(cpus))),
                    TestRegisterKvCachesBackendBranch._patch_torch_factories(),
                ):
                    manager._init_fused_overlap_planner()
                self.assertEqual(manager.lru_workspace_threads, expected)
                manager._warmup_external_lru_planner_threads.assert_called_once_with()

    def test_workspace_budget_falls_back_to_cpu_count(self):
        for error in (AttributeError, OSError):
            for cpu_count, expected in ((3, 3), (None, 1)):
                with self.subTest(error=error, cpu_count=cpu_count):
                    manager = self._make_planner(8)
                    with (
                        patch.object(manager_module.os, "sched_getaffinity", side_effect=error),
                        patch.object(manager_module.os, "cpu_count", return_value=cpu_count),
                        TestRegisterKvCachesBackendBranch._patch_torch_factories(),
                    ):
                        manager._init_fused_overlap_planner()
                    self.assertEqual(manager.lru_workspace_threads, expected)
                    manager._warmup_external_lru_planner_threads.assert_called_once_with()
