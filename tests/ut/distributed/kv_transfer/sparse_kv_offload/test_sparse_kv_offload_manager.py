import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from torch import Tensor
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.request_identity import RequestIdentityRegistry
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    get_sparse_kv_offload_cpu_pool_size_bytes,
    plan_sparse_kv_offload_memory,
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

    def test_owner_exhaustion_does_not_publish_partial_metadata(self):
        self.identity._next_owner_id = (1 << 63) - 1
        with self.assertRaises(OverflowError):
            self.update(["last", "overflow"])
        self.req_ids.copy_to_gpu.assert_not_called()
        self.token_to_req.copy_to_gpu.assert_not_called()

    def test_v1_mtp_dummy_uses_same_disjoint_domain_as_target(self):
        self.req_ids.gpu = self.req_ids.np
        self.token_to_req.gpu = self.token_to_req.np
        config = SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=True))
        with patch.object(manager_module, "get_ascend_config", return_value=config):
            req_ids, token_to_req = manager_module.prepare_sparse_kv_offload_mtp_dummy_metadata(
                3,
                2,
                manager_module.torch.tensor([0, 1, 3]),
                self.req_ids,
                self.token_to_req,
            )

        self.assertEqual(req_ids.tolist(), [-2, -3])
        self.assertEqual(token_to_req.tolist(), [0, 1, 1])
        self.req_ids.copy_to_gpu.assert_called_once_with(2)
        self.token_to_req.copy_to_gpu.assert_called_once_with(3)


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

    def _make_manager_init_inputs(self, dram_size_per_dp_gb=1):
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
            dram_size_per_dp_GB=dram_size_per_dp_gb,
        )
        return vllm_config, kv_cache_config, offload_config

    def test_manager_initializes_offload_with_planned_pool_size(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        planned_pool_size = 4096
        for rank, expected_alloc_size, dp_index in (
            (0, planned_pool_size, 0),
            (1, 0, 0),
            (0, planned_pool_size, 1),
            (1, 0, 1),
        ):
            with self.subTest(rank=rank, dp_index=dp_index):
                vllm_config.parallel_config.data_parallel_index = dp_index
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
                    SparseKVOffloadManager(
                        vllm_config,
                        kv_cache_config,
                        offload_config,
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
                self.assertEqual(
                    initialized_config.store_url,
                    f"tcp://127.0.0.1:{manager_module.OFFLOAD_STORE_PORT_BASE + dp_index}",
                )
                tp_group.barrier.assert_called_once_with()

    def test_manager_rejects_pool_larger_than_dram_limit(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
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
