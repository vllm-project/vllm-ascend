import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from torch import Tensor
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    get_sparse_kv_offload_cpu_pool_size_bytes,
    plan_sparse_kv_offload_memory,
)
from vllm_ascend.utils import AscendDeviceType, vllm_version_is

if vllm_version_is("0.32.0"):
    from vllm.v1.core import kv_cache_utils as planning
else:
    from vllm.v1.core import kv_cache_planning as planning


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
    def test_selective_sdma_reserves_rank_local_staging(self):
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=256),
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)),
            cache_config=SimpleNamespace(block_size=128),
            speculative_config=None,
        )
        sparse = SimpleNamespace(selective_offload=True, use_fused_copy_sfa=False, topk=2048, topk_buffer_size=2048)
        for rank, expected in ((0, 10_592_384), (1, 9_699_328)):
            with (
                self.subTest(rank=rank),
                patch.object(manager_module, "get_tensor_model_parallel_rank", return_value=rank),
            ):
                self.assertEqual(manager_module.get_selective_sdma_device_memory_bytes(config, sparse), expected)
        sparse.indexer_types = ("full", "shared", "shared", "shared")
        config.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[1, 2, 4])
        for rank, expected in ((0, 10_592_384), (1, 9_699_328)):
            with (
                self.subTest(rank=rank, grouped_descriptors=True),
                patch.object(manager_module, "get_tensor_model_parallel_rank", return_value=rank),
            ):
                lru_order_bytes = 3 * 4 * sparse.topk_buffer_size * 4
                self.assertEqual(
                    manager_module.get_selective_sdma_device_memory_bytes(config, sparse),
                    expected + 1152 + lru_order_bytes,
                )
        sparse.selective_offload = False
        self.assertEqual(manager_module.get_selective_sdma_device_memory_bytes(config, sparse), 0)

    def test_selective_lru_order_budget_uses_hot_capacity_on_every_tp_rank(self):
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=3, max_num_batched_tokens=8),
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(kv_lora_rank=8, qk_rope_head_dim=4)),
            cache_config=SimpleNamespace(block_size=128),
            speculative_config=SimpleNamespace(num_speculative_tokens=1),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
        )
        sparse = SimpleNamespace(
            selective_offload=True,
            use_fused_copy_sfa=False,
            topk=128,
            topk_buffer_size=256,
            indexer_types=("full", "shared", "shared"),
        )
        rows = 6
        for rank in (0, 1):
            with self.subTest(rank=rank), patch.object(manager_module, "get_tensor_model_parallel_rank", return_value=rank):
                sparse.indexer_types = ("full", "full", "full")
                without_shared = manager_module.get_selective_sdma_device_memory_bytes(config, sparse)
                sparse.indexer_types = ("full", "shared", "shared")
                with_shared = manager_module.get_selective_sdma_device_memory_bytes(config, sparse)
                self.assertEqual(with_shared - without_shared, 2 * rows * 256 * 4)

    @patch.object(planning, "_pool_bytes_per_block", return_value=2048)
    @patch.object(planning, "get_kv_cache_groups")
    def test_selective_layout_counts_all_groups_in_the_shared_block_pool(self, groups, pool_bytes):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()
        for names, expected in ((("host.0", "device.0"), 401), (("device.0",), 201)):
            with self.subTest(groups=len(names)):
                groups.return_value = [SimpleNamespace(kv_cache_spec=specs[name]) for name in names]
                budget = plan_sparse_kv_offload_memory(
                    kv_cache_spec=specs,
                    vllm_config=vllm_config,
                    available_device_memory_bytes=1000 * 512,
                    dram_limit_bytes=alignment_reserve + 1000 * 2048,
                    keep_device_kv_cache=False,
                    selective_offload=True,
                )
                self.assertEqual(budget.final_num_blocks, expected)
                self.assertEqual(budget.final_planner_bytes, expected * 2048)
                self.assertEqual(budget.planned_device_bytes, expected * 512)
                self.assertEqual(budget.planned_host_bytes, expected * 2048)

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
        for rank, expected_alloc_size, dp_index, selective in (
            (0, planned_pool_size, 0, False),
            (1, 0, 0, False),
            (0, planned_pool_size, 1, False),
            (1, 0, 1, False),
            (0, planned_pool_size, 0, True),
        ):
            with self.subTest(rank=rank, dp_index=dp_index):
                offload_config.selective_offload = selective
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
                    patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()) as sparse_ops,
                ):
                    SparseKVOffloadManager(
                        vllm_config,
                        kv_cache_config,
                        offload_config,
                    )
                    self.assertEqual(sparse_ops.call_count, 0 if selective else 1)

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
    def test_selective_registration_keeps_only_shared_main_kv(self):
        names = [f"model.layers.{i}.self_attn" for i in range(7)]
        shared = set(names[3:6])
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager.num_target_layers = 7
        manager.tp_rank = 0
        manager.sparse_kv_offload_config = SimpleNamespace(
            selective_offload=True,
            is_host_layer=lambda name: name in shared,
        )
        manager.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=[name], kv_cache_spec=SimpleNamespace(store_on_host=name in shared))
                for name in names
            ]
        )
        caches = dict.fromkeys(names, object())
        caches[names[0] + ".indexer"] = object()

        manager._register_offload_layers(caches)

        self.assertEqual(manager.offload_layer_names, names[3:6])
        self.assertEqual(manager.num_layers, 3)
        self.assertEqual(manager.mtp_layer_id, -1)
        self.assertEqual(manager._get_offload_layer_id(names[5]), 2)
        with self.assertRaises(KeyError):
            manager._get_offload_layer_id(names[0])
        with self.assertRaises(KeyError):
            manager.offload_new_kv(names[0], None, None, None, None, None, None, None)
        del caches[names[4]]
        with self.assertRaisesRegex(ValueError, "tensors are missing"):
            manager._register_offload_layers(caches)
        caches[names[4]] = object()
        manager.kv_cache_config.kv_cache_groups[0].kv_cache_spec.store_on_host = True
        with self.assertRaisesRegex(ValueError, "placement disagrees"):
            manager._register_offload_layers(caches)

    def test_legacy_registration_keeps_all_host_main_layers(self):
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager.num_target_layers = 2
        manager.tp_rank = 0
        manager.sparse_kv_offload_config = SimpleNamespace(selective_offload=False)
        manager.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["model.layers.0.attn", "model.layers.1.attn"],
                    kv_cache_spec=SimpleNamespace(store_on_host=True),
                )
            ]
        )
        manager._register_offload_layers(dict.fromkeys(["model.layers.0.attn", "model.layers.1.attn"], object()))
        self.assertEqual(manager.num_layers, 2)
        self.assertEqual(manager.mtp_layer_id, -1)

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
                    patch.object(manager, "_register_offload_layers"),
                    patch.object(manager, "_bind_copy_sfa_bases") as bind_bases,
                    patch.object(manager, "_init_fused_overlap_planner") as init_planner,
                    patch.object(
                        manager,
                        "_restore_bfloat16_tensor",
                        side_effect=lambda ptr, shape: torch.empty(shape, dtype=torch.bfloat16),
                    ),
                ):
                    manager.register_kv_caches({"layer.0": cache})

                # SfaRemoteD2H consumes these strides on every TP rank.
                self.assertEqual(manager.cpu_block_lens, [(1024, 512)])
                bind_bases.assert_called_once()
                init_planner.assert_not_called()


class TestSelectiveLruState(unittest.TestCase):
    def test_each_shared_layer_allocates_independent_order_on_each_rank(self):
        torch = manager_module.torch
        original_zeros, original_tensor = torch.zeros, torch.tensor
        names = ["model.layers.1.attn", "model.layers.2.attn"]
        caches = {
            name: (
                torch.empty(0),
                torch.empty(0),
                torch.empty((3, 128, 1, 4), dtype=torch.bfloat16),
                torch.empty((3, 128, 1, 2), dtype=torch.bfloat16),
                torch.empty((2, 384, 1, 4), dtype=torch.bfloat16),
                torch.empty((2, 384, 1, 2), dtype=torch.bfloat16),
            )
            for name in names
        }

        def cpu_zeros(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_zeros(*args, **kwargs)

        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)

        for rank in (0, 1):
            with self.subTest(rank=rank):
                manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
                manager.tp_rank = rank
                manager.tp_size = 2
                manager.tp_group = SimpleNamespace(broadcast=MagicMock())
                manager.num_layers = len(names)
                manager.offload_layer_names = names
                manager.kv_cache_config = SimpleNamespace(num_blocks=3)
                manager.sparse_kv_offload_config = SimpleNamespace(
                    selective_offload=True, indexer_types=("full", "shared", "shared")
                )
                manager.topk = 4
                manager.topk_buffer_size = 128
                manager.block_size = 128
                manager.max_num_tokens = 2
                manager.use_fused_copy_sfa = False
                manager.use_fused_overlap = False
                with (
                    patch.object(torch, "zeros", side_effect=cpu_zeros),
                    patch.object(torch, "tensor", side_effect=cpu_tensor),
                    patch.object(manager, "_register_offload_layers"),
                    patch.object(manager_module, "HostGatherCoordinator"),
                    patch.object(manager_module, "SelectiveSDMATransfer"),
                    patch.object(manager_module.torch_npu.npu, "Stream"),
                    patch.object(manager_module.torch_npu.npu, "Event"),
                ):
                    manager.register_kv_caches(caches)

                self.assertEqual(len(manager.selective_lru_order), 2)
                for order in manager.selective_lru_order:
                    self.assertEqual(tuple(order.shape), (2, 128))
                    self.assertEqual(order.dtype, torch.int32)
                    self.assertEqual(order.device, caches[names[0]][4].device)
                self.assertNotEqual(manager.selective_lru_order[0].data_ptr(), manager.selective_lru_order[1].data_ptr())
                order_ptrs = [order.data_ptr() for order in manager.selective_lru_order]
                for keys, requests in zip(manager.selective_hot_keys, manager.selective_hot_requests):
                    keys.fill_(7)
                    requests.fill_(11)
                manager.reset_selective_hot_cache()
                self.assertTrue(all(torch.all(keys == -1) for keys in manager.selective_hot_keys))
                self.assertTrue(all(torch.all(requests == -1) for requests in manager.selective_hot_requests))
                self.assertEqual([order.data_ptr() for order in manager.selective_lru_order], order_ptrs)

    def test_group_descriptor_references_and_retains_each_layer_lru_order(self):
        torch = manager_module.torch
        original_tensor = torch.tensor
        rows, hot, topk = 2, 128, 4
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager.topk = topk
        manager.topk_buffer_size = hot
        manager.block_size = 128
        manager.selective_row_stride = 384
        manager.token_size_bytes_k = 8
        manager.token_size_bytes_v = 4
        manager.kv_cache_config = SimpleNamespace(num_blocks=3)
        manager.selective_hot_keys = [torch.full((rows, hot + 1), -1, dtype=torch.int64) for _ in range(2)]
        manager.selective_hot_requests = [torch.full((rows, 1), -1, dtype=torch.int64) for _ in range(2)]
        manager.selective_lru_order = [torch.empty((rows, hot), dtype=torch.int32) for _ in range(2)]
        manager.selective_physical = torch.empty((2, rows, 1, topk), dtype=torch.int32)
        manager.selective_copy_src = torch.empty((2, 2 * rows * topk), dtype=torch.int64)
        manager.selective_copy_dst = torch.empty_like(manager.selective_copy_src)
        manager.selective_copy_lengths = torch.empty((2, 2 * rows * topk), dtype=torch.int32)
        manager.gvas_k_bases = [1000, 2000]
        manager.gvas_v_bases = [3000, 4000]
        manager.topk_buffers_k = [torch.empty(16), torch.empty(16)]
        manager.topk_buffers_v = [torch.empty(16), torch.empty(16)]
        manager.selective_lim_capture_buffers = []
        manager.selective_prefetch_ready = [SimpleNamespace(record=MagicMock()) for _ in range(2)]
        manager.prefetch_selective_kv = MagicMock()
        metadata = SimpleNamespace(
            block_table=torch.zeros((rows, 2), dtype=torch.int32),
            req_ids_tensor=torch.ones(rows, dtype=torch.int64),
            selective_tail_rows=torch.zeros((rows, 2), dtype=torch.int32),
        )
        seq_lengths = torch.full((rows,), 128, dtype=torch.int32)
        captured = []

        def cpu_tensor(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return original_tensor(*args, **kwargs)

        def capture_descriptor(_topk, descriptor, *_args):
            captured.append(descriptor.clone())

        with (
            patch.object(torch, "tensor", side_effect=cpu_tensor),
            patch.object(manager_module.torch_npu.npu, "is_current_stream_capturing", return_value=True),
            patch.object(
                torch.ops._C_ascend, "npu_precomputed_topk_manage", side_effect=capture_descriptor, create=True
            ),
        ):
            manager.prepare_selective_kv(
                [(0, metadata, seq_lengths), (1, metadata, seq_lengths)],
                torch.zeros((rows, topk), dtype=torch.int64),
            )

        self.assertEqual(tuple(captured[0].shape), (2, 16))
        self.assertEqual(captured[0][:, 15].tolist(), [order.data_ptr() for order in manager.selective_lru_order])
        self.assertEqual(len(manager.selective_lim_capture_buffers), 1)
        _host, _device, retained_inputs = manager.selective_lim_capture_buffers[0]
        for layer, order in enumerate(manager.selective_lru_order):
            self.assertTrue(any(tensor is order for tensor in retained_inputs[layer]))
