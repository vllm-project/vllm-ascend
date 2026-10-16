# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.worker.dsv4_dcp_memory import plan_dsv4_dcp_memory, profile_dsv4_dcp_topk_memory


@pytest.mark.parametrize(
    "available,pool_block,gather_block,max_gather,decode",
    [
        (10_000, 100, 25, 1000, 500),
        (10_000, 100, 25, 4, 500),
        (10_000, 100, 25, 4, 2000),
        (100, 100, 1, 1, 100),
        (50, 100, 1, 1, 100),
        (31 * 812416, 812416, 9 * (32768 + 4160), 25, 500_000),
    ],
)
def test_memory_plan_fits_pool_and_workspace_at_the_largest_block_count(
    available, pool_block, gather_block, max_gather, decode
):
    budget, workspace = plan_dsv4_dcp_memory(available, pool_block, gather_block, max_gather, decode)
    num_blocks = budget // pool_block
    expected = max(
        (
            n
            for n in range(available // pool_block + 1)
            if n * pool_block + max(decode, min(n, max_gather) * gather_block) <= available
        ),
        default=0,
    )
    assert num_blocks == expected
    assert budget == expected * pool_block
    assert workspace == max(decode, min(expected, max_gather) * gather_block)


def test_profile_topk_measures_merge_peak_separately_from_activation_peak():
    fake_npu = MagicMock()
    fake_npu.memory_allocated.return_value = 1000
    fake_npu.max_memory_allocated.return_value = 1800
    from vllm_ascend.attention.context_parallel import dsv4_indexer

    with (
        patch.object(torch, "npu", fake_npu, create=True),
        patch.object(dsv4_indexer, "merge_dcp_topk", wraps=dsv4_indexer.merge_dcp_topk) as merge,
    ):
        reserve = profile_dsv4_dcp_topk_memory(3, 4, 8, 32, torch.device("cpu"))

    assert reserve == 800
    fake_npu.reset_peak_memory_stats.assert_called_once_with(torch.device("cpu"))
    assert merge.call_args.args[0].shape == (24, 1, 4)
    assert merge.call_args.args[1].dtype == torch.float32


def _worker(model_type="deepseek_v4", dcp_size=8, pcp_size=1):
    from vllm_ascend.worker.worker import NPUWorker

    worker = NPUWorker.__new__(NPUWorker)
    worker.device = torch.device("cpu")
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=512),
            max_model_len=131072,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=pcp_size,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=12, max_num_batched_tokens=4096),
    )
    worker.get_kv_cache_spec = MagicMock()
    return worker


@pytest.mark.parametrize(
    "model_type,dcp_size,pcp_size",
    [("deepseek_v4", 1, 1), ("deepseek_v4", 8, 2), ("deepseek_v41", 8, 1)],
)
def test_workspace_budget_does_not_change_other_paths(model_type, dcp_size, pcp_size):
    worker = _worker(model_type, dcp_size, pcp_size)
    assert worker._apply_dsv4_dcp_workspace_budget(12345) == 12345
    worker.get_kv_cache_spec.assert_not_called()


def test_workspace_budget_uses_pool_geometry_null_block_and_single_layer_peak():
    worker = _worker()
    indexer_spec = SimpleNamespace(
        model_version="deepseek_v4",
        tokens_per_state=4,
        scale_dim=1,
        dtype=torch.int8,
        block_size=128,
        page_size_bytes=4160,
    )
    main_spec = SimpleNamespace(
        model_version="deepseek_v4",
        tokens_per_state=4,
        scale_dim=0,
        dtype=torch.bfloat16,
        block_size=128,
        page_size_bytes=32768,
    )
    worker.get_kv_cache_spec.return_value = {
        **{f"indexer.{i}": indexer_spec for i in range(21)},
        **{f"main.{i}": main_spec for i in range(21)},
    }
    with (
        patch("vllm_ascend.worker.worker.get_kv_cache_groups", return_value=[]),
        patch("vllm_ascend.patch.platform.patch_kv_cache_utils._ascend_pool_bytes_per_block", return_value=812416),
        patch("vllm_ascend.worker.worker.profile_dsv4_dcp_topk_memory", return_value=500_000) as profile,
        patch("vllm_ascend.worker.worker.plan_dsv4_dcp_memory", wraps=plan_dsv4_dcp_memory) as plan,
    ):
        budget = worker._apply_dsv4_dcp_workspace_budget(10 * 1024**3)

    profile.assert_called_once_with(12, 512, 8, 32, torch.device("cpu"))
    # ceil(131072 / (128 * 8)) = 128 blocks per request, plus one null ID.
    plan.assert_called_once_with(10 * 1024**3, 812416, 9 * (32768 + 4160), 1 + 12 * 128, 500_000)
    assert budget % 812416 == 0
    assert worker._dsv4_dcp_kv_cache_memory_bytes == budget


def test_workspace_budget_keeps_a5_indexer_layout_unchanged():
    worker = _worker()
    worker.get_kv_cache_spec.return_value = {
        "indexer": SimpleNamespace(
            model_version="deepseek_v4", tokens_per_state=4, scale_dim=1, dtype=torch.float8_e4m3fn
        )
    }
    assert worker._apply_dsv4_dcp_workspace_budget(12345) == 12345


@pytest.mark.parametrize("dcp_size,legacy_plan", [(1, False), (8, False), (8, True)])
def test_startup_plan_preserves_dcp_budget_and_other_paths(tmp_path, dcp_size, legacy_plan):
    import vllm.envs as upstream_envs
    import vllm.v1.worker.startup_plan as startup_plan

    warmup_module = importlib.import_module("vllm_ascend.model_executor.warmup.kernel_warmup")
    before_workspace = 10 << 30
    indexer_spec = SimpleNamespace(
        model_version="deepseek_v4",
        tokens_per_state=4,
        scale_dim=1,
        dtype=torch.int8,
        block_size=128,
        page_size_bytes=4160,
    )
    main_spec = SimpleNamespace(
        model_version="deepseek_v4",
        tokens_per_state=4,
        scale_dim=0,
        page_size_bytes=32768,
    )

    def make_worker():
        worker = _worker(dcp_size=dcp_size)
        worker.rank = 0
        worker.parallel_config = worker.vllm_config.parallel_config
        worker.parallel_config.world_size = 8
        worker.model_config = worker.vllm_config.model_config
        worker.model_config.enforce_eager = True
        worker.model_config.seed = 0
        worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=None, gpu_memory_utilization=0.9)
        worker.vllm_config.cache_config = worker.cache_config
        worker.vllm_config.kv_transfer_config = None
        worker.vllm_config.compilation_config = SimpleNamespace(compile_sizes=[], compilation_time=0.0)
        worker.model_runner = MagicMock()
        worker.model_runner.model_memory_usage = 2 << 30
        worker.requested_memory = before_workspace + (2 << 30) + (1 << 30) + (512 << 20)
        worker.init_snapshot = SimpleNamespace(free_memory=16 << 30, total_memory=24 << 30)
        worker._kvpp_cache_allocation_plan = None
        worker._scale_kv_cache_memory_for_multi_group = MagicMock(side_effect=lambda value: value)
        worker._apply_kv_offload_decode_memory_constraints = MagicMock(side_effect=lambda value: value)
        worker.get_kv_cache_spec.return_value = {"main": main_spec, "indexer": indexer_spec}
        return worker

    profile_result = SimpleNamespace(
        weights_memory=2 << 30,
        non_torch_increase=512 << 20,
        before_profile=SimpleNamespace(torch_peak=0),
        after_profile=SimpleNamespace(free_memory=15 << 30),
    )
    plan_path = tmp_path / "startup_plan.json"
    worker = make_worker()
    with (
        patch.object(upstream_envs, "VLLM_ENABLE_STARTUP_PLAN", True),
        patch.object(startup_plan, "compute_plan_fingerprint", return_value="dcp-test"),
        patch.object(startup_plan, "_plan_path", return_value=str(plan_path)),
        patch("vllm_ascend.worker.worker.memory_profiling") as profiling,
        patch("vllm_ascend.worker.worker.torch.npu.memory_stats", return_value={"allocated_bytes.all.peak": 1 << 30}),
        patch("vllm_ascend.worker.worker.torch.npu.memory_reserved", return_value=0),
        patch("vllm_ascend.worker.worker.torch.npu.memory_allocated", return_value=0),
        patch("vllm_ascend.worker.worker.get_kv_cache_groups", return_value=[]),
        patch("vllm_ascend.patch.platform.patch_kv_cache_utils._ascend_pool_bytes_per_block", return_value=812416),
        patch("vllm_ascend.worker.worker.profile_dsv4_dcp_topk_memory", return_value=500_000) as topk_profile,
        patch.object(warmup_module, "kernel_warmup"),
        patch("vllm_ascend.worker.worker.get_ascend_config", return_value=SimpleNamespace(enable_cpu_binding=False)),
        patch("vllm_ascend.worker.worker.get_current_hardware_profile") as hardware,
        patch("vllm_ascend.worker.worker.set_random_seed"),
        patch("vllm_ascend.worker.worker.logger") as logger,
    ):
        profiling.return_value.__enter__.return_value = profile_result
        hardware.return_value.supports.return_value = False
        first_budget = worker.determine_available_memory()
        worker.compile_or_warm_up_model()
        saved_bytes = json.loads(plan_path.read_text())["kv_cache_memory_bytes"]

        if dcp_size > 1:
            assert saved_bytes == first_budget
            assert saved_bytes < before_workspace - (150 << 20)
            topk_profile.assert_called_once()
            suggestion = next(
                call.args[0] for call in logger.info.call_args_list if "--kv-cache-memory=" in call.args[0]
            )
            assert suggestion.count(f"--kv-cache-memory={first_budget}") == 2
        else:
            # The ordinary requested/free-memory suggestions retain their
            # existing 150 MiB margin and may differ from the profiled budget.
            assert saved_bytes == before_workspace - (150 << 20)
            topk_profile.assert_not_called()

        profiling.reset_mock()
        if legacy_plan:
            plan = json.loads(plan_path.read_text())
            plan["kv_cache_memory_bytes"] = before_workspace - (150 << 20)
            plan_path.write_text(json.dumps(plan))
        reused_worker = make_worker()
        with patch.object(
            reused_worker, "_apply_dsv4_dcp_workspace_budget", wraps=reused_worker._apply_dsv4_dcp_workspace_budget
        ) as reserve:
            reused_budget = reused_worker.determine_available_memory()

        if dcp_size > 1:
            # Both legacy and corrected plans share the upstream fingerprint.
            # Always reprofile DCP, without trusting either cached value.
            assert reused_worker.cache_config.kv_cache_memory_bytes is None
            assert reused_budget == first_budget
            reserve.assert_called_once_with(before_workspace)
            profiling.assert_called_once()
        else:
            assert reused_worker.cache_config.kv_cache_memory_bytes == saved_bytes
            assert reused_budget == saved_bytes
            reserve.assert_not_called()
            profiling.assert_not_called()
        reused_worker.model_runner.profile_run.assert_called_once_with()


def test_explicit_dcp_kv_memory_does_not_deduct_workspace_twice():
    import vllm.envs as upstream_envs

    worker = _worker()
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=12345)
    worker.init_snapshot = SimpleNamespace(free_memory=16 << 30)
    worker.model_runner = MagicMock()
    worker._kvpp_cache_allocation_plan = None
    worker._apply_kv_offload_decode_memory_constraints = MagicMock(side_effect=lambda value: value)
    with (
        patch.object(upstream_envs, "VLLM_ENABLE_STARTUP_PLAN", True),
        patch("vllm_ascend.worker.worker.maybe_apply_startup_plan") as apply_plan,
        patch.object(worker, "_apply_dsv4_dcp_workspace_budget") as reserve,
    ):
        assert worker.determine_available_memory() == 12345

    apply_plan.assert_called_once_with(worker)
    reserve.assert_not_called()
    worker.model_runner.profile_run.assert_called_once_with()
