from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from vllm.config import AttentionConfig, CUDAGraphMode, KVTransferConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from tests.ut.base import TestBase
from vllm_ascend.utils import vllm_version_is


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_triton_warmup_follows_dummy_runs_before_capture(enforce_eager):
    from vllm_ascend.worker.worker import NPUWorker

    events: list[tuple[str] | tuple[str, int]] = []
    compilation_config = SimpleNamespace(
        compile_sizes=[2, 4],
        cudagraph_mode=CUDAGraphMode.NONE,
        get_compile_ranges=lambda: [],
        compilation_time=0.0,
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(compilation_config=compilation_config),
        model_config=SimpleNamespace(enforce_eager=enforce_eager, seed=0),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=1),
        model_runner=SimpleNamespace(
            _dummy_run=lambda size: events.append(("dummy", size)),
            capture_model=lambda: events.append(("capture",)),
        ),
    )
    with (
        patch(
            "vllm_ascend.model_executor.warmup.kernel_warmup.kernel_warmup",
            side_effect=lambda worker: events.append(("kernels",)),
        ),
        patch("vllm_ascend.worker.worker.get_current_hardware_profile") as profile,
        patch("vllm_ascend.worker.worker.get_ascend_config", return_value=SimpleNamespace(enable_cpu_binding=False)),
        patch("vllm_ascend.worker.worker.set_random_seed"),
    ):
        profile.return_value.supports.return_value = False
        NPUWorker.compile_or_warm_up_model(worker)
    expected = [("dummy", 4), ("dummy", 2), ("kernels",)]
    if not enforce_eager:
        expected.append(("capture",))
    assert events == expected


def test_triton_warmup_failure_prevents_capture():
    from vllm_ascend.worker.worker import NPUWorker

    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                compile_sizes=[2, 4],
                cudagraph_mode=CUDAGraphMode.NONE,
                get_compile_ranges=lambda: [],
            ),
        ),
        model_config=SimpleNamespace(enforce_eager=False),
        model_runner=MagicMock(),
    )
    with (
        patch(
            "vllm_ascend.model_executor.warmup.kernel_warmup.kernel_warmup",
            side_effect=RuntimeError("KDA compilation failed"),
        ),
        pytest.raises(RuntimeError, match="KDA compilation failed"),
    ):
        NPUWorker.compile_or_warm_up_model(worker)
    assert worker.model_runner._dummy_run.call_args_list == [call(4), call(2)]
    worker.model_runner.capture_model.assert_not_called()


class TestNPUWorkerV2(TestBase):
    def test_worker_imports_planning_helpers_from_supported_lane(self):
        from vllm.v1.core import kv_cache_planning as planning
        from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder

        from vllm_ascend.worker import worker

        self.assertIs(worker.get_kv_cache_config_builder, get_kv_cache_config_builder)
        self.assertIs(worker.unify_hybrid_kv_cache_specs, planning.unify_hybrid_kv_cache_specs)

    def test_get_encoder_cudagraph_stats_is_serializable(self):
        from vllm_ascend.worker.worker import NPUWorker

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            manager = MagicMock()
            manager.is_captured.return_value = True
            manager.get_cumulative_stats.return_value = {
                "graph_hits": 4,
                "graph_misses": 0,
                "hit_rate": 1.0,
                "num_budgets": 1,
                "token_budgets": [2048],
            }
            worker.model_runner = SimpleNamespace(
                model_state=SimpleNamespace(
                    encoder_runner=SimpleNamespace(cudagraph_manager=manager),
                ),
            )

            stats = worker.get_encoder_cudagraph_stats()

        self.assertEqual(
            stats,
            {
                "manager": type(manager).__name__,
                "captured": True,
                "graph_hits": 4,
                "graph_misses": 0,
                "hit_rate": 1.0,
                "num_budgets": 1,
                "token_budgets": [2048],
            },
        )

    @staticmethod
    def _make_worker_for_lifecycle_test():
        from vllm_ascend.worker.worker import NPUWorker

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
        worker._static_kernel_runtime_initialized = False
        worker.model_runner = MagicMock()
        worker.vllm_config = MagicMock()
        worker.vllm_config.model_config.enable_sleep_mode = False
        worker.vllm_config.weight_transfer_config = None
        return worker

    def test_init_device_selects_mm_encoder_runner_under_cuda_wrapper(self):
        from vllm_ascend.worker.worker import NPUWorker

        device = object()
        runner = MagicMock()
        events = []

        @contextmanager
        def wrapper():
            events.append("wrapper_enter")
            yield
            events.append("wrapper_exit")

        def make_runner(vllm_config, runner_device):
            events.append("runner")
            return runner

        with (
            patch.object(NPUWorker, "__init__", lambda self, **kwargs: None),
            patch.object(NPUWorker, "_init_device", return_value=device),
            patch(
                "vllm_ascend.worker.worker.get_ascend_config",
                return_value=SimpleNamespace(
                    ascend_compilation_config=SimpleNamespace(
                        enable_static_kernel=False, enable_super_kernel=False, enable_npugraph_ex=False
                    )
                ),
            ),
            patch("vllm_ascend.worker.worker.init_workspace_manager"),
            patch("vllm_ascend.worker.worker.report_usage_stats") as report_usage_stats,
            patch(
                "vllm_ascend.worker.v2.utils.torch_cuda_wrapper",
                side_effect=wrapper,
            ) as torch_cuda_wrapper,
            patch(
                "vllm.v1.worker.mm_encoder_model_runner.MMEncoderModelRunner",
                side_effect=make_runner,
            ) as mm_encoder_runner,
        ):
            worker = NPUWorker()
            worker.vllm_config = SimpleNamespace(is_mm_encoder_only=True)
            worker.rank = 1

            worker.init_device()

        self.assertEqual(events, ["wrapper_enter", "runner", "wrapper_exit"])
        torch_cuda_wrapper.assert_called_once_with()
        mm_encoder_runner.assert_called_once_with(worker.vllm_config, device)
        self.assertIs(worker.model_runner, runner)
        report_usage_stats.assert_not_called()

    @patch("vllm_ascend.worker.worker.get_ec_transfer")
    @patch("vllm_ascend.worker.worker.has_ec_transfer", return_value=True)
    @patch("vllm_ascend.worker.worker.shared_expert_tp_enable", return_value=False)
    def test_load_model_starts_ec_worker_services(
        self,
        _mock_shared_tp,
        mock_has_ec_transfer,
        mock_get_ec_transfer,
    ):
        """Start MRV2 EC services only after model loading completes."""
        events = []
        connector = mock_get_ec_transfer.return_value
        connector.start_worker_services.side_effect = lambda: events.append("ec")
        worker = self._make_worker_for_lifecycle_test()
        worker.model_runner.load_model.side_effect = lambda: events.append("model")

        worker.load_model()

        self.assertEqual(events, ["model", "ec"])
        mock_has_ec_transfer.assert_called_once_with()
        mock_get_ec_transfer.assert_called_once_with()
        connector.start_worker_services.assert_called_once_with()

    @patch("vllm_ascend.worker.worker.get_ec_transfer")
    @patch("vllm_ascend.worker.worker.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.worker.shared_expert_tp_enable", return_value=False)
    def test_load_model_skips_ec_worker_services_when_disabled(
        self,
        _mock_shared_tp,
        mock_has_ec_transfer,
        mock_get_ec_transfer,
    ):
        worker = self._make_worker_for_lifecycle_test()

        worker.load_model()

        mock_has_ec_transfer.assert_called_once_with()
        mock_get_ec_transfer.assert_not_called()

    @patch("vllm_ascend.worker.worker.ensure_ec_transfer_shutdown")
    @patch("vllm_ascend.worker.worker.ensure_kv_transfer_shutdown")
    def test_shutdown_releases_ec_connector(
        self,
        mock_kv_shutdown,
        mock_ec_shutdown,
    ):
        worker = self._make_worker_for_lifecycle_test()
        worker.profiler = None

        worker.shutdown()

        mock_kv_shutdown.assert_called_once_with()
        mock_ec_shutdown.assert_called_once_with()

    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.enable_sp", return_value=False)
    @patch("vllm_ascend.worker.worker.get_pp_group")
    @patch("vllm_ascend.worker.worker.get_tp_group")
    def test_execute_model_middle_rank_pp(
        self, mock_get_tp_group, mock_get_pp_group, mock_enable_sp, mock_get_ascend_config
    ):
        """MRV2 PP middle ranks send intermediate tensors and return None."""
        from vllm_ascend.worker.worker import NPUWorker

        mock_ascend_config = MagicMock()
        mock_ascend_config.msmonitor_use_daemon = False
        mock_get_ascend_config.return_value = mock_ascend_config

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.vllm_config = MagicMock()
            worker.vllm_config.parallel_config = MagicMock()
            worker.vllm_config.parallel_config.distributed_executor_backend = "ray"
            worker.profiler = None
            worker._pp_send_work = []

            mock_pp_group = MagicMock()
            mock_pp_group.is_first_rank = False
            mock_pp_group.is_last_rank = False
            mock_pp_group.irecv_tensor_dict.return_value = ({"tensor": "data"}, None, None)
            mock_pp_group.isend_tensor_dict.return_value = []
            mock_get_pp_group.return_value = mock_pp_group

            intermediate_output = IntermediateTensors({"output_tensor": "data"})
            worker.model_runner.execute_model.return_value = intermediate_output

            scheduler_output = MagicMock()
            scheduler_output.total_num_scheduled_tokens = 1

            result = worker.execute_model(scheduler_output)

            mock_pp_group.irecv_tensor_dict.assert_called_once_with(all_gather_group=mock_get_tp_group.return_value)
            worker.model_runner.execute_model.assert_called_once()
            mock_pp_group.isend_tensor_dict.assert_called_once_with(
                intermediate_output.tensors,
                all_gather_group=mock_get_tp_group.return_value,
            )
            self.assertIsNone(result)

    @patch("vllm_ascend.worker.worker.get_ascend_config")
    @patch("vllm_ascend.worker.worker.get_pp_group")
    def test_execute_model_pooling_uses_pool_output(self, mock_get_pp_group, mock_get_ascend_config):
        """MRV2 pooling must finish in execute_model instead of sampling logits."""
        from vllm.v1.outputs import ModelRunnerOutput

        from vllm_ascend.worker.worker import NPUWorker

        mock_ascend_config = MagicMock()
        mock_ascend_config.msmonitor_use_daemon = False
        mock_get_ascend_config.return_value = mock_ascend_config
        mock_get_pp_group.return_value.is_first_rank = True

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            worker.model_runner = MagicMock()
            worker.model_runner.is_pooling_model = True
            worker.vllm_config = MagicMock()
            worker.profiler = None
            worker._pp_send_work = []

            pooling_output = ModelRunnerOutput(req_ids=[])
            worker.model_runner.execute_model.return_value = None
            worker.model_runner.pool.return_value = pooling_output

            scheduler_output = MagicMock()
            scheduler_output.total_num_scheduled_tokens = 1

            result = worker.execute_model(scheduler_output)

        self.assertIs(result, pooling_output)
        worker.model_runner.execute_model.assert_called_once()
        worker.model_runner.pool.assert_called_once_with()


class TestNPUWorkerV2WeightUpdate(TestBase):
    @staticmethod
    def _make_worker():
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        draft = torch.nn.Linear(2, 2)
        worker.weight_transfer_engine = MagicMock(supports_draft_weight_update=True)
        worker._weight_update_active = False
        worker._weight_update_is_draft = False
        worker._incomplete_weight_updates = set()
        worker.use_v2_model_runner = True
        worker.parallel_config = SimpleNamespace(pipeline_parallel_size=1)
        worker.vllm_config = SimpleNamespace(
            weight_transfer_config=SimpleNamespace(backend="hccl"),
            speculative_config=SimpleNamespace(draft_model_config=object(), prefill_context_kv_only=False),
        )
        worker.model_runner = SimpleNamespace(
            get_draft_model=lambda: draft,
            reset_lora_state=MagicMock(),
            on_target_weights_replaced=MagicMock(),
        )
        return worker, draft

    @patch("vllm_ascend.worker.worker.get_ascend_config")
    def test_draft_update_then_target_update(self, ascend_config):
        ascend_config.return_value.weight_nz_mode = 0
        for is_release in (True, False):
            with (
                self.subTest(vllm_version="0.33.0" if is_release else "main"),
                patch("vllm_ascend.worker.worker.vllm_version_is", return_value=is_release),
            ):
                worker, draft = self._make_worker()
                engine = worker.weight_transfer_engine
                worker.start_draft_weight_update()
                assert worker.has_incomplete_weight_update()
                engine.set_weight_update_target.assert_called_once_with(
                    draft, worker.vllm_config.speculative_config.draft_model_config
                )
                worker.update_weights({})
                worker.finish_weight_update()
                self.assertFalse(worker._weight_update_active)
                self.assertFalse(worker._weight_update_is_draft)
                assert not worker.has_incomplete_weight_update()
                worker.model_runner.reset_lora_state.assert_not_called()
                worker.model_runner.on_target_weights_replaced.assert_not_called()
                engine.reset_weight_update_target.assert_called_once_with()

                worker.start_weight_update()
                worker.finish_weight_update()
                engine.update_weights.assert_called_once_with({})
                self.assertEqual(engine.start_weight_update.call_count, 2)
                self.assertEqual(engine.finish_weight_update.call_count, 2)
                engine.reset_weight_update_target.assert_called_once_with()
                if is_release:
                    worker.model_runner.reset_lora_state.assert_called_once_with()
                    worker.model_runner.on_target_weights_replaced.assert_not_called()
                else:
                    worker.model_runner.on_target_weights_replaced.assert_called_once_with()
                    worker.model_runner.reset_lora_state.assert_not_called()

    @patch("vllm_ascend.worker.worker.get_ascend_config")
    def test_update_weights_selects_dp2_tp2_worker_payload(self, ascend_config):
        ascend_config.return_value.weight_nz_mode = 0
        dp_size, tp_size = 2, 2
        payloads = [{"names": [f"worker_{rank}.weight"], "size": rank + 1} for rank in range(dp_size * tp_size)]
        for is_draft in (False, True):
            for dp_index in range(dp_size):
                for tp_rank in range(tp_size):
                    with self.subTest(is_draft=is_draft, dp_index=dp_index, tp_rank=tp_rank):
                        worker, _ = self._make_worker()
                        worker.rank = tp_rank
                        worker.parallel_config = SimpleNamespace(
                            data_parallel_index=dp_index,
                            tensor_parallel_size=tp_size,
                            pipeline_parallel_size=1,
                            world_size=tp_size,
                        )
                        worker.vllm_config.parallel_config = worker.parallel_config
                        if is_draft:
                            worker.start_draft_weight_update()
                        else:
                            worker.start_weight_update()

                        engine = worker.weight_transfer_engine
                        expected_payload = payloads[dp_index * tp_size + tp_rank]
                        worker.update_weights(payloads)
                        engine.update_weights.assert_called_once_with(expected_payload)
                        assert engine.update_weights.call_args.args[0] is expected_payload

                        engine.update_weights.reset_mock()
                        worker.update_weights(expected_payload)
                        engine.update_weights.assert_called_once_with(expected_payload)
                        assert engine.update_weights.call_args.args[0] is expected_payload
                        worker.finish_weight_update()

    @patch("vllm_ascend.worker.worker.get_ascend_config")
    def test_failed_draft_update_restores_routing(self, ascend_config):
        ascend_config.return_value.weight_nz_mode = 0
        for operation in ("start_weight_update", "update_weights", "finish_weight_update"):
            with self.subTest(operation=operation):
                worker, _ = self._make_worker()
                engine = worker.weight_transfer_engine

                def fail_reload(*args, worker=worker):
                    assert worker.has_incomplete_weight_update()
                    raise RuntimeError("reload failed")

                getattr(engine, operation).side_effect = fail_reload
                with self.assertRaisesRegex(RuntimeError, "reload failed"):
                    worker.start_draft_weight_update()
                    worker.update_weights({})
                    worker.finish_weight_update()
                engine.reset_weight_update_target.assert_called_once_with()
                self.assertFalse(worker._weight_update_active)
                self.assertFalse(worker._weight_update_is_draft)
                worker.model_runner.reset_lora_state.assert_not_called()
                worker.model_runner.on_target_weights_replaced.assert_not_called()
                if vllm_version_is("0.33.0"):
                    assert worker.has_incomplete_weight_update()
                else:
                    from vllm.v1.engine.core import EngineCore

                    core = SimpleNamespace(
                        is_sleeping=lambda: False,
                        collective_rpc=lambda method, worker=worker: [getattr(worker, method)()],
                    )
                    assert not EngineCore.is_inference_ready(core)

    def test_draft_capability_follows_engine_support(self):
        for backend, use_v2, engine_support, supported in (
            ("hccl", True, True, True),
            ("npu_ipc", True, True, True),
            ("hccl", False, True, False),
            ("hccl", True, False, False),
            ("npu_ipc", True, False, False),
        ):
            with self.subTest(backend=backend, use_v2=use_v2, engine_support=engine_support):
                worker, _ = self._make_worker()
                worker.vllm_config.weight_transfer_config.backend = backend
                worker.use_v2_model_runner = use_v2
                worker.weight_transfer_engine.supports_draft_weight_update = engine_support
                self.assertEqual(worker.supports_draft_weight_updates(), supported)
                if not supported:
                    with self.assertRaisesRegex(RuntimeError, "supporting weight transfer engine"):
                        worker.start_draft_weight_update()
                    worker.weight_transfer_engine.set_weight_update_target.assert_not_called()

    def test_draft_update_rejects_runner_without_draft_model(self):
        worker, _ = self._make_worker()
        worker.model_runner.get_draft_model = lambda: None
        self.assertFalse(worker.supports_draft_weight_updates())
        with self.assertRaisesRegex(RuntimeError, "configured draft model"):
            worker.start_draft_weight_update()
        worker.weight_transfer_engine.start_weight_update.assert_not_called()

    def test_draft_update_rejects_pipeline_parallelism(self):
        for has_draft in (True, False):
            with self.subTest(has_draft=has_draft):
                worker, _ = self._make_worker()
                worker.parallel_config.pipeline_parallel_size = 2
                if not has_draft:
                    worker.model_runner.get_draft_model = lambda: None
                self.assertFalse(worker.supports_draft_weight_updates())
                with self.assertRaisesRegex(RuntimeError, "without pipeline parallelism"):
                    worker.start_draft_weight_update()
                worker.weight_transfer_engine.start_weight_update.assert_not_called()

    @patch("vllm_ascend.worker.worker.get_ascend_config")
    def test_draft_update_rejects_nz(self, ascend_config):
        ascend_config.return_value.weight_nz_mode = 1
        worker, _ = self._make_worker()
        with self.assertRaises(ValueError):
            worker.start_draft_weight_update()
        worker.weight_transfer_engine.start_weight_update.assert_not_called()


@pytest.mark.parametrize("with_kv_transfer", [False, True])
@pytest.mark.parametrize("use_compress", [False, True])
def test_mrv2_block_outermost_hybrid_budget_respects_shared_backing(with_kv_transfer, use_compress):
    from vllm_ascend.worker.worker import NPUWorker

    compressed_spec = MLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float16,
        tokens_per_state=4,
    )
    mamba_spec = MambaSpec(
        block_size=8,
        shapes=((2, 4),),
        dtypes=(torch.float32,),
    )
    groups = [
        KVCacheGroupSpec(
            layer_names=["compressed_attn"],
            kv_cache_spec=compressed_spec,
        ),
        KVCacheGroupSpec(
            layer_names=["linear_attn"],
            kv_cache_spec=mamba_spec,
        ),
    ]
    worker = NPUWorker.__new__(NPUWorker)
    worker.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen4_exp"))
    worker.vllm_config = SimpleNamespace(
        attention_config=AttentionConfig(),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC),
        kv_transfer_config=(
            KVTransferConfig(kv_connector="MooncakeConnector", kv_role="kv_both") if with_kv_transfer else None
        ),
    )
    worker.get_kv_cache_spec = MagicMock(
        return_value={
            "compressed_attn": compressed_spec,
            "linear_attn": mamba_spec,
        }
    )
    for use_sparse in (False, True):
        worker.model_runner = SimpleNamespace(
            supports_standardized_shared_kv_backing=True,
            use_sparse=use_sparse,
            use_compress=use_compress,
        )
        with patch(
            "vllm_ascend.worker.worker.get_kv_cache_groups",
            return_value=groups,
        ):
            expected = 8230 if use_sparse or use_compress else 12345
            assert worker._scale_kv_cache_memory_for_multi_group(12345) == expected
