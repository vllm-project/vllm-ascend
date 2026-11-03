from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode, KVTransferConfig, get_current_vllm_config
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from tests.ut.base import TestBase
from vllm_ascend.utils import vllm_version_is


class TestNPUWorkerV2(TestBase):
    def test_worker_imports_planning_helpers_from_supported_lane(self):
        from vllm_ascend.utils import vllm_version_is
        from vllm_ascend.worker import worker

        if vllm_version_is("0.32.0"):
            from vllm.v1.core import kv_cache_utils as planning
        else:
            from vllm.v1.core import kv_cache_planning as planning

        self.assertIs(worker.get_kv_cache_groups, planning.get_kv_cache_groups)
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
        worker.use_v2_model_runner = True
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
            worker.use_v2_model_runner = True
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
    def test_load_model_starts_ec_worker_services(
        self,
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
    def test_load_model_skips_ec_worker_services_when_disabled(
        self,
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
            worker.use_v2_model_runner = True
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
            worker.use_v2_model_runner = True
            worker.profiler = None
            worker._pp_send_work = []

            pooling_output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
            worker.model_runner.execute_model.return_value = None
            worker.model_runner.pool.return_value = pooling_output

            scheduler_output = MagicMock()
            scheduler_output.total_num_scheduled_tokens = 1

            result = worker.execute_model(scheduler_output)

        self.assertIs(result, pooling_output)
        worker.model_runner.execute_model.assert_called_once()
        worker.model_runner.pool.assert_called_once_with()

    @patch("vllm_ascend.worker.worker.torch.npu.synchronize")
    @patch("vllm_ascend.worker.worker.get_pp_group")
    def test_profile_prefill_latency_v2_delegates_cpp_state(
        self,
        mock_get_pp_group,
        mock_synchronize,
    ):
        from vllm_ascend.worker.worker import NPUWorker

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            worker.use_v2_model_runner = True
            worker.scheduler_config = SimpleNamespace(max_num_batched_tokens=1024)

            model_runner = MagicMock()
            model_runner.max_num_reqs = 8
            worker.model_runner = model_runner

            mock_get_pp_group.return_value.is_first_rank = True

            with patch("time.perf_counter", side_effect=[10.0, 10.25]):
                latency_ms = worker.profile_prefill_latency(512)

            self.assertEqual(latency_ms, 250.0)
            model_runner._dummy_run.assert_called_once_with(
                num_tokens=512,
                force_attention=True,
                is_profile=False,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                profile_cpp=True,
            )
            self.assertEqual(mock_synchronize.call_count, 2)

    @patch("vllm_ascend.worker.worker.torch.npu.synchronize")
    @patch("vllm_ascend.worker.worker.get_pp_group")
    def test_profile_prefill_latency_v2_propagates_dummy_error(
        self,
        mock_get_pp_group,
        mock_synchronize,
    ):
        from vllm_ascend.worker.worker import NPUWorker

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            worker.use_v2_model_runner = True
            worker.scheduler_config = SimpleNamespace(max_num_batched_tokens=1024)

            model_runner = MagicMock()
            model_runner.max_num_reqs = 8
            model_runner._dummy_run.side_effect = RuntimeError("dummy failure")
            worker.model_runner = model_runner

            mock_get_pp_group.return_value.is_first_rank = True

            with self.assertRaisesRegex(RuntimeError, "dummy failure"):
                worker.profile_prefill_latency(512)

            self.assertEqual(mock_synchronize.call_count, 1)

    def test_sample_tokens_v2_attaches_execution_time(self):
        from vllm_ascend.worker.worker import NPUWorker

        with patch.object(NPUWorker, "__init__", lambda self, **kwargs: None):
            worker = NPUWorker()
            worker.use_v2_model_runner = True

            profiling_config = SimpleNamespace(need_timing=True)
            model_runner = MagicMock()
            model_runner.ascend_config = SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    profiling_chunk_config=profiling_config,
                )
            )
            model_runner._cpp_execution_time_ms = 125.0
            model_runner_output = SimpleNamespace()
            output = SimpleNamespace(model_runner_output=model_runner_output)
            model_runner.sample_tokens.return_value = output
            worker.model_runner = model_runner

            grammar_output = SimpleNamespace()
            result = worker.sample_tokens(grammar_output)

            self.assertIs(result, output)
            self.assertEqual(model_runner_output.execution_time_ms, 125.0)
            self.assertIsNone(model_runner._cpp_execution_time_ms)
            model_runner.sample_tokens.assert_called_once_with(grammar_output)


class TestNPUWorkerV2WeightUpdate(TestBase):
    @staticmethod
    def _make_worker():
        from vllm_ascend.worker.worker import NPUWorker

        worker = NPUWorker.__new__(NPUWorker)
        draft = torch.nn.Linear(2, 2)
        worker.use_v2_model_runner = True
        worker.weight_transfer_engine = MagicMock()
        worker.weight_transfer_engine.supports_draft_weight_update = True
        worker._weight_update_active = False
        worker._weight_update_is_draft = False
        worker.vllm_config = SimpleNamespace()
        worker.speculative_config = SimpleNamespace(draft_model_config=object())
        speculator = SimpleNamespace(model=draft)
        worker.model_runner = SimpleNamespace(
            speculator=speculator,
            get_draft_model=lambda: speculator.model,
            reset_lora_state=MagicMock(),
        )
        return worker, draft

    @pytest.mark.skipif(vllm_version_is("0.32.0"), reason="Online-quantized LM-head copies are main-only")
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    def test_draft_session_selects_speculator_and_restores_quantized_heads(self, ascend_config):
        ascend_config.return_value.weight_nz_mode = 0
        worker, draft = self._make_worker()
        engine = worker.weight_transfer_engine
        engine.start_weight_update.side_effect = lambda: self.assertIs(get_current_vllm_config(), worker.vllm_config)
        with (
            patch("vllm_ascend.worker.worker.share_source_lm_heads") as share,
            patch("vllm_ascend.worker.worker.restore_quantized_lm_heads") as restore,
        ):
            worker.start_draft_weight_update()
            self.assertTrue(worker._weight_update_active)
            self.assertTrue(worker._weight_update_is_draft)
            engine.set_weight_update_target.assert_called_once_with(draft, worker.speculative_config.draft_model_config)
            worker.finish_weight_update()
        share.assert_called_once_with(draft)
        restore.assert_called_once_with(draft)
        engine.reset_weight_update_target.assert_called_once_with()
        worker.model_runner.reset_lora_state.assert_not_called()
        self.assertFalse(worker._weight_update_active)
        self.assertFalse(worker._weight_update_is_draft)

    @pytest.mark.skipif(vllm_version_is("0.32.0"), reason="Online-quantized LM-head copies are main-only")
    @patch("vllm_ascend.worker.worker.get_ascend_config")
    def test_draft_failures_restore_quantized_head_references(self, ascend_config):
        ascend_config.return_value.weight_nz_mode = 0
        for stage in ("start_weight_update", "update_weights", "finish_weight_update"):
            with self.subTest(stage=stage):
                worker, draft = self._make_worker()
                engine = worker.weight_transfer_engine
                getattr(engine, stage).side_effect = RuntimeError("reload failed")
                with (
                    patch("vllm_ascend.worker.worker.share_source_lm_heads"),
                    patch("vllm_ascend.worker.worker.restore_quantized_lm_heads") as restore,
                ):
                    if stage != "start_weight_update":
                        worker.start_draft_weight_update()
                    with self.assertRaisesRegex(RuntimeError, "reload failed"):
                        if stage == "start_weight_update":
                            worker.start_draft_weight_update()
                        elif stage == "update_weights":
                            worker.update_weights({})
                        else:
                            worker.finish_weight_update()
                restore.assert_called_once_with(draft, refresh=False)
                engine.reset_weight_update_target.assert_called_once_with()
                self.assertFalse(worker._weight_update_active)
                self.assertFalse(worker._weight_update_is_draft)

    @patch("vllm_ascend.worker.worker.get_ascend_config")
    def test_entrypoints_reject_nz_and_missing_engine(self, ascend_config):
        for name in ("start_weight_update", "start_draft_weight_update"):
            for configured in (False, True):
                with self.subTest(entrypoint=name, configured=configured):
                    worker, _ = self._make_worker()
                    engine = worker.weight_transfer_engine
                    if not configured:
                        worker.weight_transfer_engine = None
                    ascend_config.return_value.weight_nz_mode = int(configured)
                    error = ValueError if configured else RuntimeError
                    with self.assertRaises(error):
                        getattr(worker, name)()
                    engine.start_weight_update.assert_not_called()
                    self.assertFalse(worker._weight_update_active)


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
    worker.use_v2_model_runner = True
    worker.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen4_exp"))
    worker.vllm_config = SimpleNamespace(
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


def test_worker_imports_lm_head_helpers_only_on_main():
    from vllm_ascend.worker import worker

    if vllm_version_is("0.32.0"):
        assert not hasattr(worker, "share_source_lm_heads")
        assert not hasattr(worker, "restore_quantized_lm_heads")
    else:
        from vllm.model_executor.layers.quantization.online import lm_head

        assert worker.share_source_lm_heads is lm_head.share_source_lm_heads
        assert worker.restore_quantized_lm_heads is lm_head.restore_quantized_lm_heads


@pytest.mark.parametrize("failure_stage", [None, "start_weight_update", "update_weights", "finish_weight_update"])
def test_release_draft_lifecycle_does_not_call_main_lm_head_helpers(failure_stage):
    from vllm_ascend.worker import worker as worker_module

    worker, _ = TestNPUWorkerV2WeightUpdate._make_worker()
    engine = worker.weight_transfer_engine
    if failure_stage is not None:
        getattr(engine, failure_stage).side_effect = RuntimeError("reload failed")
    with (
        patch.object(worker_module, "vllm_version_is", return_value=True),
        patch.object(worker_module, "get_ascend_config", return_value=SimpleNamespace(weight_nz_mode=0)),
        patch.object(worker_module, "share_source_lm_heads", create=True) as share,
        patch.object(worker_module, "restore_quantized_lm_heads", create=True) as restore,
    ):

        def update():
            worker.start_draft_weight_update()
            worker.update_weights({})
            worker.finish_weight_update()

        if failure_stage is None:
            update()
        else:
            with pytest.raises(RuntimeError, match="reload failed"):
                update()
        share.assert_not_called()
        restore.assert_not_called()
    engine.reset_weight_update_target.assert_called_once_with()
    assert not worker._weight_update_active
    assert not worker._weight_update_is_draft
