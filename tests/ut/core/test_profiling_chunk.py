#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from contextlib import ExitStack
from unittest.mock import MagicMock, PropertyMock, patch

import torch
from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import ProfilingChunkConfig, clear_ascend_config, init_ascend_config
from vllm_ascend.core.profiling_chunk_predictor import ChunkSizePredictor, ProfilingChunkManager
from vllm_ascend.core.scheduler_profiling_chunk import ProfilingChunkScheduler
from vllm_ascend.core.short_request_first_scheduler import (
    ShortRequestFirstRequestQueue,
)
from vllm_ascend.utils import vllm_version_is

MODEL = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 16
MAX_NUM_BATCHED_TOKENS = 8192
MAX_NUM_SEQS = 16


def create_requests(num_requests, num_tokens=10, max_tokens=16, request_id_prefix=""):
    init_none_hash(sha256)
    sampling_params = SamplingParams(ignore_eos=False, max_tokens=max_tokens)
    requests = []
    for i in range(num_requests):
        request = Request(
            request_id=f"{request_id_prefix}{i}",
            prompt_token_ids=[i] * num_tokens,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
        )
        requests.append(request)
    return requests


def make_output(scheduler):
    req_ids = [req.request_id for req in scheduler.running]
    req_id_to_index = {req.request_id: i for i, req in enumerate(scheduler.running)}
    sampled_token_ids = [[1000]] * len(scheduler.running)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


# ===================================================================
# ProfilingChunkConfig
# ===================================================================


class TestProfilingChunkConfig(TestBase):
    def test_default_values(self):
        cfg = ProfilingChunkConfig()
        self.assertFalse(cfg.enabled)
        self.assertFalse(cfg.need_timing)
        self.assertAlmostEqual(cfg.smooth_factor, 1.0)
        self.assertEqual(cfg.min_chunk, 4096)

    @patch("vllm_ascend.ascend_config.logger.warning")
    def test_need_timing_is_disabled_when_profiling_chunk_is_disabled(self, mock_warning):
        cfg = ProfilingChunkConfig(enabled=False, need_timing=True)

        self.assertFalse(cfg.need_timing)
        mock_warning.assert_called_once()

    def test_invalid_smooth_factor_raises(self):
        with self.assertRaises(ValueError):
            ProfilingChunkConfig(**{"smooth_factor": 0.0})
        with self.assertRaises(ValueError):
            ProfilingChunkConfig(**{"smooth_factor": 1.5})

    def test_invalid_min_chunk_raises(self):
        with self.assertRaises(ValueError):
            ProfilingChunkConfig(**{"min_chunk": 0})

    def test_need_timing_defaults_to_enabled(self):
        # When need_timing is not provided, it defaults to enabled.
        cfg = ProfilingChunkConfig(enabled=True)
        self.assertTrue(cfg.need_timing)
        cfg = ProfilingChunkConfig(enabled=False)
        self.assertFalse(cfg.need_timing)

    def test_need_timing_explicit_false_is_preserved(self):
        # Regression: previously `need_timing if need_timing else enabled`
        # turned explicit False back into enabled. The None sentinel must
        # distinguish "not provided" from "explicitly False".
        cfg = ProfilingChunkConfig(enabled=True, need_timing=False)
        self.assertFalse(cfg.need_timing)

    @patch("vllm.config.VllmConfig.__post_init__", MagicMock())
    @patch("vllm.config.device.DeviceConfig.__post_init__", MagicMock())
    @patch("vllm_ascend.platform._fix_incompatible_config")
    def test_enabled_without_pp_raises(self, _mock):
        clear_ascend_config()
        vllm_config = VllmConfig()
        vllm_config.model_config = MagicMock()
        vllm_config.additional_config = {
            "scheduler_config": {"profiling_chunk_config": {"enabled": True}},
            "refresh": True,
        }
        vllm_config.parallel_config.pipeline_parallel_size = 1
        with self.assertRaises(ValueError) as ctx:
            init_ascend_config(vllm_config)
        self.assertIn("pipeline parallelism", str(ctx.exception))
        clear_ascend_config()

    @patch("vllm.config.VllmConfig.__post_init__", MagicMock())
    @patch("vllm.config.device.DeviceConfig.__post_init__", MagicMock())
    @patch("vllm_ascend.platform._fix_incompatible_config")
    def test_enabled_with_pp_ok(self, _mock):
        clear_ascend_config()
        vllm_config = VllmConfig()
        vllm_config.model_config = MagicMock()
        vllm_config.additional_config = {
            "scheduler_config": {"profiling_chunk_config": {"enabled": True}},
            "refresh": True,
        }
        vllm_config.parallel_config.pipeline_parallel_size = 2
        ascend_config = init_ascend_config(vllm_config)
        self.assertTrue(ascend_config.scheduler_config.profiling_chunk_config.enabled)
        clear_ascend_config()

    @patch("vllm.config.VllmConfig.__post_init__", MagicMock())
    @patch("vllm.config.device.DeviceConfig.__post_init__", MagicMock())
    @patch("vllm_ascend.platform._fix_incompatible_config")
    def test_disabled_without_pp_ok(self, _mock):
        clear_ascend_config()
        vllm_config = VllmConfig()
        vllm_config.model_config = MagicMock()
        vllm_config.additional_config = {"refresh": True}
        ascend_config = init_ascend_config(vllm_config)
        self.assertFalse(ascend_config.scheduler_config.profiling_chunk_config.enabled)
        clear_ascend_config()


# ===================================================================
# ChunkSizePredictor
# ===================================================================


class TestChunkSizePredictor(TestBase):
    @staticmethod
    def _make_data(a, b, c, seq_lens):
        return [a * seq_len * seq_len + b * seq_len + c for seq_len in seq_lens]

    def test_fit_and_predict(self):
        predictor = ChunkSizePredictor()
        seq_lens = list(range(64, 8256, 128))
        latencies = self._make_data(1e-6, 0.01, 1.0, seq_lens)

        self.assertTrue(predictor.fit(seq_lens, latencies))
        predictor.set_target_latency(8192)
        predictor.is_ready = True

        chunk = predictor.predict(num_computed_tokens=0, base_chunk_size=8192, page_size=128)
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk % 128, 0)

    def test_predict_decreases_with_history(self):
        predictor = ChunkSizePredictor()
        seq_lens = list(range(64, 8256, 128))
        latencies = self._make_data(1e-6, 0.01, 1.0, seq_lens)
        predictor.fit(seq_lens, latencies)
        predictor.set_target_latency(8192)
        predictor.is_ready = True

        c0 = predictor.predict(0, 8192, 128)
        c1 = predictor.predict(4096, 8192, 128)
        c2 = predictor.predict(16384, 8192, 128)
        self.assertGreaterEqual(c0, c1)
        self.assertGreaterEqual(c1, c2)

    def test_predict_not_ready_returns_none(self):
        predictor = ChunkSizePredictor()
        self.assertIsNone(predictor.predict(0, 8192, 128))

    def test_fit_chunk_and_predict_with_history(self):
        predictor = ChunkSizePredictor()
        predictor.is_ready = True
        predictor.target_latency = 50.0

        data = []
        for i in range(10):
            c, h = 1000 + i * 100, i * 500
            data.append([(c + h) * c, c + h, 1, 1e-9 * (c + h) * c + 0.001 * (c + h) + 0.5])
        self.assertTrue(predictor.fit_chunk(data))
        predictor.with_history_ready = True

        result = predictor.predict_with_history(1000, 8192, 128)
        self.assertIsNotNone(result)
        self.assertEqual(result % 128, 0)


# ===================================================================
# ProfilingChunkManager
# ===================================================================


class TestProfilingChunkManager(TestBase):
    def test_not_ready_before_profiling(self):
        mgr = ProfilingChunkManager(base_chunk_size=8192, page_size=128)
        self.assertFalse(mgr.is_ready)
        self.assertIsNone(mgr.predict_chunk_size(0, 1.0))

    def test_run_profiling_success(self):
        mgr = ProfilingChunkManager(base_chunk_size=8192, page_size=128)
        seq_lens = list(range(64, 8256, 128))
        latencies = [1e-6 * seq_len * seq_len + 0.01 * seq_len + 1.0 for seq_len in seq_lens]
        self.assertTrue(mgr.predictor.fit(seq_lens, latencies))
        mgr.predictor.set_target_latency(8192)
        mgr.predictor.is_ready = True
        mgr._profiling_done = True

        self.assertTrue(mgr.is_ready)
        self.assertIsNotNone(mgr.predict_chunk_size(0, 1.0))

    def test_run_profiling_all_fail(self):
        mgr = ProfilingChunkManager(base_chunk_size=8192, page_size=128)
        too_few_seq_lens = [64, 128, 256]
        too_few_latencies = [1.0, 2.0, 3.0]
        self.assertFalse(mgr.predictor.fit(too_few_seq_lens, too_few_latencies))
        self.assertFalse(mgr.is_ready)
        self.assertIsNone(mgr.predict_chunk_size(0, 1.0))

    def test_record_batch_refines_model(self):
        mgr = ProfilingChunkManager(base_chunk_size=8192, page_size=128)
        seq_lens = list(range(64, 8256, 128))
        latencies = [1e-6 * seq_len * seq_len + 0.01 * seq_len + 1.0 for seq_len in seq_lens]
        mgr.predictor.fit(seq_lens, latencies)
        mgr.predictor.set_target_latency(8192)
        mgr.predictor.is_ready = True
        mgr._profiling_done = True

        for i in range(10):
            mgr.record_batch_execution_time([(4096 - i * 100, i * 500)], 0.05 + i * 0.01)
        self.assertGreaterEqual(len(mgr.chunked_fit_data), 10)
        self.assertTrue(mgr.history_ready)


# ===================================================================
# ProfilingChunkScheduler
# ===================================================================


class TestProfilingChunkScheduler(TestBase):
    @patch("vllm_ascend.patch.platform.patch_balance_schedule.init_ascend_config")
    # ProfilingChunkScheduler imports these names inside __init__, so patch the
    # source module from which that inline import resolves them.
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.ascend_config.get_ascend_config")
    @patch("vllm.config.ModelConfig.__post_init__", MagicMock())
    @patch("vllm.config.VllmConfig.__post_init__", MagicMock())
    @patch("vllm.config.device.DeviceConfig.__post_init__", MagicMock())
    def create_scheduler(
        self,
        mock_get_ascend_config,
        _mock_profiling_init_ascend_config,
        mock_balance_init_ascend_config,
        srf_enabled=False,
    ):
        profiling_cfg = MagicMock()
        profiling_cfg.enabled = True
        profiling_cfg.smooth_factor = 0.8
        profiling_cfg.min_chunk = 256
        mock_get_ascend_config.return_value.scheduler_config.profiling_chunk_config = profiling_cfg
        short_request_first_cfg = MagicMock()
        short_request_first_cfg.enabled = srf_enabled
        short_request_first_cfg.threshold = 256
        short_request_first_cfg.long_max_wait_ms = 2000.0
        mock_get_ascend_config.return_value.scheduler_config.short_request_first_config = short_request_first_cfg
        mock_balance_init_ascend_config.return_value.scheduler_config.short_request_first_config.enabled = False

        mock_hf_config = MagicMock()
        mock_hf_config.model_type = "qwen3"
        mock_hf_config.is_encoder_decoder = False
        mock_hf_config.architectures = ["Qwen3ForCausalLM"]
        model_config = ModelConfig(
            model=MODEL,
            tokenizer=MODEL,
            trust_remote_code=True,
            dtype="float16",
            seed=42,
            max_model_len=MAX_NUM_BATCHED_TOKENS,
        )
        model_config.hf_config = mock_hf_config
        model_config.hf_text_config = MagicMock()
        model_config.hf_text_config.is_encoder_decoder = False
        model_config.runner_type = "generate"

        scheduler_config = SchedulerConfig(
            max_num_seqs=MAX_NUM_SEQS,
            max_model_len=MAX_NUM_BATCHED_TOKENS,
            long_prefill_token_threshold=0,
            disable_chunked_mm_input=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            is_encoder_decoder=False,
        )
        scheduler_config.max_num_encoder_input_tokens = 10000
        scheduler_config.encoder_cache_size = 10000
        scheduler_config.chunked_prefill_enabled = True

        cache_config = CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
        )

        vllm_config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
        )
        vllm_config.parallel_config.pipeline_parallel_size = 2
        vllm_config.model_config.hf_config.is_encoder_decoder = False

        kv_cache_config = KVCacheConfig(
            num_blocks=10000,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32),
                )
            ],
        )
        kv_cache_config.hash_block_size = BLOCK_SIZE
        cache_config.num_gpu_blocks = 10000

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(ModelConfig, "is_encoder_decoder", new_callable=PropertyMock, return_value=False)
            )
            if not vllm_version_is("0.28.0"):
                # vLLM main (post-v0.28.0) reads model_config.uses_mrope in
                # Scheduler.__init__, which infinitely recurses on a bare
                # MagicMock hf_config. Override it to keep the UT runnable.
                stack.enter_context(
                    patch.object(ModelConfig, "uses_mrope", new_callable=PropertyMock, return_value=False)
                )
            scheduler = ProfilingChunkScheduler(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                block_size=BLOCK_SIZE,
                log_stats=True,
                structured_output_manager=MagicMock(spec=StructuredOutputManager),
            )

        should_advance = MagicMock()
        should_advance.return_value = False
        scheduler.structured_output_manager.should_advance = should_advance

        return scheduler

    def test_scheduler_init(self):
        scheduler = self.create_scheduler()
        self.assertIsNotNone(scheduler.profiling_chunk_manager)
        self.assertFalse(scheduler._profiling_initialized)
        self.assertFalse(scheduler._short_request_first_enabled)
        self.assertNotIsInstance(
            scheduler.waiting,
            ShortRequestFirstRequestQueue,
        )

    def test_scheduler_init_with_short_request_first(self):
        scheduler = self.create_scheduler(srf_enabled=True)

        self.assertIsInstance(
            scheduler.waiting,
            ShortRequestFirstRequestQueue,
        )

    def test_schedule_short_request_before_earlier_long_request(self):
        scheduler = self.create_scheduler(srf_enabled=True)

        long_request = create_requests(
            num_requests=1,
            num_tokens=512,
            request_id_prefix="long-",
        )[0]
        short_request = create_requests(
            num_requests=1,
            num_tokens=64,
            request_id_prefix="short-",
        )[0]

        # Simulate a long request arriving before a short request.
        scheduler.add_request(long_request)
        scheduler.add_request(short_request)

        output = scheduler.schedule()

        self.assertEqual(len(output.scheduled_new_reqs), 2)
        self.assertEqual(
            [request.request_id for request in scheduler.running],
            ["short-0", "long-0"],
        )

    def test_run_profiling_chunk_init_success(self):
        scheduler = self.create_scheduler()
        mock_executor = MagicMock()
        mock_executor.collective_rpc.return_value = [10.0]

        scheduler.run_profiling_chunk_init(mock_executor)

        self.assertTrue(scheduler._profiling_initialized)
        self.assertTrue(scheduler.profiling_chunk_manager.is_ready)
        self.assertFalse(scheduler.profiling_chunk_manager._set_time_done)

    def test_run_profiling_chunk_init_failure(self):
        scheduler = self.create_scheduler()
        mock_executor = MagicMock()
        mock_executor.collective_rpc.return_value = []

        scheduler.run_profiling_chunk_init(mock_executor)

        self.assertTrue(scheduler._profiling_initialized)
        self.assertFalse(scheduler.profiling_chunk_manager.is_ready)
        self.assertIsNone(scheduler.profiling_chunk_manager.predictor.target_latency)
        self.assertFalse(scheduler.profiling_chunk_manager._set_time_done)

    def test_run_profiling_chunk_init_skips_second_call(self):
        scheduler = self.create_scheduler()
        mock_executor = MagicMock()
        mock_executor.collective_rpc.return_value = [10.0]

        scheduler.run_profiling_chunk_init(mock_executor)
        call_count = mock_executor.collective_rpc.call_count

        scheduler.run_profiling_chunk_init(mock_executor)
        self.assertEqual(mock_executor.collective_rpc.call_count, call_count)

    def test_run_profiling_chunk_init_none_executor(self):
        scheduler = self.create_scheduler()
        scheduler.run_profiling_chunk_init(None)
        self.assertTrue(scheduler._profiling_initialized)
        self.assertFalse(scheduler.profiling_chunk_manager.is_ready)

    def test_schedule_new_requests(self):
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=5)
        for req in requests:
            scheduler.add_request(req)

        output = scheduler.schedule()
        self.assertEqual(len(output.scheduled_new_reqs), 5)
        self.assertEqual(len(scheduler.waiting), 0)
        self.assertEqual(len(scheduler.running), 5)

    def test_schedule_with_profiling_ready(self):
        """After profiling is ready, schedule() should still work correctly."""
        scheduler = self.create_scheduler()
        mock_executor = MagicMock()
        mock_executor.collective_rpc.return_value = [10.0]
        scheduler.run_profiling_chunk_init(mock_executor)
        self.assertTrue(scheduler.profiling_chunk_manager.is_ready)

        requests = create_requests(num_requests=3, num_tokens=100)
        for req in requests:
            scheduler.add_request(req)

        output = scheduler.schedule()
        self.assertGreater(len(output.scheduled_new_reqs), 0)
        total = sum(output.num_scheduled_tokens.values())
        self.assertGreater(total, 0)

    def test_schedule_chunked_prefill_running(self):
        """Running requests with num_computed_tokens > 0 get dynamic chunk."""
        scheduler = self.create_scheduler()
        mock_executor = MagicMock()
        mock_executor.collective_rpc.return_value = [10.0]
        scheduler.run_profiling_chunk_init(mock_executor)

        requests = create_requests(num_requests=1, num_tokens=2000, max_tokens=16)
        for req in requests:
            scheduler.add_request(req)

        output1 = scheduler.schedule()
        self.assertEqual(len(output1.scheduled_new_reqs), 1)

        model_output = make_output(scheduler)
        scheduler.update_from_output(output1, model_output)

        output2 = scheduler.schedule()
        self.assertGreater(output2.total_num_scheduled_tokens, 0)

    def test_update_from_output(self):
        scheduler = self.create_scheduler()
        requests = create_requests(num_requests=3)
        for req in requests:
            scheduler.add_request(req)

        output = scheduler.schedule()
        model_output = make_output(scheduler)
        scheduler.update_from_output(output, model_output)

        self.assertEqual(len(scheduler.running), 3)

    def _mark_ready(self, scheduler, predicted_chunk=64, need_timing=False):
        scheduler.profiling_chunk_config.need_timing = need_timing
        scheduler.profiling_chunk_manager._profiling_done = True
        scheduler.profiling_chunk_manager.predictor.is_ready = True
        scheduler.profiling_chunk_manager.predictor.target_latency = 10.0
        scheduler.profiling_chunk_manager.predict_chunk_size = MagicMock(return_value=predicted_chunk)
        scheduler.profiling_chunk_manager.predict_time = MagicMock(return_value=0.01)
        return scheduler

    def test_extract_latency_and_build_rpc_kwargs(self):
        self.assertEqual(ProfilingChunkScheduler._extract_latency(2), 2.0)
        self.assertIsNone(ProfilingChunkScheduler._extract_latency([]))
        self.assertIsNone(ProfilingChunkScheduler._extract_latency("bad"))
        self.assertEqual(ProfilingChunkScheduler._build_rpc_kwargs(object()), {})

        class NoRank:
            def collective_rpc(self, method, args=()):
                pass

        self.assertEqual(ProfilingChunkScheduler._build_rpc_kwargs(NoRank()), {})

        class WithRank:
            vllm_config = None

            def collective_rpc(self, method, args=(), unique_reply_rank=None):
                pass

        self.assertEqual(ProfilingChunkScheduler._build_rpc_kwargs(WithRank()), {})
        exe = WithRank()
        exe.vllm_config = MagicMock()
        exe.vllm_config.parallel_config.world_size = 8
        exe.vllm_config.parallel_config.tensor_parallel_size = 2
        exe.vllm_config.parallel_config.prefill_context_parallel_size = 1
        self.assertEqual(ProfilingChunkScheduler._build_rpc_kwargs(exe)["unique_reply_rank"], 6)

    def test_run_profiling_chunk_init_error_paths(self):
        scheduler = self.create_scheduler()
        scheduler.profiling_chunk_manager.base_chunk_size = 1
        n = {"i": 0}

        def rpc(*_a, **_k):
            n["i"] += 1
            if n["i"] % 2 == 0:
                raise RuntimeError("fail")
            return None

        mock_executor = MagicMock()
        mock_executor.collective_rpc.side_effect = rpc
        scheduler.run_profiling_chunk_init(mock_executor)
        self.assertFalse(scheduler.profiling_chunk_manager.is_ready)

        scheduler2 = self.create_scheduler()
        mock_executor2 = MagicMock()
        mock_executor2.collective_rpc.return_value = [10.0]
        with patch.object(scheduler2.profiling_chunk_manager.predictor, "fit", return_value=False):
            scheduler2.run_profiling_chunk_init(mock_executor2)
        self.assertFalse(scheduler2.profiling_chunk_manager.is_ready)

    def test_schedule_pause_dynamic_chunk_spec_and_mamba(self):
        scheduler = self.create_scheduler()
        scheduler._pause_state = PauseState.PAUSED_ALL
        paused = scheduler.schedule()
        self.assertEqual(paused.total_num_scheduled_tokens, 0)
        scheduler._pause_state = PauseState.UNPAUSED

        self._mark_ready(scheduler)
        scheduler.use_v2_model_runner = True
        scheduler.scheduler_config.long_prefill_token_threshold = 32
        scheduler.need_mamba_block_aligned_split = True
        mamba_calls = {"n": 0}

        def split(req, num, *_a, **_k):
            mamba_calls["n"] += 1
            return 0 if mamba_calls["n"] > 2 else num

        scheduler._mamba_block_aligned_split = split
        reqs = create_requests(num_requests=1, num_tokens=200, max_tokens=16)
        scheduler.add_request(reqs[0])
        out1 = scheduler.schedule()
        self.assertGreater(out1.total_num_scheduled_tokens, 0)

        reqs[0].spec_token_ids = [1, 2, 3, 4, 5, 6]
        reqs[0].num_computed_tokens = reqs[0].num_tokens
        out2 = scheduler.schedule()
        self.assertGreaterEqual(out2.total_num_scheduled_tokens, 0)
        reqs[0].num_computed_tokens = 10
        scheduler.schedule()

        reqs[0].num_output_placeholders = 1
        reqs[0].num_computed_tokens = 1000
        scheduler.schedule()

        self._mark_ready(scheduler, predicted_chunk=None, need_timing=True)
        reqs[0].num_output_placeholders = 0
        reqs[0].num_computed_tokens = 10
        scheduler.schedule()
        self._mark_ready(scheduler, predicted_chunk=None, need_timing=False)
        scheduler.schedule()

    def test_schedule_preempt_lora_connector_and_waiting_edges(self):
        scheduler = self.create_scheduler()
        self._mark_ready(scheduler)
        reqs = create_requests(num_requests=2, num_tokens=80, max_tokens=16)
        for req in reqs:
            scheduler.add_request(req)
        scheduler.schedule()

        orig_alloc = scheduler.kv_cache_manager.allocate_slots
        n = {"c": 0}

        def alloc(*a, **k):
            n["c"] += 1
            return orig_alloc(*a, **k) if n["c"] == 1 else None

        scheduler.kv_cache_manager.allocate_slots = alloc
        scheduler.policy = SchedulingPolicy.PRIORITY
        reqs[0].priority = 10
        scheduler.schedule()

        scheduler2 = self.create_scheduler()
        self._mark_ready(scheduler2)
        reqs2 = create_requests(num_requests=2, num_tokens=80, max_tokens=16)
        for req in reqs2:
            scheduler2.add_request(req)
        scheduler2.schedule()
        scheduler2.kv_cache_manager.allocate_slots = MagicMock(return_value=None)
        scheduler2.schedule()

        scheduler3 = self.create_scheduler()
        self._mark_ready(scheduler3)
        scheduler3.lora_config = MagicMock(max_loras=1)
        scheduler3.connector = MagicMock()
        scheduler3.ec_connector = MagicMock()
        scheduler3._build_kv_connector_meta = MagicMock(return_value="meta")
        scheduler3.connector.get_num_new_matched_tokens.side_effect = [(2, True), (2, False)]
        scheduler3.connector_prefix_cache_stats = MagicMock()
        scheduler3._try_schedule_encoder_inputs = MagicMock(return_value=([0], 8, 100, [1]))
        scheduler3.encoder_cache_manager.allocate = MagicMock()
        scheduler3.is_encoder_decoder = True
        scheduler3._is_blocked_waiting_status = lambda status: status == RequestStatus.WAITING_FOR_REMOTE_KVS
        scheduler3._try_promote_blocked_waiting_request = MagicMock(return_value=False)
        scheduler3.max_num_running_reqs = 2

        edge_reqs = create_requests(num_requests=4, num_tokens=40, max_tokens=16)
        edge_reqs[2].lora_request = MagicMock(lora_int_id=1)
        edge_reqs[2].get_num_encoder_embeds = MagicMock(return_value=1)
        edge_reqs[3].lora_request = MagicMock(lora_int_id=2)
        for req in edge_reqs:
            scheduler3.add_request(req)
        edge_reqs[0].status = RequestStatus.WAITING_FOR_REMOTE_KVS
        with patch.object(Request, "has_encoder_inputs", new_callable=PropertyMock, return_value=True):
            scheduler3.schedule()
            scheduler3.schedule()

        scheduler4 = self.create_scheduler()
        self._mark_ready(scheduler4)
        preempted = create_requests(num_requests=1, num_tokens=20)[0]
        scheduler4.add_request(preempted)
        preempted.status = RequestStatus.PREEMPTED
        preempted.num_computed_tokens = 4
        scheduler4.schedule()

        scheduler5 = self.create_scheduler()
        self._mark_ready(scheduler5)
        scheduler5.max_num_running_reqs = 0
        scheduler5.add_request(create_requests(num_requests=1, num_tokens=20)[0])
        scheduler5.schedule()

        scheduler6 = self.create_scheduler()
        self._mark_ready(scheduler6)
        scheduler6.kv_cache_manager.allocate_slots = MagicMock(return_value=None)
        scheduler6.add_request(create_requests(num_requests=1, num_tokens=20)[0])
        scheduler6.schedule()

        scheduler7 = self.create_scheduler()
        self._mark_ready(scheduler7)
        scheduler7.scheduler_config.enable_chunked_prefill = False
        scheduler7.max_num_scheduled_tokens = 4
        scheduler7.add_request(create_requests(num_requests=1, num_tokens=20)[0])
        scheduler7.schedule()
