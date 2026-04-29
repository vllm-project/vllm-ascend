import numpy as np
import pytest
import torch
from vllm.config import (
    CacheConfig,
    CUDAGraphMode,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

import vllm_ascend.compilation.acl_graph as acl_graph
from tests.ut.conftest import RunnerDeviceType, npu_test
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

BLOCK_SIZE = 128
NUM_BLOCKS = 10
DEVICE_TYPE = current_platform.device_type


def initialize_kv_cache(runner: NPUModelRunner):
    """
    Only perform necessary steps in NPUModelRunner.initialize_kv_cache()
    """
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size, shared_by=["layer.0"]),
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)],
    )
    runner.kv_cache_config = kv_cache_config
    runner.input_batch = NPUInputBatch(
        max_num_reqs=runner.max_num_reqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size],
        kernel_block_sizes=[[kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size]],
    )
    runner.initialize_attn_backend(kv_cache_config)


def get_vllm_config():
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config


@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        model_config = vllm_config.model_config
        num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = model_config.get_head_size()
        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(num_heads, head_size, 0.1)
        runner = NPUModelRunner(vllm_config, DEVICE_TYPE)
        initialize_kv_cache(runner)
        yield runner
        # Reset global state set by _check_and_update_cudagraph_mode
        # so the next test case can reinitialize cleanly.
        acl_graph._graph_params = None
        acl_graph._draft_graph_params = None


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
class TestDetermineBatchExecutionAndPadding:
    """Tests for ``NPUModelRunner._determine_batch_execution_and_padding``."""

    @staticmethod
    def _set_spec_decode(runner, num_spec_tokens):
        """Override speculative_config + uniform_decode_query_len for one case."""
        if num_spec_tokens > 0:
            runner.speculative_config = type("FakeSpecConfig", (), {"num_speculative_tokens": num_spec_tokens})()
            runner.uniform_decode_query_len = 1 + num_spec_tokens
        else:
            runner.speculative_config = None
            runner.uniform_decode_query_len = 1

    @staticmethod
    def _invoke(
        runner,
        *,
        num_computed_tokens,
        num_scheduled_tokens,
        num_tokens,
        max_num_scheduled_tokens,
        use_cascade_attn,
        force_eager,
        force_uniform_decode=None,
    ):
        num_reqs = len(num_computed_tokens)
        runner.input_batch.num_computed_tokens_cpu[:num_reqs] = num_computed_tokens
        kwargs = dict(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=np.array(num_scheduled_tokens, dtype=np.int32),
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=use_cascade_attn,
            force_eager=force_eager,
        )
        if force_uniform_decode is not None:
            kwargs["force_uniform_decode"] = force_uniform_decode
        return runner._determine_batch_execution_and_padding(**kwargs)

    @staticmethod
    def _assert_common(runner, result, num_tokens, force_eager):
        cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, cudagraph_stats = result
        if force_eager:
            # force_eager always bypasses cudagraph dispatch
            assert cudagraph_mode == CUDAGraphMode.NONE
            assert batch_desc.num_tokens == num_tokens
        else:
            # The resolved cudagraph_mode is determined during initialize_attn_backend
            # and stored in the dispatcher.
            resolved_mode = runner.cudagraph_dispatcher.cudagraph_mode
            if resolved_mode == CUDAGraphMode.NONE:
                assert cudagraph_mode == CUDAGraphMode.NONE
                assert batch_desc.num_tokens == num_tokens
            else:
                # Dispatcher may match a captured key (PIECEWISE/FULL) or fall
                # back to NONE when num_tokens exceeds max capture size.
                assert cudagraph_mode in (
                    CUDAGraphMode.NONE,
                    CUDAGraphMode.PIECEWISE,
                    CUDAGraphMode.FULL,
                )
                # Padding can only increase, never shrink
                assert batch_desc.num_tokens >= num_tokens
        # dp_size=1: no micro-batching, no cross-dp coordination
        assert should_ubatch is False
        assert num_tokens_across_dp is None
        # cudagraph_metrics disabled by default
        assert cudagraph_stats is None

    # ---------- force_eager=True: bypass cudagraph dispatch ----------

    def test_prefill_eager(self, model_runner):
        """All requests are fresh prefill (num_computed=0); force_eager bypasses dispatch."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[0, 0, 0],
            num_scheduled_tokens=[10, 10, 10],
            num_tokens=30,
            max_num_scheduled_tokens=10,
            use_cascade_attn=False,
            force_eager=True,
        )
        self._assert_common(model_runner, result, num_tokens=30, force_eager=True)

    def test_decode_eager(self, model_runner):
        """All requests are decoding (num_computed>0, 1 token each); force_eager."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[5, 10, 15],
            num_scheduled_tokens=[1, 1, 1],
            num_tokens=3,
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
            force_eager=True,
        )
        self._assert_common(model_runner, result, num_tokens=3, force_eager=True)

    def test_mixed_eager(self, model_runner):
        """Mixed prefill + decode batch with force_eager."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[0, 5, 10],
            num_scheduled_tokens=[10, 1, 1],
            num_tokens=12,
            max_num_scheduled_tokens=10,
            use_cascade_attn=False,
            force_eager=True,
        )
        self._assert_common(model_runner, result, num_tokens=12, force_eager=True)

    # ---------- force_eager=False: go through real dispatch path ----------

    def test_prefill_dispatch(self, model_runner):
        """All-prefill batch goes through the real cudagraph dispatcher."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[0, 0, 0],
            num_scheduled_tokens=[10, 10, 10],
            num_tokens=30,
            max_num_scheduled_tokens=10,
            use_cascade_attn=False,
            force_eager=False,
        )
        self._assert_common(model_runner, result, num_tokens=30, force_eager=False)

    def test_decode_uniform_dispatch(self, model_runner):
        """Pure decode batch (uniform_decode=True) routed through dispatch."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[5, 10, 15],
            num_scheduled_tokens=[1, 1, 1],
            num_tokens=3,
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
            force_eager=False,
        )
        self._assert_common(model_runner, result, num_tokens=3, force_eager=False)

    def test_mixed_dispatch(self, model_runner):
        """Mixed prefill+decode batch routed through dispatch."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[0, 5, 10],
            num_scheduled_tokens=[10, 1, 1],
            num_tokens=12,
            max_num_scheduled_tokens=10,
            use_cascade_attn=False,
            force_eager=False,
        )
        self._assert_common(model_runner, result, num_tokens=12, force_eager=False)

    def test_single_prefill_dispatch(self, model_runner):
        """Single prefill request, large num_tokens."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[0],
            num_scheduled_tokens=[50],
            num_tokens=50,
            max_num_scheduled_tokens=50,
            use_cascade_attn=False,
            force_eager=False,
        )
        self._assert_common(model_runner, result, num_tokens=50, force_eager=False)

    def test_single_decode_dispatch(self, model_runner):
        """Single decode request, num_tokens=1."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[100],
            num_scheduled_tokens=[1],
            num_tokens=1,
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
            force_eager=False,
        )
        self._assert_common(model_runner, result, num_tokens=1, force_eager=False)

    # ---------- cascade attention ----------

    def test_prefill_cascade_attn(self, model_runner):
        """use_cascade_attn=True disables FULL cudagraph in dispatch."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[0, 0, 0],
            num_scheduled_tokens=[10, 10, 10],
            num_tokens=30,
            max_num_scheduled_tokens=10,
            use_cascade_attn=True,
            force_eager=False,
        )
        self._assert_common(model_runner, result, num_tokens=30, force_eager=False)

    # ---------- force_uniform_decode override ----------

    def test_decode_force_uniform_true(self, model_runner):
        """Explicit force_uniform_decode=True overrides auto-detection."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[5, 10, 15],
            num_scheduled_tokens=[1, 1, 1],
            num_tokens=3,
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
            force_eager=False,
            force_uniform_decode=True,
        )
        self._assert_common(model_runner, result, num_tokens=3, force_eager=False)

    def test_decode_force_uniform_false(self, model_runner):
        """Explicit force_uniform_decode=False overrides auto-detection."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[5, 10, 15],
            num_scheduled_tokens=[1, 1, 1],
            num_tokens=3,
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
            force_eager=False,
            force_uniform_decode=False,
        )
        self._assert_common(model_runner, result, num_tokens=3, force_eager=False)

    # ---------- spec_decode: uniform_decode depends on is_all_decode ----------

    def test_spec_decode_all_decode(self, model_runner):
        """Spec decode + all-decode batch; uniform_decode requires is_all_decode."""
        try:
            self._set_spec_decode(model_runner, 3)
            result = self._invoke(
                model_runner,
                num_computed_tokens=[5, 10, 15],
                num_scheduled_tokens=[4, 4, 4],
                num_tokens=12,
                max_num_scheduled_tokens=4,
                use_cascade_attn=False,
                force_eager=False,
            )
            self._assert_common(model_runner, result, num_tokens=12, force_eager=False)
        finally:
            self._set_spec_decode(model_runner, 0)

    def test_spec_decode_all_prefill(self, model_runner):
        """Spec decode + all-prefill: is_all_decode=False, so uniform_decode=False."""
        try:
            self._set_spec_decode(model_runner, 3)
            result = self._invoke(
                model_runner,
                num_computed_tokens=[0, 0, 0],
                num_scheduled_tokens=[4, 4, 4],
                num_tokens=12,
                max_num_scheduled_tokens=4,
                use_cascade_attn=False,
                force_eager=False,
            )
            self._assert_common(model_runner, result, num_tokens=12, force_eager=False)
        finally:
            self._set_spec_decode(model_runner, 0)

    def test_spec_decode_mixed(self, model_runner):
        """Spec decode + mixed batch; is_all_decode=False, uniform_decode=False."""
        try:
            self._set_spec_decode(model_runner, 3)
            result = self._invoke(
                model_runner,
                num_computed_tokens=[0, 5, 10],
                num_scheduled_tokens=[4, 4, 4],
                num_tokens=12,
                max_num_scheduled_tokens=4,
                use_cascade_attn=False,
                force_eager=False,
            )
            self._assert_common(model_runner, result, num_tokens=12, force_eager=False)
        finally:
            self._set_spec_decode(model_runner, 0)

    # ---------- large batch ----------

    def test_large_prefill_dispatch(self, model_runner):
        """5-request prefill batch with 100 total tokens."""
        self._set_spec_decode(model_runner, 0)
        result = self._invoke(
            model_runner,
            num_computed_tokens=[0, 0, 0, 0, 0],
            num_scheduled_tokens=[20, 20, 20, 20, 20],
            num_tokens=100,
            max_num_scheduled_tokens=20,
            use_cascade_attn=False,
            force_eager=False,
        )
        self._assert_common(model_runner, result, num_tokens=100, force_eager=False)


@npu_test(num_npus=1, npu_type=RunnerDeviceType.A2)
class TestCalcSpecDecodeMetadata:
    """Tests for ``NPUModelRunner._calc_spec_decode_metadata``."""

    @staticmethod
    def _run(
        runner,
        *,
        num_draft_tokens,
        cu_num_scheduled_tokens,
        expected_logits_indices,
        expected_target_logits_indices,
        expected_bonus_logits_indices,
        expected_cu_num_draft_tokens,
        expected_cu_num_sampled_tokens,
    ):
        num_draft_np = np.array(num_draft_tokens, dtype=np.int32)
        cu_num_scheduled_np = np.array(cu_num_scheduled_tokens, dtype=np.int32)

        # input_ids.gpu must contain valid token data so the index lookup at
        # `self.input_ids.gpu[logits_indices]` doesn't read past the buffer.
        # Fill with a deterministic pattern (token id == position) so that
        # draft_token_ids equals logits_indices[target_logits_indices + 1].
        total_scheduled = int(cu_num_scheduled_np[-1])
        pattern = torch.arange(total_scheduled, dtype=runner.input_ids.gpu.dtype, device=runner.device)
        runner.input_ids.gpu[:total_scheduled].copy_(pattern)

        metadata = runner._calc_spec_decode_metadata(
            num_draft_np,
            cu_num_scheduled_np,
            num_pcp_pads=None,
        )

        assert metadata.num_draft_tokens == num_draft_tokens
        np.testing.assert_array_equal(
            metadata.logits_indices.cpu().numpy(),
            np.array(expected_logits_indices, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            metadata.target_logits_indices.cpu().numpy(),
            np.array(expected_target_logits_indices, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            metadata.bonus_logits_indices.cpu().numpy(),
            np.array(expected_bonus_logits_indices, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            metadata.cu_num_draft_tokens.cpu().numpy(),
            np.array(expected_cu_num_draft_tokens, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            metadata.cu_num_sampled_tokens.cpu().numpy(),
            np.array(expected_cu_num_sampled_tokens, dtype=np.int32),
        )

        # draft_token_ids = input_ids[logits_indices][target_logits_indices + 1]
        # With the identity pattern, this equals logits_indices[target_logits_indices + 1].
        expected_draft_ids = np.array(expected_logits_indices, dtype=np.int64)[
            np.array(expected_target_logits_indices, dtype=np.int64) + 1
        ]
        np.testing.assert_array_equal(
            metadata.draft_token_ids.cpu().numpy(),
            expected_draft_ids,
        )

    def test_mixed_draft_counts(self, model_runner):
        """Docstring example: 5 requests with mixed draft counts [3, 0, 2, 0, 1]."""
        self._run(
            model_runner,
            num_draft_tokens=[3, 0, 2, 0, 1],
            cu_num_scheduled_tokens=[4, 104, 107, 207, 209],
            expected_logits_indices=[0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208],
            expected_target_logits_indices=[0, 1, 2, 5, 6, 9],
            expected_bonus_logits_indices=[3, 4, 7, 8, 10],
            expected_cu_num_draft_tokens=[3, 3, 5, 5, 6],
            expected_cu_num_sampled_tokens=[4, 5, 8, 9, 11],
        )

    def test_no_draft_tokens(self, model_runner):
        """All requests have 0 draft tokens; only bonus tokens are sampled."""
        self._run(
            model_runner,
            num_draft_tokens=[0, 0, 0],
            cu_num_scheduled_tokens=[10, 20, 30],
            expected_logits_indices=[9, 19, 29],
            expected_target_logits_indices=[],
            expected_bonus_logits_indices=[0, 1, 2],
            expected_cu_num_draft_tokens=[0, 0, 0],
            expected_cu_num_sampled_tokens=[1, 2, 3],
        )

    def test_uniform_drafts(self, model_runner):
        """Every request has the same number of draft tokens."""
        self._run(
            model_runner,
            num_draft_tokens=[2, 2, 2],
            cu_num_scheduled_tokens=[3, 6, 9],
            expected_logits_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            expected_target_logits_indices=[0, 1, 3, 4, 6, 7],
            expected_bonus_logits_indices=[2, 5, 8],
            expected_cu_num_draft_tokens=[2, 4, 6],
            expected_cu_num_sampled_tokens=[3, 6, 9],
        )

    def test_single_request_with_drafts(self, model_runner):
        """Single request with 3 draft tokens (minimal spec_decode case)."""
        self._run(
            model_runner,
            num_draft_tokens=[3],
            cu_num_scheduled_tokens=[4],
            expected_logits_indices=[0, 1, 2, 3],
            expected_target_logits_indices=[0, 1, 2],
            expected_bonus_logits_indices=[3],
            expected_cu_num_draft_tokens=[3],
            expected_cu_num_sampled_tokens=[4],
        )

    def test_single_request_no_draft(self, model_runner):
        """Single request with 0 draft tokens (most degenerate case)."""
        self._run(
            model_runner,
            num_draft_tokens=[0],
            cu_num_scheduled_tokens=[5],
            expected_logits_indices=[4],
            expected_target_logits_indices=[],
            expected_bonus_logits_indices=[0],
            expected_cu_num_draft_tokens=[0],
            expected_cu_num_sampled_tokens=[1],
        )
