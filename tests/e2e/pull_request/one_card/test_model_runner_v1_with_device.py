import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

import vllm_ascend.compilation.acl_graph as acl_graph
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

BLOCK_SIZE = 128
NUM_BLOCKS = 10
DEVICE_TYPE = current_platform.device_type
FAKE_WEIGHT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ut", "_fake_weight")


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
        model=FAKE_WEIGHT_PATH,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
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
    with (
        set_current_vllm_config(vllm_config),
        patch("vllm_ascend.worker.block_table.get_dcp_group") as mock_get_dcp_group,
        patch(
            "vllm_ascend.worker.block_table.get_decode_context_model_parallel_world_size",
            return_value=1,
        ),
    ):
        mock_dcp_group = MagicMock(spec=GroupCoordinator)
        mock_dcp_group.world_size = 1
        mock_dcp_group.rank_in_group = 0
        mock_get_dcp_group.return_value = mock_dcp_group
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


@pytest.mark.parametrize(
    "num_computed_tokens, num_scheduled_tokens, num_tokens, num_reqs, "
    "max_num_scheduled_tokens, use_cascade_attn, force_eager, "
    "force_uniform_decode, spec_decode_tokens",
    [
        # ---- force_eager=True: bypass cudagraph dispatch ----
        pytest.param(
            [0, 0, 0],
            [10, 10, 10],
            30,
            3,
            10,
            False,
            True,
            None,
            0,
            id="prefill_eager",
        ),
        pytest.param(
            [5, 10, 15],
            [1, 1, 1],
            3,
            3,
            1,
            False,
            True,
            None,
            0,
            id="decode_eager",
        ),
        pytest.param(
            [0, 5, 10],
            [10, 1, 1],
            12,
            3,
            10,
            False,
            True,
            None,
            0,
            id="mixed_eager",
        ),
        # ---- force_eager=False: go through real dispatch path ----
        pytest.param(
            [0, 0, 0],
            [10, 10, 10],
            30,
            3,
            10,
            False,
            False,
            None,
            0,
            id="prefill_dispatch",
        ),
        pytest.param(
            [5, 10, 15],
            [1, 1, 1],
            3,
            3,
            1,
            False,
            False,
            None,
            0,
            id="decode_uniform_dispatch",
        ),
        pytest.param(
            [0, 5, 10],
            [10, 1, 1],
            12,
            3,
            10,
            False,
            False,
            None,
            0,
            id="mixed_dispatch",
        ),
        pytest.param(
            [0],
            [50],
            50,
            1,
            50,
            False,
            False,
            None,
            0,
            id="single_prefill_dispatch",
        ),
        pytest.param(
            [100],
            [1],
            1,
            1,
            1,
            False,
            False,
            None,
            0,
            id="single_decode_dispatch",
        ),
        # ---- cascade attention ----
        pytest.param(
            [0, 0, 0],
            [10, 10, 10],
            30,
            3,
            10,
            True,
            False,
            None,
            0,
            id="prefill_cascade_attn",
        ),
        # ---- force_uniform_decode override ----
        pytest.param(
            [5, 10, 15],
            [1, 1, 1],
            3,
            3,
            1,
            False,
            False,
            True,
            0,
            id="decode_force_uniform_true",
        ),
        pytest.param(
            [5, 10, 15],
            [1, 1, 1],
            3,
            3,
            1,
            False,
            False,
            False,
            0,
            id="decode_force_uniform_false",
        ),
        # ---- spec_decode: uniform_decode depends on is_all_decode ----
        pytest.param(
            [5, 10, 15],
            [4, 4, 4],
            12,
            3,
            4,
            False,
            False,
            None,
            3,
            id="spec_decode_all_decode",
        ),
        pytest.param(
            [0, 0, 0],
            [4, 4, 4],
            12,
            3,
            4,
            False,
            False,
            None,
            3,
            id="spec_decode_all_prefill",
        ),
        pytest.param(
            [0, 5, 10],
            [4, 4, 4],
            12,
            3,
            4,
            False,
            False,
            None,
            3,
            id="spec_decode_mixed",
        ),
        # ---- large batch ----
        pytest.param(
            [0, 0, 0, 0, 0],
            [20, 20, 20, 20, 20],
            100,
            5,
            20,
            False,
            False,
            None,
            0,
            id="large_prefill_dispatch",
        ),
    ],
)
def test_determine_batch_execution_and_padding(
    model_runner,
    num_computed_tokens,
    num_scheduled_tokens,
    num_tokens,
    num_reqs,
    max_num_scheduled_tokens,
    use_cascade_attn,
    force_eager,
    force_uniform_decode,
    spec_decode_tokens,
):
    runner = model_runner

    # Set up spec decode scenario by overriding runner attributes
    saved_spec_config = runner.speculative_config
    saved_query_len = runner.uniform_decode_query_len
    if spec_decode_tokens > 0:
        runner.speculative_config = type("FakeSpecConfig", (), {"num_speculative_tokens": spec_decode_tokens})()
        runner.uniform_decode_query_len = 1 + spec_decode_tokens
    else:
        runner.speculative_config = None
        runner.uniform_decode_query_len = 1

    try:
        runner.input_batch.num_computed_tokens_cpu[:num_reqs] = num_computed_tokens
        num_scheduled_tokens_np = np.array(num_scheduled_tokens, dtype=np.int32)

        kwargs = dict(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=use_cascade_attn,
            force_eager=force_eager,
        )
        if force_uniform_decode is not None:
            kwargs["force_uniform_decode"] = force_uniform_decode

        (
            cudagraph_mode,
            batch_desc,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        ) = runner._determine_batch_execution_and_padding(**kwargs)

        # force_eager always bypasses cudagraph dispatch
        if force_eager:
            assert cudagraph_mode == CUDAGraphMode.NONE
            assert batch_desc.num_tokens == num_tokens
        else:
            # The resolved cudagraph_mode is determined during
            # initialize_attn_backend and stored in the dispatcher.
            resolved_mode = runner.cudagraph_dispatcher.cudagraph_mode
            if resolved_mode == CUDAGraphMode.NONE:
                assert cudagraph_mode == CUDAGraphMode.NONE
                assert batch_desc.num_tokens == num_tokens
            else:
                # Dispatcher may match a captured key (PIECEWISE/FULL)
                # or fall back to NONE if num_tokens exceeds max capture size.
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
    finally:
        runner.speculative_config = saved_spec_config
        runner.uniform_decode_query_len = saved_query_len


@pytest.mark.parametrize(
    ("num_spec_tokens", "computed", "prompts", "scheduled", "expected_mode"),
    [
        pytest.param(0, [7], [8], [1], CUDAGraphMode.FULL, id="stateful_one_token_handoff"),
        pytest.param(0, [0], [1], [1], CUDAGraphMode.NONE, id="first_token_without_state"),
        pytest.param(7, [16, 24], [8, 8], [8, 8], CUDAGraphMode.FULL, id="steady_spec_decode"),
        pytest.param(7, [16, 7], [8, 8], [8, 8], CUDAGraphMode.FULL, id="handoff_padded_to_spec_width"),
        pytest.param(7, [16, 0], [8, 8], [8, 8], CUDAGraphMode.NONE, id="spec_width_prefill_without_state"),
        pytest.param(7, [16, 7], [8, 8], [8, 1], CUDAGraphMode.NONE, id="nonuniform_handoff"),
    ],
)
@pytest.mark.parametrize("dp_size", [1, 4])
def test_stateful_handoff_preserves_decode_graph(
    monkeypatch,
    num_spec_tokens,
    computed,
    prompts,
    scheduled,
    expected_mode,
    dp_size,
):
    # Exercise the real dispatcher and DP synchronization using CPU metadata only.
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.dp_size = dp_size
    runner.dp_rank = 0
    runner.parallel_config = SimpleNamespace(
        data_parallel_size=dp_size,
        data_parallel_rank=0,
        tensor_parallel_size=8,
        use_sequence_parallel_moe=True,
    )
    runner.compilation_config = SimpleNamespace(
        pass_config=SimpleNamespace(enable_sp=True),
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[8, 16, 24, 32],
        max_cudagraph_capture_size=32,
        compile_sizes=[],
    )
    runner.vllm_config = SimpleNamespace(
        parallel_config=runner.parallel_config,
        compilation_config=runner.compilation_config,
        scheduler_config=SimpleNamespace(max_num_seqs=32),
        observability_config=SimpleNamespace(cudagraph_metrics=False),
        num_speculative_tokens=num_spec_tokens,
        lora_config=None,
    )
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.uniform_decode_query_len = 1 + num_spec_tokens
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array(computed),
        num_prompt_tokens=np.array(prompts),
        lora_id_to_lora_request={},
    )
    runner.cudagraph_dispatcher = CudagraphDispatcher(runner.vllm_config)
    runner.cudagraph_dispatcher.initialize_cudagraph_keys(
        CUDAGraphMode.FULL_DECODE_ONLY, runner.uniform_decode_query_len
    )

    def all_reduce(packed_tensor, group):
        # Other DP replicas are already decoding at the largest captured size.
        packed_tensor[0, 1:] = 32
        packed_tensor[1, 1:] = CUDAGraphMode.FULL.value

    module = "vllm_ascend.worker.model_runner_v1"
    monkeypatch.setattr(f"{module}.should_skip_allreduce_across_dp_group", lambda *args: False)
    monkeypatch.setattr(f"{module}.get_dp_group", lambda: SimpleNamespace(cpu_group=None))
    monkeypatch.setattr(f"{module}.dist.all_reduce", all_reduce)
    mode, descriptor, _, tokens_across_dp, _ = runner._determine_batch_execution_and_padding(
        num_tokens=sum(scheduled),
        num_reqs=len(scheduled),
        num_scheduled_tokens_np=np.array(scheduled, dtype=np.int32),
        max_num_scheduled_tokens=max(scheduled),
        use_cascade_attn=False,
    )

    assert mode == expected_mode
    assert descriptor.uniform == (expected_mode == CUDAGraphMode.FULL)
    if dp_size > 1:
        assert descriptor.num_tokens == 32
        torch.testing.assert_close(tokens_across_dp, torch.full((dp_size,), 32, dtype=torch.int32))
