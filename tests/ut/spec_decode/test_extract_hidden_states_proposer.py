#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Unit tests for AscendExtractHiddenStatesProposer.

This test file follows the pattern from vllm's test_extract_hidden_states.py,
with Ascend-specific additions for ACL graph differences.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.config import CacheConfig, CUDAGraphMode, VllmConfig, set_current_vllm_config

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.spec_decode.extract_hidden_states_proposer import (
    AscendExtractHiddenStatesProposer,
)


@pytest.fixture(autouse=True)
def _no_pin_memory():
    # On Ascend/NPU CI runners without physical hardware, torch.zeros(...,
    # pin_memory=True) triggers aclInit and fails.  Patch
    # is_pin_memory_available so vllm's ExtractHiddenStatesProposer.__init__
    # creates CpuGpuBuffer with pin_memory=False.
    with patch(
        "vllm.v1.spec_decode.extract_hidden_states.is_pin_memory_available",
        return_value=False,
    ):
        yield


class MockCachedRequestState:
    """Mock CachedRequestState for testing (same pattern as vllm)."""

    def __init__(self, req_id: str, token_ids: list[int]):
        self.req_id = req_id
        self.token_ids = token_ids

    def get_token_id(self, position: int) -> int:
        if 0 <= position < len(self.token_ids):
            return self.token_ids[position]
        return 0


class MockInputBatch:
    """Mock InputBatch for testing (same pattern as vllm)."""

    def __init__(
        self,
        num_reqs: int,
        req_ids: list[str],
        vocab_size: int,
        num_tokens_no_spec: list[int] | None = None,
    ):
        self.num_reqs = num_reqs
        self.req_ids = req_ids
        self.vocab_size = vocab_size
        if num_tokens_no_spec is None:
            self.num_tokens_no_spec = np.array([5] * num_reqs, dtype=np.int64)
        else:
            self.num_tokens_no_spec = np.array(num_tokens_no_spec, dtype=np.int64)


def _create_vllm_config(num_speculative_tokens: int = 1, layer_ids: list[int] | None = None):
    """Create a VllmConfig for testing (simplified version of vllm's pattern)."""
    from unittest.mock import MagicMock

    if layer_ids is None:
        layer_ids = [1, 2, 3, 4]

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.speculative_config = MagicMock()
    vllm_config.speculative_config.num_speculative_tokens = num_speculative_tokens
    vllm_config.speculative_config.draft_tensor_parallel_size = 1
    vllm_config.speculative_config.draft_model_config = MagicMock()
    vllm_config.speculative_config.draft_model_config.uses_xdrope_dim = 0
    vllm_config.speculative_config.draft_model_config.uses_mrope = False
    vllm_config.speculative_config.disable_padded_drafter_batch = False

    vllm_config.cache_config = MagicMock(spec=CacheConfig)
    vllm_config.cache_config.block_size = 16

    vllm_config.scheduler_config = MagicMock()
    vllm_config.scheduler_config.max_num_batched_tokens = 1024
    vllm_config.scheduler_config.max_num_seqs = 32

    vllm_config.model_config = MagicMock()
    vllm_config.model_config.dtype = torch.float16
    vllm_config.model_config.max_model_len = 2048
    vllm_config.model_config.uses_mrope = False
    vllm_config.model_config.uses_xdrope_dim = 0
    vllm_config.model_config.hf_text_config = MagicMock(spec=[])
    vllm_config.model_config.hf_text_config.to_dict = MagicMock(return_value={})
    vllm_config.model_config.get_hidden_size = MagicMock(return_value=4096)

    vllm_config.compilation_config = MagicMock()

    vllm_config.parallel_config.tensor_parallel_size = 1
    vllm_config.parallel_config.data_parallel_rank = 0
    vllm_config.parallel_config.data_parallel_size = 1
    vllm_config.parallel_config.prefill_context_parallel_size = 1
    vllm_config.parallel_config.enable_expert_parallel = False

    vllm_config.additional_config = None

    init_ascend_config(vllm_config)
    return vllm_config


def test_proposer_initialization():
    """Test that the proposer initializes correctly (matches vllm pattern)."""
    from unittest.mock import MagicMock

    vllm_config = _create_vllm_config(num_speculative_tokens=1, layer_ids=[1, 2, 3, 4])
    device = torch.device("cpu")
    runner = MagicMock()
    runner.pin_memory = False
    runner.pcp_size = 1
    runner.dcp_size = 1

    with set_current_vllm_config(vllm_config):
        proposer = AscendExtractHiddenStatesProposer(vllm_config=vllm_config, device=device, runner=runner)

        # Verify it's an instance of ExtractHiddenStatesProposer
        from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer

        assert isinstance(proposer, ExtractHiddenStatesProposer)
        assert proposer.runner == runner


def test_dummy_run_basic():
    """Test dummy_run with Ascend-specific ACL graph signature.

    This is Ascend-specific because ACL graph capture has different parameters
    than CUDA graph capture.
    """
    from unittest.mock import MagicMock, patch

    vllm_config = _create_vllm_config()
    device = torch.device("cpu")
    runner = MagicMock()
    runner.pin_memory = False
    runner.pcp_size = 1
    runner.dcp_size = 1

    with set_current_vllm_config(vllm_config):
        proposer = AscendExtractHiddenStatesProposer(vllm_config=vllm_config, device=device, runner=runner)

        proposer.model = MagicMock()
        proposer.dp_rank = 0
        proposer.hidden_states = torch.zeros(1024, 4096, dtype=torch.float16)
        runner._sync_metadata_across_dp.return_value = (16, None, CUDAGraphMode.NONE)

        with patch("vllm_ascend.spec_decode.extract_hidden_states_proposer.set_forward_context") as mock_context:
            mock_context.return_value.__enter__ = MagicMock(return_value=None)
            mock_context.return_value.__exit__ = MagicMock(return_value=None)

            proposer.dummy_run(num_tokens=16)
            proposer.model.assert_called_once()


def test_dummy_run_syncs_metadata_across_dp_as_draft_model():
    """dummy_run must issue the same drafter DP sync as propose() does on
    busy ranks (via _determine_batch_execution_and_padding), mirroring
    llm_base_proposer.dummy_run.

    Regression guard for the multi-DP deadlock: if idle DP ranks running the
    dummy path skip the drafter sync while busy ranks perform it, the DP
    cpu_group collectives desynchronize and all ranks hang.
    """
    from unittest.mock import MagicMock, patch

    vllm_config = _create_vllm_config()
    device = torch.device("cpu")
    runner = MagicMock()
    runner.pin_memory = False
    runner.pcp_size = 1
    runner.dcp_size = 1

    with set_current_vllm_config(vllm_config):
        proposer = AscendExtractHiddenStatesProposer(vllm_config=vllm_config, device=device, runner=runner)

        proposer.model = MagicMock()
        proposer.dp_rank = 0
        proposer.hidden_states = torch.zeros(1024, 4096, dtype=torch.float16)

        synced_tensor = torch.tensor([16, 16], dtype=torch.int32)
        runner._sync_metadata_across_dp.return_value = (16, synced_tensor, CUDAGraphMode.NONE)

        with patch("vllm_ascend.spec_decode.extract_hidden_states_proposer.set_forward_context") as mock_context:
            mock_context.return_value.__enter__ = MagicMock(return_value=None)
            mock_context.return_value.__exit__ = MagicMock(return_value=None)

            proposer.dummy_run(num_tokens=16)

        runner._sync_metadata_across_dp.assert_called_once()
        args, kwargs = runner._sync_metadata_across_dp.call_args
        assert (args and args[0] == 16) or kwargs.get("num_tokens") == 16
        assert kwargs.get("is_draft_model") is True
        # The synced tensor must be the one forwarded to set_forward_context.
        _, ctx_kwargs = mock_context.call_args
        assert ctx_kwargs["num_tokens_across_dp"] is synced_tensor


def test_prepare_next_token_ids_padded():
    """Test prepare_next_token_ids_padded (matches vllm's test pattern).

    Since num_speculative_tokens == 1, sampled_token_ids has shape (batch_size, 1).
    For each request we either use the sampled token (if valid and not discarded)
    or a backup token from the request state.

    Note: Ascend uses indices/count pattern instead of GPU's boolean mask.
    """
    from unittest.mock import MagicMock

    device = torch.device("cpu")
    vllm_config = _create_vllm_config()

    runner = MagicMock()
    runner.pin_memory = False
    runner.pcp_size = 1
    runner.dcp_size = 1

    with set_current_vllm_config(vllm_config):
        proposer = AscendExtractHiddenStatesProposer(vllm_config=vllm_config, device=device, runner=runner)

        # Setup test data (same pattern as vllm)
        num_requests = 4
        req_ids = [f"req_{i + 1}" for i in range(num_requests)]

        gpu_input_batch = MockInputBatch(
            num_reqs=num_requests,
            req_ids=req_ids,
            vocab_size=100,
            num_tokens_no_spec=[11, 16, 21, 26],  # Different seq_lens: [10, 15, 20, 25]
        )

        requests = {}
        for req_id in req_ids:
            idx = int(req_id.split("_")[1])
            # Different token sequences for each request
            mock_request = MockCachedRequestState(req_id, list(range(15 + idx * 5)))
            requests[req_id] = mock_request

        # sampled_token_ids shape: [batch_size, 1]
        sampled_token_ids = torch.tensor(
            [
                [1],  # valid, use 1
                [4],  # valid, use 4
                [-1],  # invalid, use backup
                [2],  # discarded, use backup
            ],
            dtype=torch.int64,
            device=device,
        )

        # Ascend uses indices/count pattern (different from GPU's boolean mask)
        discard_request_indices = torch.tensor([3], dtype=torch.int64)
        num_discarded_requests = 1

        next_token_ids, valid_sampled_tokens_count = proposer.prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_indices=discard_request_indices,
            num_discarded_requests=num_discarded_requests,
        )

        # Verify results
        # valid_sampled_tokens_count tracks token validity (not discard status)
        expected_valid_counts = torch.tensor([1, 1, 0, 1], dtype=torch.int32)
        assert torch.equal(valid_sampled_tokens_count, expected_valid_counts)

        # next_token_ids: use sampled if valid and not discarded, else backup
        # Request 1: valid (1), use 1
        # Request 2: valid (4), use 4
        # Request 3: invalid (-1), use backup (seq_len=10, token at pos 10 = 10)
        # Request 4: discarded, use backup (seq_len=15, token at pos 15 = 15)
        assert next_token_ids[0].item() == 1
        assert next_token_ids[1].item() == 4
        assert next_token_ids[2].item() == 20  # backup from request state
        assert next_token_ids[3].item() == 25  # backup from request state

        # Verify return dtypes
        assert next_token_ids.dtype == torch.int32
        assert valid_sampled_tokens_count.dtype == torch.int32


def _make_batch_desc(num_tokens: int):
    """Build a minimal mock for ``CudagraphDispatcher.dispatch``'s batch_desc."""
    from unittest.mock import MagicMock

    batch_desc = MagicMock()
    batch_desc.num_tokens = num_tokens
    return batch_desc


def _build_proposer_for_padding_test(data_parallel_size: int = 1):
    """Shared helper for _determine_batch_execution_and_padding tests.

    Returns a proposer whose ``runner``, ``cudagraph_dispatcher``, and
    DP-related attributes are mocked so we can drive
    ``_determine_batch_execution_and_padding`` without an NPU.
    """
    from unittest.mock import MagicMock

    vllm_config = _create_vllm_config()
    vllm_config.parallel_config.data_parallel_size = data_parallel_size

    runner = MagicMock()
    runner.pin_memory = False
    runner.pcp_size = 1
    runner.dcp_size = 1

    with set_current_vllm_config(vllm_config):
        proposer = AscendExtractHiddenStatesProposer(vllm_config=vllm_config, device=torch.device("cpu"), runner=runner)

    proposer.dp_rank = 0
    proposer.cudagraph_dispatcher = MagicMock()
    return proposer, runner


def test_determine_batch_execution_and_padding_asserts_when_runner_is_none():
    """Constructing without a runner must fail fast with a clear message.

    Regression guard for the AttributeError that would otherwise be raised
    on ``self.runner._pad_for_sequence_parallelism(...)`` at the entry of
    the override.
    """
    vllm_config = _create_vllm_config()

    with set_current_vllm_config(vllm_config):
        proposer = AscendExtractHiddenStatesProposer(vllm_config=vllm_config, device=torch.device("cpu"), runner=None)

    proposer.cudagraph_dispatcher = type("D", (), {"dispatch": staticmethod(lambda *a, **kw: (None, None))})()

    with pytest.raises(AssertionError, match="requires a runner reference"):
        proposer._determine_batch_execution_and_padding(num_tokens=4)


def test_determine_batch_execution_and_padding_dp1_sp_pads_and_skips_sync():
    """With DP=1, SP-pads ``num_tokens`` but never calls DP sync.

    Verifies the ``data_parallel_size == 1`` early-out and that the
    runner's ``_pad_for_sequence_parallelism`` is still consulted so the
    cache_only forward gets an SP-aligned input.
    """
    proposer, runner = _build_proposer_for_padding_test(data_parallel_size=1)

    # Simulate TP=4 SP padding: round 6 up to 8
    runner._pad_for_sequence_parallelism = lambda n: ((n + 3) // 4) * 4
    proposer.cudagraph_dispatcher.dispatch.return_value = (
        CUDAGraphMode.NONE,
        _make_batch_desc(num_tokens=8),
    )

    cudagraph_mode, num_tokens_padded, num_tokens_across_dp = proposer._determine_batch_execution_and_padding(
        num_tokens=6
    )

    assert num_tokens_padded == 8
    assert num_tokens_across_dp is None  # no DP sync when dp_size == 1
    assert cudagraph_mode == CUDAGraphMode.NONE
    # Dispatcher saw the SP-padded value, not the raw 6.
    args, kwargs = proposer.cudagraph_dispatcher.dispatch.call_args
    assert args[0] == 8 or kwargs.get("num_tokens") == 8 or args == (8,)
    runner._sync_metadata_across_dp.assert_not_called()


def test_determine_batch_execution_and_padding_dp2_uses_runner_sync():
    """With DP>1, must call ``runner._sync_metadata_across_dp`` (Ascend's
    shape ``[2, dp_size]`` path) and must NOT call upstream
    ``coordinate_batch_across_dp`` (shape ``[4, dp_size]``).

    This is the core regression test for the gloo
    ``op.preamble.length 8 vs 4`` shape mismatch on the DP cpu_group.
    """
    from unittest.mock import patch

    proposer, runner = _build_proposer_for_padding_test(data_parallel_size=2)

    runner._pad_for_sequence_parallelism = lambda n: ((n + 3) // 4) * 4
    proposer.cudagraph_dispatcher.dispatch.side_effect = [
        # First dispatch (pre-sync) with SP-padded num_tokens=8
        (CUDAGraphMode.NONE, _make_batch_desc(num_tokens=8)),
        # Re-dispatch after sync; the agreed value happens to also be 8
        (CUDAGraphMode.NONE, _make_batch_desc(num_tokens=8)),
    ]
    # Pretend both DP ranks agreed on 8 tokens
    sync_tensor = torch.tensor([8, 8], dtype=torch.int32)
    runner._sync_metadata_across_dp.return_value = (8, sync_tensor, CUDAGraphMode.NONE)

    with patch("vllm.v1.spec_decode.extract_hidden_states.coordinate_batch_across_dp") as mock_upstream_coord:
        cudagraph_mode, num_tokens_padded, num_tokens_across_dp = proposer._determine_batch_execution_and_padding(
            num_tokens=6
        )

    # Upstream DP sync must NOT be used (it would post a [4, dp_size]
    # tensor and break gloo on the shared cpu_group).
    mock_upstream_coord.assert_not_called()
    # Runner sync called once, with the SP-padded value and is_draft_model=True.
    runner._sync_metadata_across_dp.assert_called_once()
    call_kwargs = runner._sync_metadata_across_dp.call_args.kwargs
    assert call_kwargs["num_tokens"] == 8  # SP-padded 6 -> 8
    assert call_kwargs["is_draft_model"] is True

    assert num_tokens_padded == 8
    assert num_tokens_across_dp is not None
    assert num_tokens_across_dp[proposer.dp_rank].item() == 8


def test_determine_batch_execution_and_padding_dp2_keeps_tp_aligned_for_main_forward():
    """If the runner's SP padding produces a TP-aligned value, the final
    ``num_tokens_padded`` returned to the proposer (and downstream main
    forward) is guaranteed to be TP-aligned too. Regression guard for
    the ``reduce_scatter`` assertion ``input.shape[0] % world_size == 0``.
    """
    proposer, runner = _build_proposer_for_padding_test(data_parallel_size=2)

    tp = 4
    runner._pad_for_sequence_parallelism = lambda n: ((n + tp - 1) // tp) * tp
    proposer.cudagraph_dispatcher.dispatch.side_effect = [
        (CUDAGraphMode.NONE, _make_batch_desc(num_tokens=8)),
        (CUDAGraphMode.NONE, _make_batch_desc(num_tokens=8)),
    ]
    runner._sync_metadata_across_dp.return_value = (
        8,
        torch.tensor([8, 8], dtype=torch.int32),
        CUDAGraphMode.NONE,
    )

    _mode, num_tokens_padded, _across = proposer._determine_batch_execution_and_padding(num_tokens=6)

    # The whole point of the fix: never returns 6 (which would crash
    # SP reduce_scatter as 6 % 4 != 0).
    assert num_tokens_padded % tp == 0
