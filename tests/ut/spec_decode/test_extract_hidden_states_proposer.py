# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AscendExtractHiddenStatesProposer.

This test file follows the pattern from vllm's test_extract_hidden_states.py,
with Ascend-specific additions for ACL graph differences.
"""

from __future__ import annotations

import numpy as np
import torch
from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.spec_decode.extract_hidden_states_proposer import (
    AscendExtractHiddenStatesProposer,
)


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

        with patch("vllm_ascend.spec_decode.extract_hidden_states_proposer.set_forward_context") as mock_context:
            mock_context.return_value.__enter__ = MagicMock(return_value=None)
            mock_context.return_value.__exit__ = MagicMock(return_value=None)

            proposer.dummy_run(num_tokens=16)
            proposer.model.assert_called_once()


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
