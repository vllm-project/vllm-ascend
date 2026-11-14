# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode import adaptive_verification

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.dsa_v41 import (
    AscendDSAV41MetadataBuilder,
)
from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS


@pytest.mark.parametrize("method", ["dspark", "mtp"])
@pytest.mark.parametrize("num_drafts", [5, 7, 15, STATE_RING_ROWS - 1, STATE_RING_ROWS])
def test_v41_adaptive_real_initialization_gate(num_drafts, method):
    # All five role backends inherit the bounded graph capability. Exercise the real
    # upstream checks; only construction of device-side manager buffers is mocked.
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            enable_adaptive_verification=True,
            num_speculative_tokens=num_drafts,
            parallel_drafting=False,
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
    )
    groups = [
        [
            SimpleNamespace(
                backend=getattr(dsa_v41, f"DeepseekV41{role}Backend"),
                layer_names=[role],
                kv_cache_spec=object(),
                get_metadata_builder=lambda index, role=role: getattr(dsa_v41, f"AscendDSAV41{role}MetadataBuilder"),
            )
        ]
        for role in ("SWA", "Attention", "Indexer", "State", "FoldedIndex")
    ]
    kwargs = dict(
        enable_adaptive_verification=True,
        attn_groups=groups,
        req_states=SimpleNamespace(num_speculative_steps=num_drafts),
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        num_bonus_tokens=1,
        max_total_logits=128,
        vllm_config=config,
    )
    with patch.object(adaptive_verification, "AdaptiveVerificationManager") as manager:
        if num_drafts >= STATE_RING_ROWS:
            with pytest.raises(ValueError, match="per-request query lengths vary"):
                adaptive_verification.maybe_create_adaptive_verification_manager(**kwargs)
            manager.assert_not_called()
            return
        assert adaptive_verification.maybe_create_adaptive_verification_manager(**kwargs) is manager.return_value
        manager.assert_called_once()
        kwargs["req_states"].num_speculative_steps += 1
        with pytest.raises(ValueError, match="per-request query lengths vary"):
            adaptive_verification.maybe_create_adaptive_verification_manager(**kwargs)
        assert manager.call_count == 1


@pytest.mark.parametrize("method", ["dspark", "mtp", "dflash"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("num_drafts,expected", [(None, 1), (5, 6), (STATE_RING_ROWS, STATE_RING_ROWS)])
def test_v41_query_bound_is_independent_of_adaptive_policy(method, enabled, num_drafts, expected):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            enable_adaptive_verification=enabled,
            num_speculative_tokens=num_drafts,
            parallel_drafting=False,
        ),
        use_v2_model_runner=True,
    )
    # No parallel_config is needed to report the backend's query-width bound.
    # Unsupported CP combinations are rejected at platform config validation.
    assert AscendDSAV41MetadataBuilder.get_varlen_cudagraph_max_query_len(config, object()) == expected


def test_v41_query_bound_without_speculation():
    config = SimpleNamespace(speculative_config=None)
    assert AscendDSAV41MetadataBuilder.get_varlen_cudagraph_max_query_len(config, object()) == 1
