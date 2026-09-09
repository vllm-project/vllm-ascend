# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Kimi GDN state metadata, not NPU memory protection.

Fault injection below demonstrates that a content/ownership oracle detects
corruption that a block-ID/cardinality check alone cannot detect. The builder
is not expected to validate physical cache contents. Prefix lookup itself is
owned by the scheduler; these tests cover consumption of an already-hit state.
"""

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from tests.ut.ops.test_gdn_attn_builder import (
    BatchSpec,
    _make_builder,
    create_common_attn_metadata,
)
from tests.ut.ops.test_gdn_attn_builder import _no_pin_memory as _no_pin_memory
from tests.ut.ops.test_gdn_attn_builder import _patch_triton_cdiv as _patch_triton_cdiv
from vllm_ascend.ops.gdn_attn_builder import (
    _treat_single_token_prefills_with_state_as_decodes,
)


def _common(lengths, tables):
    common = create_common_attn_metadata(BatchSpec(lengths, [8, 8]), 16, torch.device("cpu"))
    common.block_table_tensor = tables.clone()
    return common


def _builder():
    return _make_builder(
        device=torch.device("cpu"),
        num_heads=32,
        num_speculative_tokens=7,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )


@pytest.mark.parametrize("prefix_tokens", [0, 15, 16, 17, 63, 64, 65, 8191])
def test_prefix_hit_single_remaining_token_preserves_source_metadata(prefix_tokens):
    common = create_common_attn_metadata(BatchSpec([prefix_tokens + 1], [1]), 16, torch.device("cpu"))
    common.is_prefilling.fill_(True)
    common.block_table_tensor.fill_(37)
    result = _treat_single_token_prefills_with_state_as_decodes(common)
    assert result.is_prefilling.tolist() == [prefix_tokens == 0]
    assert common.is_prefilling.tolist() == [True]
    assert result.block_table_tensor.data_ptr() == common.block_table_tensor.data_ptr()
    assert result.block_table_tensor.flatten()[0] == 37
    assert result.num_computed_tokens_cpu.tolist() == [prefix_tokens]


@pytest.mark.parametrize("prefix_tokens", [15, 16, 17, 63, 64, 65])
@pytest.mark.parametrize("accepted", [1, 4, 8])
def test_verification_cross_boundary_refreshes_state_ownership(prefix_tokens, accepted):
    # KDA state block IDs are not token-paged MLA block IDs. Verify state
    # routing when the token positions cross a page/chunk boundary, without
    # incorrectly deriving recurrent state IDs from token position // 16.
    builder = _builder()
    first = torch.arange(16, dtype=torch.int32).reshape(2, 8) + 10
    captured = builder.build(0, _common([8, 8], first), torch.ones(2, dtype=torch.int32), torch.tensor([7, 7]))
    stable = captured.spec_decode_metadata.spec_causal_conv1d
    pointers = (stable.cache_indices.data_ptr(), stable.num_accepted_tokens.data_ptr())
    second = torch.tensor([[52, 41, 63, 37, 55, 49, 33, 61], [28, 19, 24, 31, 22, 18, 27, 30]], dtype=torch.int32)
    common = _common([prefix_tokens + 8, prefix_tokens + 9], second)
    runtime = builder.build(0, common, torch.tensor([accepted, 1], dtype=torch.int32), torch.tensor([7, 7]))
    actual = runtime.spec_decode_metadata.spec_causal_conv1d
    assert pointers == (actual.cache_indices.data_ptr(), actual.num_accepted_tokens.data_ptr())
    torch.testing.assert_close(actual.cache_indices, second)
    assert actual.num_accepted_tokens.tolist() == [accepted, 1]
    assert actual.query_start_loc.tolist() == [0, 8, 16]
    # Distinct physical contents make swapped rows and stale mappings visible.
    state = torch.arange(128 * 3).reshape(128, 3)
    torch.testing.assert_close(state[actual.cache_indices.long()], state[second.long()])
    assert not torch.equal(actual.cache_indices, first)
    torch.testing.assert_close(common.block_table_tensor, second)


@pytest.mark.parametrize(
    "fault", ["stale_ids", "swapped_requests", "alias_requests", "payload", "nan", "neighbor_write"]
)
def test_physical_state_fault_injection_is_detected_by_content_oracle(fault):
    builder = _builder()
    expected_ids = torch.arange(16, dtype=torch.int32).reshape(2, 8) + 10
    runtime = builder.build(0, _common([71, 72], expected_ids), torch.ones(2, dtype=torch.int32), torch.tensor([7, 7]))
    actual_ids = runtime.spec_decode_metadata.spec_causal_conv1d.cache_indices.clone()
    cache = torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4)
    expected = cache[expected_ids.long()].clone()
    canary = cache[26].clone()
    if fault == "stale_ids":
        actual_ids[0] += 16
    elif fault == "swapped_requests":
        actual_ids = actual_ids.flip(0)
    elif fault == "alias_requests":
        actual_ids[1] = actual_ids[0]
    elif fault == "payload":
        cache[10, 0] += 1000
    elif fault == "nan":
        cache[10, 0] = float("nan")
    else:
        cache[26, 0] += 1000
    # Clean metadata passed through the real builder. Inject only afterwards:
    # these are negative controls for the oracle, not runtime error handlers.
    with pytest.raises(AssertionError):
        torch.testing.assert_close(cache[actual_ids.long()], expected, equal_nan=False)
        torch.testing.assert_close(cache[26], canary)
