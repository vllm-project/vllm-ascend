# SPDX-License-Identifier: Apache-2.0
"""Keep the SFA offload backend's bounds aligned with the kernel sources."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import regex as re
import torch

from vllm_ascend.attention.sfa_kv_offload import (
    COPY_SFA_TAIL_BLOCKS,
    COPY_SFA_TAIL_TOKENS,
    LIM_CACHE_BLOCK_SIZE,
    LIM_MAX_HOT_TOKENS,
    LIM_MAX_QUERY_ROWS,
    LIM_TOPK,
    AscendSFAKVOffloadImpl,
    AscendSFAKVOffloadMetadataBuilder,
    _validate_fused_copy_sfa_config,
)
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadataBuilder

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("kernel", ["fused_lightning_indexer_manage", "fused_quant_lightning_indexer_manage"])
@pytest.mark.parametrize(
    "cpp_name,python_value",
    [
        ("TOPK", LIM_TOPK),
        ("MAX_ROUTES", LIM_MAX_QUERY_ROWS),
        ("MAX_CACHE_TOKENS", LIM_MAX_HOT_TOKENS),
        ("CACHE_BLOCK_SIZE", LIM_CACHE_BLOCK_SIZE),
    ],
)
def test_lim_contract_matches_cpp_header(kernel, cpp_name, python_value):
    header = REPO_ROOT / "csrc" / "attention" / kernel / "op_kernel" / f"{kernel}_constants.h"
    # A missing/changed definition fails rather than silently skipping parity.
    matches = re.findall(rf"\bconstexpr\s+uint32_t\s+{cpp_name}\s*=\s*(\d+)U?\s*;", header.read_text(encoding="utf-8"))
    assert len(matches) == 1, f"Missing or ambiguous {cpp_name} in {header}"
    assert int(matches[0]) == python_value, f"Update the shared Python contract for {header}:{cpp_name}"


def test_copy_sfa_alignment_is_derived_from_circular_tail_layout():
    assert COPY_SFA_TAIL_BLOCKS == 2
    assert COPY_SFA_TAIL_TOKENS == COPY_SFA_TAIL_BLOCKS * LIM_CACHE_BLOCK_SIZE
    assert COPY_SFA_TAIL_TOKENS > LIM_CACHE_BLOCK_SIZE
    # Kernel capacity and serving alignment are separate: the largest valid
    # serving budget rounds down to a whole circular-tail period.
    largest_aligned = LIM_MAX_HOT_TOKENS // COPY_SFA_TAIL_TOKENS * COPY_SFA_TAIL_TOKENS
    assert largest_aligned <= LIM_MAX_HOT_TOKENS < largest_aligned + COPY_SFA_TAIL_TOKENS
    assert largest_aligned >= LIM_MAX_QUERY_ROWS * LIM_TOPK


@pytest.mark.parametrize("builder", [False, True])
def test_invalid_fused_config_fails_before_backend_buffer_allocation(builder):
    runtime = SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=8))
    cfg = SimpleNamespace(use_fused_copy_sfa=True, topk=LIM_TOPK, topk_buffer_size=8192)
    parent = AscendSFAMetadataBuilder if builder else AscendSFAImpl
    with (
        patch("vllm_ascend.attention.sfa_kv_offload.get_current_vllm_config", return_value=runtime),
        patch(
            "vllm_ascend.attention.sfa_kv_offload.get_ascend_config",
            return_value=SimpleNamespace(sparse_kv_offload_config=cfg),
        ),
        patch.object(parent, "__init__", return_value=None) as parent_init,
        pytest.raises(ValueError, match="hot budget"),
    ):
        if builder:
            AscendSFAKVOffloadMetadataBuilder(None, [], runtime, torch.device("cpu"))
        else:
            AscendSFAKVOffloadImpl(1, 1, 1.0, 1, None, None, "auto", None, "decoder", None)
    parent_init.assert_not_called()


def test_non_fused_config_does_not_require_fused_kernel_fields():
    _validate_fused_copy_sfa_config(SimpleNamespace(), SimpleNamespace(use_fused_copy_sfa=False))


def test_fused_config_rejects_topk_outside_kernel_contract():
    with pytest.raises(ValueError, match="TopK=2048"):
        _validate_fused_copy_sfa_config(
            SimpleNamespace(speculative_config=None),
            SimpleNamespace(use_fused_copy_sfa=True, topk=1024, topk_buffer_size=4096),
        )
