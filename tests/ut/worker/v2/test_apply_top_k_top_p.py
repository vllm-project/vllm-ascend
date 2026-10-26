# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.worker.v2.sample import apply_top_k_top_p as v2_apply_top_k_top_p


def _logits() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_uses_triton_op_when_enabled():
    logits = _logits()
    k = torch.tensor([1, 2])
    p = torch.tensor([0.9, 0.9])
    triton_out = torch.full_like(logits, float("-inf"))

    with (
        patch.object(
            v2_apply_top_k_top_p,
            "get_ascend_config",
            return_value=SimpleNamespace(enable_triton_apply_topk_topp=True),
        ),
        patch.object(v2_apply_top_k_top_p, "HAS_TRITON", True),
        patch.object(v2_apply_top_k_top_p, "triton_apply_top_k_top_p", return_value=triton_out) as triton_op,
        patch.object(v2_apply_top_k_top_p, "_apply_top_k_top_p_pytorch") as pytorch_op,
        patch.object(v2_apply_top_k_top_p.logger, "info_once"),
    ):
        out = v2_apply_top_k_top_p.apply_top_k_top_p_npu(logits, k, p)

    triton_op.assert_called_once_with(logits, k, p)
    pytorch_op.assert_not_called()
    assert torch.equal(out, triton_out)


def test_falls_back_to_pytorch_when_disabled():
    logits = _logits()
    k = torch.tensor([1, 2])
    expected = logits.clone()

    with (
        patch.object(
            v2_apply_top_k_top_p,
            "get_ascend_config",
            return_value=SimpleNamespace(enable_triton_apply_topk_topp=False),
        ),
        patch.object(v2_apply_top_k_top_p, "HAS_TRITON", True),
        patch.object(v2_apply_top_k_top_p, "triton_apply_top_k_top_p") as triton_op,
        patch.object(v2_apply_top_k_top_p, "_apply_top_k_top_p_pytorch", return_value=expected) as pytorch_op,
    ):
        out = v2_apply_top_k_top_p.apply_top_k_top_p_npu(logits, k, None)

    pytorch_op.assert_called_once_with(logits, k, None)
    triton_op.assert_not_called()
    assert out is expected


def test_falls_back_to_pytorch_without_triton():
    logits = _logits()
    expected = logits.clone()

    with (
        patch.object(
            v2_apply_top_k_top_p,
            "get_ascend_config",
            return_value=SimpleNamespace(enable_triton_apply_topk_topp=True),
        ),
        patch.object(v2_apply_top_k_top_p, "HAS_TRITON", False),
        patch.object(v2_apply_top_k_top_p, "triton_apply_top_k_top_p") as triton_op,
        patch.object(v2_apply_top_k_top_p, "_apply_top_k_top_p_pytorch", return_value=expected) as pytorch_op,
    ):
        out = v2_apply_top_k_top_p.apply_top_k_top_p_npu(logits, None, torch.tensor([0.9, 0.9]))

    pytorch_op.assert_called_once()
    triton_op.assert_not_called()
    assert out is expected


@pytest.mark.skipif(
    "has_topp_only_rows" not in inspect.signature(v2_apply_top_k_top_p.apply_top_k_top_p_npu).parameters,
    reason="has_topp_only_rows exists only on the vLLM main lane",
)
def test_accepts_has_topp_only_rows_on_main_lane():
    logits = _logits()
    k = torch.tensor([1, 2])
    p = torch.tensor([0.9, 0.9])
    triton_out = torch.full_like(logits, float("-inf"))

    with (
        patch.object(
            v2_apply_top_k_top_p,
            "get_ascend_config",
            return_value=SimpleNamespace(enable_triton_apply_topk_topp=True),
        ),
        patch.object(v2_apply_top_k_top_p, "HAS_TRITON", True),
        patch.object(v2_apply_top_k_top_p, "triton_apply_top_k_top_p", return_value=triton_out) as triton_op,
        patch.object(v2_apply_top_k_top_p, "_apply_top_k_top_p_pytorch") as pytorch_op,
        patch.object(v2_apply_top_k_top_p.logger, "info_once"),
    ):
        out = v2_apply_top_k_top_p.apply_top_k_top_p_npu(logits, k, p, has_topp_only_rows=False)

    # The hint is accepted (newer vLLM main passes it) but not forwarded:
    # the Triton op handles per-row k/p without a split pipeline.
    triton_op.assert_called_once_with(logits, k, p)
    pytorch_op.assert_not_called()
    assert torch.equal(out, triton_out)


def test_returns_logits_untouched_without_k_and_p():
    logits = _logits()

    with (
        patch.object(v2_apply_top_k_top_p, "get_ascend_config") as config,
        patch.object(v2_apply_top_k_top_p, "triton_apply_top_k_top_p") as triton_op,
        patch.object(v2_apply_top_k_top_p, "_apply_top_k_top_p_pytorch") as pytorch_op,
    ):
        out = v2_apply_top_k_top_p.apply_top_k_top_p_npu(logits, None, None)

    config.assert_not_called()
    triton_op.assert_not_called()
    pytorch_op.assert_not_called()
    assert out is logits
