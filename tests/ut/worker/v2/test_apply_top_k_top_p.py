# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.worker.v2.sample import apply_top_k_top_p as topk_op


def _logits() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def _config(enabled: bool, disable_gate: bool = True):
    return SimpleNamespace(
        enable_triton_apply_topk_topp=enabled,
        triton_topk_topp_disable_gate=disable_gate,
    )


def test_prefers_cann_when_available():
    logits = _logits()
    k = torch.tensor([1, 2])
    p = torch.tensor([0.9, 0.9])
    expected = logits.clone()

    with (
        patch.object(topk_op, "_HAS_CANN_TOP_K_TOP_P", True),
        patch.object(topk_op, "_apply_top_k_top_p_npu", return_value=expected) as cann_op,
        patch.object(topk_op, "get_ascend_config", return_value=_config(True)),
        patch.object(topk_op, "HAS_TRITON", True),
        patch.object(topk_op, "is_triton_apply_topk_topp_supported", return_value=True),
        patch.object(topk_op, "triton_apply_top_k_top_p") as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch") as pytorch_op,
    ):
        out = topk_op.apply_top_k_top_p(logits, k, p)

    cann_op.assert_called_once()
    triton_op.assert_not_called()
    pytorch_op.assert_not_called()
    assert out is expected


def test_uses_triton_when_cann_unavailable():
    logits = _logits()
    k = torch.tensor([1, 2])
    p = torch.tensor([0.9, 0.9])
    triton_out = torch.full_like(logits, float("-inf"))

    with (
        patch.object(topk_op, "_HAS_CANN_TOP_K_TOP_P", False),
        patch.object(topk_op, "get_ascend_config", return_value=_config(True)),
        patch.object(topk_op, "HAS_TRITON", True),
        patch.object(topk_op, "is_triton_apply_topk_topp_supported", return_value=True),
        patch.object(topk_op, "triton_apply_top_k_top_p", return_value=triton_out) as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch") as pytorch_op,
        patch.object(topk_op.logger, "info_once"),
    ):
        out = topk_op.apply_top_k_top_p(logits, k, p)

    # The fallback must match the native op / upstream reference semantics.
    triton_op.assert_called_once_with(logits, k, p)
    pytorch_op.assert_not_called()
    assert torch.equal(out, triton_out)


def test_falls_back_to_pytorch_without_cann_or_triton_switch():
    logits = _logits()
    k = torch.tensor([1, 2])
    expected = logits.clone()

    with (
        patch.object(topk_op, "_HAS_CANN_TOP_K_TOP_P", False),
        patch.object(topk_op, "get_ascend_config", return_value=_config(False)),
        patch.object(topk_op, "HAS_TRITON", True),
        patch.object(topk_op, "triton_apply_top_k_top_p") as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch", return_value=expected) as pytorch_op,
    ):
        out = topk_op.apply_top_k_top_p(logits, k, None)

    pytorch_op.assert_called_once()
    triton_op.assert_not_called()
    assert out is expected


def test_falls_back_to_pytorch_without_triton():
    logits = _logits()
    expected = logits.clone()

    with (
        patch.object(topk_op, "_HAS_CANN_TOP_K_TOP_P", False),
        patch.object(topk_op, "get_ascend_config", return_value=_config(True)),
        patch.object(topk_op, "HAS_TRITON", False),
        patch.object(topk_op, "triton_apply_top_k_top_p") as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch", return_value=expected) as pytorch_op,
    ):
        out = topk_op.apply_top_k_top_p(logits, None, torch.tensor([0.9, 0.9]))

    pytorch_op.assert_called_once()
    triton_op.assert_not_called()
    assert out is expected


def test_falls_back_to_pytorch_below_profitable_gate():
    logits = _logits()
    k = torch.tensor([1, 2])
    expected = logits.clone()

    with (
        patch.object(topk_op, "_HAS_CANN_TOP_K_TOP_P", False),
        patch.object(topk_op, "get_ascend_config", return_value=_config(True, disable_gate=False)),
        patch.object(topk_op, "HAS_TRITON", True),
        patch.object(topk_op, "is_triton_apply_topk_topp_supported", return_value=True),
        patch.object(topk_op, "is_triton_topk_topp_profitable", return_value=False),
        patch.object(topk_op, "triton_apply_top_k_top_p") as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch", return_value=expected) as pytorch_op,
        patch.object(topk_op.logger, "warning_once"),
    ):
        out = topk_op.apply_top_k_top_p(logits, k, None)

    pytorch_op.assert_called_once()
    triton_op.assert_not_called()
    assert out is expected


def test_falls_back_to_pytorch_on_unsupported_triton_device():
    logits = _logits()
    expected = logits.clone()

    with (
        patch.object(topk_op, "_HAS_CANN_TOP_K_TOP_P", False),
        patch.object(topk_op, "get_ascend_config", return_value=_config(True)),
        patch.object(topk_op, "HAS_TRITON", True),
        patch.object(topk_op, "is_triton_apply_topk_topp_supported", return_value=False),
        patch.object(topk_op, "triton_apply_top_k_top_p") as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch", return_value=expected) as pytorch_op,
        patch.object(topk_op.logger, "warning_once"),
    ):
        out = topk_op.apply_top_k_top_p(logits, None, torch.tensor([0.9, 0.9]))

    pytorch_op.assert_called_once()
    triton_op.assert_not_called()
    assert out is expected


def test_accepts_has_topp_only_rows():
    logits = _logits()
    k = torch.tensor([1, 2])
    p = torch.tensor([0.9, 0.9])
    triton_out = torch.full_like(logits, float("-inf"))

    with (
        patch.object(topk_op, "_HAS_CANN_TOP_K_TOP_P", False),
        patch.object(topk_op, "get_ascend_config", return_value=_config(True)),
        patch.object(topk_op, "HAS_TRITON", True),
        patch.object(topk_op, "is_triton_apply_topk_topp_supported", return_value=True),
        patch.object(topk_op, "triton_apply_top_k_top_p", return_value=triton_out) as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch") as pytorch_op,
        patch.object(topk_op.logger, "info_once"),
    ):
        out = topk_op.apply_top_k_top_p(logits, k, p, has_topp_only_rows=False)

    # The flag is accepted on both lanes but not forwarded: the Triton op
    # handles per-row k/p without a split pipeline.
    triton_op.assert_called_once_with(logits, k, p)
    pytorch_op.assert_not_called()
    assert torch.equal(out, triton_out)


def test_returns_logits_untouched_without_k_and_p():
    logits = _logits()

    with (
        patch.object(topk_op, "_HAS_CANN_TOP_K_TOP_P") as cann,
        patch.object(topk_op, "get_ascend_config") as config,
        patch.object(topk_op, "triton_apply_top_k_top_p") as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch") as pytorch_op,
    ):
        out = topk_op.apply_top_k_top_p(logits, None, None)

    cann.assert_not_called()
    config.assert_not_called()
    triton_op.assert_not_called()
    pytorch_op.assert_not_called()
    assert out is logits


def test_triton_profitability_gate_matches_measured_crossover():
    # A5, topk_renorm fallback: large V needs B >= 16, small V B >= 8.
    assert topk_op.is_triton_topk_topp_profitable(16, 151936)
    assert not topk_op.is_triton_topk_topp_profitable(15, 151936)
    assert topk_op.is_triton_topk_topp_profitable(8, 32000)
    assert not topk_op.is_triton_topk_topp_profitable(7, 32000)
    assert topk_op.is_triton_topk_topp_profitable(1, 151936, disable_gate=True)
