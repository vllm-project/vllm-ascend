# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3Next MoE shared-expert gate quantization patch."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.patch.worker import patch_qwen3_5

_QWEN3_NEXT_MODULE = "vllm.model_executor.models.qwen3_next"
_GATE_SUFFIX = ".shared_expert_gate"
_PREFIX = "model.layers.0.mlp"


class _RecordingReplicatedLinear:
    """Stand-in for ReplicatedLinear that records the constructor kwargs."""

    calls: dict[str, dict] = {}

    def __init__(self, *args, **kwargs):
        del args
        _RecordingReplicatedLinear.calls[kwargs.get("prefix", "")] = kwargs


def _run_block_init(vllm_config, prefix: str = _PREFIX) -> dict[str, dict]:
    """Run the patched block constructor with a recording ReplicatedLinear."""
    qwen3_next = importlib.import_module(_QWEN3_NEXT_MODULE)

    def fake_block_init(self, vllm_config, prefix: str = "", reduce_results: bool = True):
        del self, vllm_config, reduce_results
        # Mirrors upstream: the gate is the only ReplicatedLinear of the block.
        qwen3_next.ReplicatedLinear(8, 1, bias=False, quant_config=None, prefix=f"{prefix}{_GATE_SUFFIX}")
        qwen3_next.ReplicatedLinear(8, 8, bias=False, quant_config=None, prefix=f"{prefix}.gate")

    _RecordingReplicatedLinear.calls = {}
    with (
        patch.object(patch_qwen3_5, "_QWEN3_NEXT_MOE_ORIGINAL_INIT", fake_block_init),
        patch(f"{_QWEN3_NEXT_MODULE}.ReplicatedLinear", _RecordingReplicatedLinear),
    ):
        patch_qwen3_5._qwen3_next_moe_init_with_quantized_gate(object.__new__(object), vllm_config, prefix=prefix)
    return _RecordingReplicatedLinear.calls


def test_shared_expert_gate_uses_block_quant_config():
    quant_config = object()
    calls = _run_block_init(SimpleNamespace(quant_config=quant_config))

    assert calls[f"{_PREFIX}{_GATE_SUFFIX}"]["quant_config"] is quant_config
    # Every other linear keeps the quant_config it was created with upstream.
    assert calls[f"{_PREFIX}.gate"]["quant_config"] is None


def test_shared_expert_gate_stays_unquantized_without_quant_config():
    calls = _run_block_init(SimpleNamespace(quant_config=None))

    assert calls[f"{_PREFIX}{_GATE_SUFFIX}"]["quant_config"] is None


def test_shared_expert_gate_stays_unquantized_without_vllm_config():
    calls = _run_block_init(None)

    assert calls[f"{_PREFIX}{_GATE_SUFFIX}"]["quant_config"] is None


def test_replicated_linear_is_restored_after_block_init():
    qwen3_next = importlib.import_module(_QWEN3_NEXT_MODULE)
    sentinel = type("SentinelReplicatedLinear", (), {})

    with (
        patch.object(patch_qwen3_5, "_QWEN3_NEXT_MOE_ORIGINAL_INIT", lambda *args, **kwargs: None),
        patch.object(qwen3_next, "ReplicatedLinear", sentinel),
    ):
        patch_qwen3_5._qwen3_next_moe_init_with_quantized_gate(
            object.__new__(object), SimpleNamespace(quant_config=object()), prefix=_PREFIX
        )
        assert qwen3_next.ReplicatedLinear is sentinel
