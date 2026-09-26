# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Runtime tensors the level-2 sleep cycle must not lose.

``CaMemAllocator.sleep()`` discards every allocation taken from the sleep-managed
pool, and the worker's level-2 backup is built from ``model.named_buffers()``
only. A tensor a forward reads but that is held as a *plain attribute* is
therefore invisible to the backup and comes back from a wake as zeros: a
weightless ``RMSNorm`` weight multiplied by zeros collapses its activations, so
the live weight-update lane can no longer reproduce a server that loaded the
same payload at startup. These tests pin the ownership that prevents it.

The end-to-end effect is covered by
``tests/e2e/pull_request/one_card/rlhf/state_transitions/test_npu_ipc_weight_transfer.py``.
"""

import contextlib
from unittest.mock import MagicMock

import torch
from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_ascend.models.layer.attention.layer import _own_quant_scales_as_non_persistent
from vllm_ascend.utils import own_as_non_persistent_buffer

QUANT_SCALE_NAMES = ("_k_scale", "_v_scale", "_q_scale", "_prob_scale")


@contextlib.contextmanager
def _weightless_norm(hidden_size: int):
    """Build ``RMSNorm(has_weight=False)``; it is a CustomOp, so it needs a config."""
    config = MagicMock()
    # CustomOp.default_on() requires exactly one base mode.
    config.compilation_config.custom_ops = ["all"]
    with set_current_vllm_config(config):
        yield RMSNorm(hidden_size, has_weight=False)


def _level2_backup(model: nn.Module) -> dict[str, torch.Tensor]:
    """The level-2 buffer backup exactly as ``NPUWorker.sleep`` builds it."""
    return {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}


def _level2_restore(model: nn.Module, saved: dict[str, torch.Tensor]) -> None:
    """The level-2 buffer restore exactly as ``NPUWorker.wake_up`` applies it."""
    for name, buffer in model.named_buffers():
        if name in saved:
            buffer.data.copy_(saved[name].data)


def test_weightless_norm_weight_is_owned_as_a_non_persistent_buffer():
    with _weightless_norm(64) as norm:
        # Precondition: upstream keeps the weightless weight as a plain tensor,
        # so neither named_buffers() nor a later "is it registered?" check sees it.
        assert "weight" not in norm._buffers
        assert isinstance(norm.weight, torch.Tensor)
        assert not isinstance(norm.weight, nn.Parameter)

        assert own_as_non_persistent_buffer(norm, "weight") is True

        assert "weight" in norm._buffers
        assert norm.weight in list(norm.buffers())
        # A dummy-weight start must not seed it: --load-format dummy feeds
        # state_dict() through initialize_dummy_weights.
        assert "weight" not in norm.state_dict()
        # A level-2 sleep must back it up.
        assert "weight" in _level2_backup(norm)


def test_owned_weight_survives_a_level2_sleep_restore_round_trip():
    with _weightless_norm(8) as norm:
        own_as_non_persistent_buffer(norm, "weight")
        saved = _level2_backup(norm)

        # What a level-2 wake leaves behind for an unowned tensor.
        with torch.no_grad():
            norm.weight.zero_()
        assert not torch.any(norm.weight)

        _level2_restore(norm, saved)

        torch.testing.assert_close(norm.weight, torch.ones(8))


def test_own_as_non_persistent_buffer_reports_what_it_changed():
    with _weightless_norm(16) as norm:
        assert own_as_non_persistent_buffer(norm, "weight") is True
        # Already a buffer now, so there is nothing left to adopt.
        assert own_as_non_persistent_buffer(norm, "weight") is False
        # Missing attributes and non-tensor attributes are not adopted.
        assert own_as_non_persistent_buffer(norm, "missing") is False
        assert own_as_non_persistent_buffer(norm, "variance_epsilon") is False
        # A weightless norm still keeps its weightless semantics.
        assert norm.pass_weight is False
        assert norm.has_weight is False


def test_attention_quant_scales_are_owned_as_non_persistent_buffers():
    layer = nn.Module()
    for name in QUANT_SCALE_NAMES:
        layer.register_buffer(name, torch.tensor(1.0, dtype=torch.float32))

    # Precondition: upstream registers them persistently, so a dummy-weight
    # start would overwrite them with the shared dummy value.
    assert set(layer.state_dict()) == set(QUANT_SCALE_NAMES)

    _own_quant_scales_as_non_persistent(layer)

    assert set(layer.state_dict()) == set()
    assert {name for name, _ in layer.named_buffers()} == set(QUANT_SCALE_NAMES)
    for name in QUANT_SCALE_NAMES:
        assert float(getattr(layer, name)) == 1.0
        assert name in _level2_backup(layer)


def test_attention_quant_scales_helper_ignores_missing_scales():
    layer = nn.Module()

    _own_quant_scales_as_non_persistent(layer)

    assert list(layer.named_buffers()) == []
