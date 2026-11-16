# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the SSD control plane (no model/NPU required)."""
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_ascend.models.deepseek_v41.engram.ssd.distributed import ProtocolEpoch


def test_fresh_worker_resets_epoch_and_activation_is_monotonic():
    ProtocolEpoch.reset()
    assert ProtocolEpoch.current() == 0
    assert ProtocolEpoch.activate(2) == 2
    assert ProtocolEpoch.activate(1) == 2
    assert ProtocolEpoch.current() == 2
    ProtocolEpoch.reset()
    assert ProtocolEpoch.current() == 0


@pytest.mark.parametrize('epoch', [0, -1, True, 1.5, '1'])
def test_invalid_epoch_cannot_activate_worker(epoch):
    ProtocolEpoch.reset()
    with pytest.raises(ValueError):
        ProtocolEpoch.activate(epoch)
    assert ProtocolEpoch.current() == 0


def test_inference_epoch_never_accesses_filesystem(monkeypatch):
    monkeypatch.setenv('VLLM_ASCEND_ENGRAM_SSD_EPOCH_FILE', '/unreachable/nfs/epoch')
    ProtocolEpoch.reset()
    with patch.object(Path, 'read_text', side_effect=AssertionError('epoch read')), \
         patch.object(Path, 'write_text', side_effect=AssertionError('epoch write')), \
         patch('builtins.open', side_effect=AssertionError('epoch open')):
        ProtocolEpoch.activate(1)
        assert [ProtocolEpoch.current() for _ in range(1000)] == [1] * 1000


def test_invalid_activation_timeout_is_rejected():
    from vllm_ascend.ascend_config import EngramSSDConfig

    with pytest.raises(ValueError):
        EngramSSDConfig(protocol_activation_timeout_s=0)


@pytest.mark.parametrize('enabled,dp_size', [(False, 2), (True, 1)])
def test_unshared_or_disabled_engines_need_no_group(monkeypatch, enabled, dp_size):
    from types import SimpleNamespace
    from vllm_ascend.models.deepseek_v41.engram.ssd.activation import activate_engine_protocol

    config = SimpleNamespace(engram_ssd=SimpleNamespace(enabled=enabled))
    monkeypatch.setattr('vllm_ascend.ascend_config.get_ascend_config', lambda: config)
    engine = SimpleNamespace(vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=dp_size)))
    # Deliberately no group/executor: these modes must work without either.
    activate_engine_protocol(engine)


def test_startup_hook_finishes_warmup_before_activation_and_propagates_failure(monkeypatch):
    from types import SimpleNamespace
    from vllm.v1.engine.core import EngineCore
    from vllm_ascend.patch.platform.patch_engine_core import _apply_engram_ssd_startup_patch

    order = []

    def warmup(self, config):
        self.vllm_config = config
        order.append('warmup-complete')

    def reject_activation(engine):
        assert engine.vllm_config.tag == 'config'
        assert order == ['warmup-complete']
        raise RuntimeError('peer rejected activation')

    monkeypatch.setattr(EngineCore, '__init__', warmup)
    monkeypatch.setattr('vllm_ascend.models.deepseek_v41.engram.ssd.activation.activate_engine_protocol', reject_activation)
    _apply_engram_ssd_startup_patch()
    _apply_engram_ssd_startup_patch()
    with pytest.raises(RuntimeError, match='peer rejected'):
        EngineCore(SimpleNamespace(tag='config'))
    assert order == ['warmup-complete']
