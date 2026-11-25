# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from vllm.config import VllmConfig, get_current_vllm_config, set_current_vllm_config
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker


@pytest.mark.parametrize("fails", [False, True])
def test_idle_dummy_batch_scopes_and_restores_config(fails):
    config = VllmConfig()
    outer_config = VllmConfig()

    def dummy(num_tokens, **kwargs):
        assert get_current_vllm_config() is config
        assert num_tokens == 2
        assert kwargs == {"uniform_decode": True, "skip_gdn_state_update": True, "preserve_active_loras": True}
        if fails:
            raise RuntimeError("dummy failed")

    worker = SimpleNamespace(
        vllm_config=config,
        log_memory_stats=Mock(),
        model_runner=SimpleNamespace(uniform_decode_query_len=2, decode_query_len=2, _dummy_run=dummy),
    )
    with set_current_vllm_config(outer_config):
        if fails:
            with pytest.raises(RuntimeError, match="dummy failed"):
                NPUWorker.execute_dummy_batch(worker)
        else:
            NPUWorker.execute_dummy_batch(worker)
        assert get_current_vllm_config() is outer_config


@pytest.mark.parametrize("preserve", [True, False])
def test_maybe_dummy_run_with_lora_preserves_active_loras(preserve: bool) -> None:
    runner = object.__new__(NPUModelRunner)
    set_mapping = Mock()
    runner.lora_manager = SimpleNamespace(_adapter_manager=SimpleNamespace(set_adapter_mapping=set_mapping))
    runner._preserve_active_loras_in_dummy_run = preserve
    super_calls: list[tuple[tuple, dict]] = []

    @contextmanager
    def _fake_super(self, *args, **kwargs):
        super_calls.append((args, kwargs))
        yield

    lora_config = object()
    num_scheduled = np.array([2, 3], dtype=np.int32)
    with (
        patch.object(GPUModelRunner, "maybe_dummy_run_with_lora", _fake_super),
        NPUModelRunner.maybe_dummy_run_with_lora(
            runner,
            lora_config,
            num_scheduled,
            None,
            remove_lora=True,
            num_active_loras=0,
            mapping_type=LoRAMappingType.LANGUAGE,
        ),
    ):
        pass

    if preserve:
        assert not super_calls
        set_mapping.assert_called_once()
        mapping = set_mapping.call_args.args[0]
        assert isinstance(mapping, LoRAMapping)
        assert mapping.index_mapping == (0, 0, 0, 0, 0)
        assert mapping.prompt_mapping == (0, 0)
        assert mapping.is_prefill is True
    else:
        assert len(super_calls) == 1
        set_mapping.assert_not_called()
