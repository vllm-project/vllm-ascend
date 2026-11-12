# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.config import VllmConfig, get_current_vllm_config, set_current_vllm_config

from vllm_ascend.worker.worker import NPUWorker


@pytest.mark.parametrize("fails", [False, True])
def test_idle_dummy_batch_scopes_and_restores_config(fails):
    config = VllmConfig()
    outer_config = VllmConfig()

    def dummy(num_tokens, **kwargs):
        assert get_current_vllm_config() is config
        assert num_tokens == 2
        assert kwargs == {"uniform_decode": True, "skip_gdn_state_update": True}
        if fails:
            raise RuntimeError("dummy failed")

    worker = SimpleNamespace(
        vllm_config=config,
        log_memory_stats=Mock(),
        model_runner=SimpleNamespace(uniform_decode_query_len=2, _dummy_run=dummy),
    )
    with set_current_vllm_config(outer_config):
        if fails:
            with pytest.raises(RuntimeError, match="dummy failed"):
                NPUWorker.execute_dummy_batch(worker)
        else:
            NPUWorker.execute_dummy_batch(worker)
        assert get_current_vllm_config() is outer_config
