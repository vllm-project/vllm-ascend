# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.models.interfaces import supports_pp

from vllm_ascend.models.deepseek_v4_dspark import (
    DSparkDeepseekV4ForCausalLM,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def test_dspark_deepseek_v4_supports_pipeline_parallelism() -> None:
    assert supports_pp(DSparkDeepseekV4ForCausalLM)


def test_dspark_configures_aux_hidden_states_with_pipeline_parallelism() -> None:
    model_runner = NPUModelRunner.__new__(NPUModelRunner)
    model_runner.speculative_config = SimpleNamespace(use_dspark=lambda: True)

    assert model_runner._uses_aux_hidden_state_outputs_with_pp()


def test_eagle3_respects_aux_hidden_state_config_with_pipeline_parallelism() -> None:
    model_runner = NPUModelRunner.__new__(NPUModelRunner)
    hf_config = SimpleNamespace(eagle_config={"use_aux_hidden_state": False})
    model_runner.speculative_config = SimpleNamespace(
        method="eagle3",
        use_dspark=lambda: False,
        draft_model_config=SimpleNamespace(hf_config=hf_config),
    )

    assert not model_runner._uses_aux_hidden_state_outputs_with_pp()

    hf_config.eagle_config = None
    assert model_runner._uses_aux_hidden_state_outputs_with_pp()
