# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.sequence import IntermediateTensors

from vllm_ascend.worker.v2 import pp_utils


@pytest.mark.parametrize(
    "method,architecture,needs_aux",
    [
        ("mtp", "Qwen3_5ForConditionalGeneration", False),
        ("eagle3", "MiniMaxM3SparseForCausalLM", True),
        ("eagle3", "MiniMaxM3SparseForConditionalGeneration", True),
        ("dspark", "DeepseekV4ForCausalLM", True),
        ("dspark", "GlmMoeDsaForCausalLM", True),
    ],
)
@pytest.mark.parametrize("pp_size", [1, 2])
def test_resolve_spec_pp_support(method, architecture, needs_aux, pp_size):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size),
        model_config=SimpleNamespace(architecture=architecture),
        speculative_config=SimpleNamespace(method=method),
    )
    support = pp_utils.resolve_spec_pp_support(config)
    if pp_size == 1:
        assert support is None
    else:
        assert support.needs_aux_hidden_states is needs_aux


@pytest.mark.parametrize("method", [None, "ngram", "eagle3"])
def test_unregistered_spec_pp_support(method):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        model_config=SimpleNamespace(architecture="OtherModel"),
        speculative_config=SimpleNamespace(method=method) if method else None,
    )
    assert pp_utils.resolve_spec_pp_support(config) is None


def test_aux_buffer_includes_stage_boundary():
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=4), start_layer=2, aux_hidden_state_layers=(0, 2, 4))
    factory = pp_utils.make_empty_intermediate_tensors(
        model, lambda batch, dtype, device: IntermediateTensors({"hidden_states": torch.zeros(batch, 4)})
    )
    tensors = factory(3, torch.float32, torch.device("cpu"))
    aux = pp_utils.get_pp_transport_tensors(tensors, pp_utils.PPTransportDataType.AUX_HIDDEN_STATES)
    assert len(aux) == 2
    assert all(t.shape == (3, 4) for t in aux)
