#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""V2 model-state wiring for the Qwen4Exp (Qwen3.8-Flash-Next) wrappers."""

from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState

from vllm_ascend.models.qwen4_exp.model import (
    AscendQwen4ExpForCausalLM,
    AscendQwen4ExpForConditionalGeneration,
)
from vllm_ascend.models.qwen4_exp.model_state import AscendQwen4ExpModelState
from vllm_ascend.models.qwen4_exp.qsa import AscendQSAImpl
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


def test_model_state_mro_keeps_upstream_ple_and_ascend_attn():
    """Order invariants, not a fixed tuple: main may insert extra bases
    between ours without breaking this contract."""
    mro = AscendQwen4ExpModelState.__mro__
    order = (
        AscendQwen4ExpModelState,
        Qwen4ExpModelState,
        AscendMambaHybridModelState,
    )
    indices = [mro.index(cls) for cls in order]
    assert indices == sorted(indices), f"MRO order broken: {mro}"
    # Upstream keeps the PLE n-gram buffers; Ascend owns attention metadata
    # construction so FULL aclgraph replay reads model_state.attn_metadata.
    assert AscendQwen4ExpModelState.prepare_attn is AscendMambaHybridModelState.prepare_attn


def test_wrappers_expose_ascend_model_state_cls():
    assert AscendQwen4ExpForCausalLM.get_model_state_cls() is AscendQwen4ExpModelState
    assert AscendQwen4ExpForConditionalGeneration.get_model_state_cls() is AscendQwen4ExpModelState


def test_qsa_update_graph_params_is_a_documented_noop():
    """QSA replay parameters live in persistent buffers rewritten per step;
    the V2 replay hook must rebind nothing and touch no argument."""
    update_stream = object()
    forward_context = object()
    num_tokens = object()
    vllm_config = object()
    speculative_config = object()
    draft_attn_metadatas = object()

    result = AscendQSAImpl.update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=speculative_config,
        draft_attn_metadatas=draft_attn_metadatas,
    )

    assert result is None
