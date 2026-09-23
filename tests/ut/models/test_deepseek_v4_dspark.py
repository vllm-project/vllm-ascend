# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.models.deepseek_v4_dspark import DeepseekV4DSparkModel


def test_context_kv_reuses_one_filtered_rope_lookup_for_all_draft_layers():
    layers = {
        str(idx): SimpleNamespace(
            self_attn=SimpleNamespace(
                rotary_emb=SimpleNamespace(layername=f"mtp.{idx}.self_attn.attn"),
            )
        )
        for idx in range(3)
    }
    projected_kv = [torch.full((2, 1, 4), idx) for idx in range(3)]
    model = SimpleNamespace(
        layers=layers,
        _project_shared_kv=MagicMock(side_effect=projected_kv),
        _store_standard_swa_kv=MagicMock(),
    )
    context_states = torch.ones((2, 4))
    context_positions = torch.tensor([7, 8])
    slot_mapping = [torch.tensor([idx]) for idx in range(3)]
    rope = (object(), object())

    with patch("vllm_ascend.models.deepseek_v4_dspark.get_cos_and_sin_dsa", return_value=rope) as get_rope:
        DeepseekV4DSparkModel.precompute_and_store_context_kv(
            model,
            context_states,
            context_positions,
            slot_mapping,
        )

    get_rope.assert_called_once_with(context_positions, layer_names="mtp.0.self_attn.attn")
    assert model._project_shared_kv.call_count == 3
    assert all(call.kwargs["rope"] is rope for call in model._project_shared_kv.call_args_list)
    assert model._store_standard_swa_kv.call_count == 3
