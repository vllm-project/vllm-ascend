# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspark_speculator


@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("deepseek_v4", "DSparkDeepseekV4ForCausalLM"),
        ("deepseek_v41", "DSparkDeepseekV41ForCausalLM"),
    ],
)
@pytest.mark.parametrize("probabilistic_sampling", [False, True])
def test_full_vocab_draft_load_skips_vocab_remapping(monkeypatch, module_name, class_name, probabilistic_sampling):
    model_cls = getattr(import_module(f"vllm_ascend.models.{module_name}.dspark"), class_name)
    # Avoid model weights and device initialization while keeping the real
    # model class's attribute lookup, which the speculator relies on.
    draft = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(draft)
    draft.model = SimpleNamespace(confidence_head=None)

    speculator = dspark_speculator.DSparkSpeculator.__new__(dspark_speculator.DSparkSpeculator)
    speculator.vllm_config = SimpleNamespace()
    speculator.draft_logits = torch.empty(1, 8) if probabilistic_sampling else None
    speculator.enable_adaptive_verification = False
    speculator._d2t_scatter_index = None
    speculator._draft_scatter_buf = None
    target = torch.nn.Module()
    loader = Mock(return_value=draft)
    monkeypatch.setattr(dspark_speculator, "load_dspark_model", loader)

    loaded = speculator.load_draft_model(target, set())

    loader.assert_called_once_with(target, speculator.vllm_config)
    assert loaded is draft
    assert speculator._d2t_scatter_index is None
    assert speculator._draft_scatter_buf is None
