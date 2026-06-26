from types import SimpleNamespace

import pytest
import torch


def test_gpt_oss_other_loader_remaps_routed_expert_weight_names(monkeypatch):
    pytest.importorskip("vllm.model_executor.models.gpt_oss")

    from vllm.model_executor.models.gpt_oss import GptOssModel

    from vllm_ascend.patch.worker import patch_gpt_oss

    routed_name = "layers.0.mlp.experts.routed_experts.w2_weight"
    legacy_name = "layers.0.mlp.experts.w2_weight"
    seen_names: list[str] = []

    def fake_original_load_weights_other(
        self,
        ep_rank_end,
        ep_rank_start,
        heads_per_rank,
        head_start,
        weights,
        stacked_params_mapping,
    ):
        seen_names.extend(name for name, _ in weights)
        return set(seen_names)

    # The patch stores the original on the class (_ascend_original_load_weights_other)
    # and the wrapper calls it from there, so override that class attribute.
    monkeypatch.setattr(GptOssModel, "_ascend_original_load_weights_other", fake_original_load_weights_other)

    model = SimpleNamespace(
        named_parameters=lambda: [(routed_name, torch.nn.Parameter(torch.empty(1)))],
    )

    loaded_params = patch_gpt_oss._patched_load_weights_other(
        model,
        ep_rank_end=1,
        ep_rank_start=0,
        heads_per_rank=1,
        head_start=0,
        weights=[(legacy_name, torch.empty(1))],
        stacked_params_mapping=[],
    )

    assert loaded_params == {routed_name}


def test_gpt_oss_other_loader_does_not_double_remap_routed_names():
    pytest.importorskip("vllm.model_executor.models.gpt_oss")

    from vllm_ascend.patch.worker.patch_gpt_oss import _remap_gpt_oss_routed_expert_weight_names

    routed_name = "layers.0.mlp.experts.routed_experts.w13_weight"
    model = SimpleNamespace(
        named_parameters=lambda: [(routed_name, torch.nn.Parameter(torch.empty(1)))],
    )

    remapped = list(_remap_gpt_oss_routed_expert_weight_names(model, [(routed_name, torch.empty(1))]))

    assert remapped[0][0] == routed_name
