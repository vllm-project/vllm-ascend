# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_ascend.models.glm54_loop import (
    LOOP_END,
    LOOP_START,
    AscendGlmMoeDsaForCausalLM,
    Glm54LoopProfileForCausalLM,
    _merge_attention,
    _window_indices,
)


def test_window_is_causal_and_restarts_for_each_request():
    positions = torch.tensor([0, 1, 127, 128, 0, 6])
    result = _window_indices(positions, torch.arange(4))
    assert result.squeeze(1).tolist() == [
        [0, -1, -1, -1],
        [0, 1, -1, -1],
        [124, 125, 126, 127],
        [125, 126, 127, 128],
        [0, -1, -1, -1],
        [3, 4, 5, 6],
    ]


def test_two_bank_merge_matches_joint_softmax():
    generator = torch.Generator().manual_seed(123)
    scores = torch.randn(5, 2, 7, generator=generator)
    values = torch.randn(5, 2, 7, 3, generator=generator)

    def attend(s, v):
        m = s.amax(dim=-1, keepdim=True)
        z = (s - m).exp().sum(dim=-1, keepdim=True)
        return (s.softmax(-1).unsqueeze(-1) * v).sum(-2), m.permute(2, 0, 1), z.permute(2, 0, 1)

    actual = _merge_attention(attend(scores[..., :4], values[:, :, :4]), attend(scores[..., 4:], values[:, :, 4:]))
    torch.testing.assert_close(actual, attend(scores, values)[0])


def test_replay_reuses_layers_and_indices_with_distinct_kv(monkeypatch):
    calls = []
    registry: dict[str, SimpleNamespace] = {}

    class Layer(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx
            impl = SimpleNamespace(
                skip_topk=idx > 2 and (idx - 2) % 4 != 0, topk_indices_buffer=torch.empty(2, 4), use_index_cache=True
            )
            self.self_attn = SimpleNamespace(
                mla_attn=SimpleNamespace(mla_attn=SimpleNamespace(impl=impl, layer_name=f"layers.{idx}.attn"))
            )

        def forward(self, positions, hidden, residual):
            cache = self.self_attn.mla_attn.mla_attn
            calls.append((self.idx, cache.layer_name, cache.impl.skip_topk, id(cache.impl.topk_indices_buffer)))
            return hidden + 1, torch.zeros_like(hidden) if residual is None else residual

    def init(self, **kwargs):
        nn.Module.__init__(self)
        self.model = SimpleNamespace(layers=[Layer(i) for i in range(78)], norm=lambda h, r: (h + r, None))

    monkeypatch.setattr(AscendGlmMoeDsaForCausalLM, "__init__", init)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enforce_eager=True,
            dtype=torch.float32,
            hf_text_config=SimpleNamespace(model_type="glm_moe_dsa", num_hidden_layers=78, hidden_size=4),
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        compilation_config=SimpleNamespace(
            static_forward_context=registry,
            static_all_moe_layers=[f"model.layers.{i}.mlp.experts" for i in range(3, 78)],
        ),
        speculative_config=None,
        kv_transfer_config=None,
        attention_config=SimpleNamespace(hisparse_config=None),
    )
    model = Glm54LoopProfileForCausalLM(vllm_config=config)
    actual = model(None, torch.arange(2), inputs_embeds=torch.zeros(2, 4))
    assert len(registry) == LOOP_END - LOOP_START
    assert config.compilation_config.static_all_moe_layers == [
        f"model.layers.{i}.mlp.experts" for i in (*range(3, LOOP_END), *range(LOOP_START, 78))
    ]
    assert [c[0] for c in calls] == [*range(LOOP_END), *range(LOOP_START, 78)]
    for first, second in zip(calls[LOOP_START:LOOP_END], calls[LOOP_END:], strict=False):
        assert first[1] != second[1] and second[1] in registry
        assert second[2] and first[3] == second[3]
    assert len({c[3] for c in calls[LOOP_START:LOOP_END]}) == 7
    torch.testing.assert_close(actual, torch.full_like(actual, 78))
    config.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="eager"):
        Glm54LoopProfileForCausalLM(vllm_config=config)
