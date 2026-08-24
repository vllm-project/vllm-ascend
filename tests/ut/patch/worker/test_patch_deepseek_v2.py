# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM

from vllm_ascend.models.deepseek_v2 import (
    AscendDeepseekV2ForCausalLM,
    AscendDeepseekV2Model,
    AscendMistralLarge3ForCausalLM,
    get_indexer_init_pattern,
    should_skip_indexer_init,
)


def _config(**overrides) -> SimpleNamespace:
    values = {"num_hidden_layers": 80}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_glm51_skip_topk_keeps_per_layer_indexer():
    assert not should_skip_indexer_init(
        _config(),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_glm52_shared_layer_skips_indexer_init():
    assert should_skip_indexer_init(
        _config(indexer_types=["full", "full", "shared"]),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_mtp_layer_keeps_indexer():
    indexer_types = ["full"] * 80 + ["shared"]
    assert not should_skip_indexer_init(
        _config(indexer_types=indexer_types),
        "model.layers.80.self_attn",
        skip_topk=True,
    )


def test_glm51_init_pattern_keeps_indexers_and_restores_runtime_skip():
    runtime_skip, init_pattern = get_indexer_init_pattern(_config(index_topk_freq=2, index_skip_topk_offset=1))

    assert runtime_skip[:4] == [False, True, False, True]
    assert set(init_pattern) == {"F"}


def test_glm52_init_pattern_only_omits_shared_indexers():
    runtime_skip, init_pattern = get_indexer_init_pattern(
        _config(
            index_topk_pattern=["F", "S", "S"],
            indexer_types=["full", "full", "shared"],
        )
    )

    assert runtime_skip[:3] == [False, True, True]
    assert init_pattern[:3] == ["F", "F", "S"]


def test_ascend_model_uses_inheritance_instead_of_global_patch():
    assert issubclass(AscendDeepseekV2Model, DeepseekV2ForCausalLM.model_cls)
    assert AscendDeepseekV2ForCausalLM.model_cls is AscendDeepseekV2Model
    assert DeepseekV2ForCausalLM.model_cls is not AscendDeepseekV2Model
    assert AscendMistralLarge3ForCausalLM.model_cls is AscendDeepseekV2Model
