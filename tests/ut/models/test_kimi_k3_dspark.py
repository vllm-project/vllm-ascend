from types import SimpleNamespace

from vllm_ascend.models.kimi_k3_dspark import K3DSparkForCausalLM, K3DSparkModel


def test_k3_mla_draft_reports_non_causal_attention_per_layer():
    draft_model = SimpleNamespace(config=SimpleNamespace(), layers=[object(), object(), object()])

    assert K3DSparkModel.get_draft_attn_causal(draft_model) == [False, False, False]


def test_k3_mla_draft_uses_checkpoint_attention_causality():
    draft_model = SimpleNamespace(
        config=SimpleNamespace(
            full_attention_causal=True,
            dflash_config={"causal": True},
        ),
        layers=[object(), object(), object()],
    )

    assert K3DSparkModel.get_draft_attn_causal(draft_model) == [True, True, True]


def test_k3_mla_causal_lm_forwards_attention_causality():
    draft_model = SimpleNamespace(get_draft_attn_causal=lambda: [False, False])
    causal_lm = SimpleNamespace(model=draft_model)

    assert K3DSparkForCausalLM.get_draft_attn_causal(causal_lm) == [False, False]
