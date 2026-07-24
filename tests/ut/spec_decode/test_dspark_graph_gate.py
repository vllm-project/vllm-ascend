from types import SimpleNamespace

from vllm_ascend.spec_decode.llm_base_proposer import _is_dspark_draft_model


def _model_config(**hf_fields):
    return SimpleNamespace(hf_config=SimpleNamespace(**hf_fields))


def test_dspark_draft_model_detects_native_architecture():
    config = _model_config(architectures=["Qwen3DSparkModel"])

    assert _is_dspark_draft_model(config)


def test_dspark_draft_model_detects_speculators_metadata():
    config = _model_config(
        architectures=["UnknownDraftModel"],
        speculators_model_type="dspark",
    )

    assert _is_dspark_draft_model(config)


def test_dspark_draft_model_rejects_other_drafters():
    config = _model_config(architectures=["Eagle3LlamaForCausalLM"])

    assert not _is_dspark_draft_model(config)
