from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend import spec_decode


def _speculative_config(
    *,
    model_architectures=(),
    text_architectures=(),
    text_model_type="qwen4_exp_text",
):
    return SimpleNamespace(
        draft_model_config=SimpleNamespace(
            architectures=model_architectures,
            hf_text_config=SimpleNamespace(
                architectures=text_architectures,
                model_type=text_model_type,
            ),
        ),
        use_gemma4_mtp=lambda: False,
        use_step3p5_mtp=lambda: False,
    )


def test_qwen4_exp_mtp_detects_model_level_architecture():
    config = _speculative_config(
        model_architectures=["Qwen4ExpMTP"],
        text_architectures=None,
    )

    assert spec_decode._use_qwen4_exp_mtp(config)


@pytest.mark.parametrize(
    ("text_architectures", "text_model_type"),
    [(["Qwen4ExpMTP"], "qwen4_exp_text"), (None, "qwen4_exp_mtp")],
)
def test_qwen4_exp_mtp_preserves_text_config_fallbacks(text_architectures, text_model_type):
    config = _speculative_config(
        text_architectures=text_architectures,
        text_model_type=text_model_type,
    )

    assert spec_decode._use_qwen4_exp_mtp(config)


def test_non_qwen_mtp_is_not_misclassified():
    config = _speculative_config(
        model_architectures=["OtherMTP"],
        text_architectures=["OtherTextModel"],
        text_model_type="other",
    )

    assert not spec_decode._use_qwen4_exp_mtp(config)


def test_model_level_architecture_dispatches_qwen_proposer():
    speculative_config = _speculative_config(
        model_architectures=["Qwen4ExpMTP"],
        text_architectures=None,
    )
    vllm_config = SimpleNamespace(speculative_config=speculative_config)

    with (
        patch.object(
            spec_decode,
            "AscendQwen4ExpMTPProposer",
            return_value="qwen4-exp-mtp",
        ) as qwen_proposer,
        patch.object(spec_decode, "AscendEagleProposer", return_value="generic"),
    ):
        selected = spec_decode.get_spec_decode_method("mtp", vllm_config, "cpu", "runner")

    assert selected == "qwen4-exp-mtp"
    qwen_proposer.assert_called_once_with(vllm_config, "cpu", "runner")
