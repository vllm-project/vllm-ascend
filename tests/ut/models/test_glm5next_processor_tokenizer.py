# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check which tokenizer the GLM-5-Next multimodal processor is built with."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import WhitespaceSplit  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast

from vllm_ascend.models.glm5next.multimodal import Glm5NextProcessingInfo
from vllm_ascend.models.glm5next.processor import Glm5NextProcessor

_TOKENS = ["[UNK]", "<|image|>", "<|video|>", "hello"]
_REMOTE_TOKENIZER = """
from transformers import PreTrainedTokenizerFast


class Glm5NextRemoteTokenizer(PreTrainedTokenizerFast):
    pass
"""


def _tokenizer_object():
    tokenizer = Tokenizer(WordLevel({token: i for i, token in enumerate(_TOKENS)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return tokenizer


@pytest.fixture
def remote_code_checkpoint(tmp_path):
    """A checkpoint whose ``tokenizer_config.json`` names a remote-code tokenizer."""
    _tokenizer_object().save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenization_remote.py").write_text(_REMOTE_TOKENIZER)
    class_ref = "tokenization_remote.Glm5NextRemoteTokenizer"
    tokenizer_config = {
        "tokenizer_class": "Glm5NextRemoteTokenizer",
        "auto_map": {"AutoTokenizer": [class_ref, class_ref]},
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    processor_config = {
        "image_processor": {"image_processor_type": "Glm5NextImageProcessor"},
        "video_processor": {"video_processor_type": "Glm5NextVideoProcessor"},
        "processor_class": "Glm5NextProcessor",
    }
    (tmp_path / "processor_config.json").write_text(json.dumps(processor_config))
    return tmp_path


def _processing_info(model, *, tokenizer=None, trust_remote_code=False, revision=None):
    info = object.__new__(Glm5NextProcessingInfo)
    model_config = SimpleNamespace(model=str(model), revision=revision, trust_remote_code=trust_remote_code)
    info.ctx = SimpleNamespace(model_config=model_config, tokenizer=tokenizer)
    return info


def test_processing_info_forwards_engine_tokenizer_and_model_config():
    tokenizer = object()
    info = _processing_info("/models/glm", tokenizer=tokenizer, trust_remote_code=True, revision="abc123")
    with patch.object(Glm5NextProcessor, "from_pretrained") as from_pretrained:
        assert info.get_hf_processor() is from_pretrained.return_value
    from_pretrained.assert_called_once_with(
        "/models/glm", tokenizer=tokenizer, revision="abc123", trust_remote_code=True
    )


def test_processor_reuses_engine_tokenizer(remote_code_checkpoint):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=_tokenizer_object())
    info = _processing_info(remote_code_checkpoint, tokenizer=tokenizer)
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("tokenizer reloaded")):
        processor = info.get_hf_processor()
    assert processor.tokenizer is tokenizer
    assert processor.image_token_id == _TOKENS.index("<|image|>")
    assert processor.video_token_id == _TOKENS.index("<|video|>")


def test_remote_code_tokenizer_requires_trust_remote_code(remote_code_checkpoint):
    with pytest.raises(ValueError, match="trust_remote_code"):
        _processing_info(remote_code_checkpoint, trust_remote_code=False).get_hf_processor()

    processor = _processing_info(remote_code_checkpoint, trust_remote_code=True).get_hf_processor()
    assert type(processor.tokenizer).__name__ == "Glm5NextRemoteTokenizer"
    assert processor(text="hello <|image|>")["input_ids"] == [[_TOKENS.index("hello"), _TOKENS.index("<|image|>")]]
