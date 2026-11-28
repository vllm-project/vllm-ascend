# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.registry import TokenizerRegistry

from vllm_ascend.tokenizers import DeepSeekR1DistillLlamaTokenizer, register_tokenizers

TOKENIZER_MODE = "deepseek_r1_distill_llama"


@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch):
    tokenizers = dict(TokenizerRegistry.tokenizers)
    renderers = dict(RENDERER_REGISTRY.renderers)
    tokenizers.pop(TOKENIZER_MODE, None)
    renderers.pop(TOKENIZER_MODE, None)
    monkeypatch.setattr(TokenizerRegistry, "tokenizers", tokenizers)
    monkeypatch.setattr(RENDERER_REGISTRY, "renderers", renderers)


@pytest.fixture
def byte_level_checkpoint(tmp_path):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["Hello world\n", "你好，世界！\n", "答案：A", "  multiple  spaces\t"],
        trainers.BpeTrainer(
            vocab_size=512,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<s>", "</s>"],
        ),
    )
    tokenizer.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizerFast",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
        )
    )
    return tmp_path, tokenizer


@pytest.mark.parametrize("text", ["", "Hello world\n", "你好，世界！\n", "答案：A", "  multiple  spaces\t"])
def test_distill_llama_preserves_serialized_tokenizer(byte_level_checkpoint, text):
    path, reference = byte_level_checkpoint
    register_tokenizers()
    tokenizer = get_tokenizer(path, tokenizer_mode="deepseek_r1_distill_llama")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    assert token_ids == reference.encode(text, add_special_tokens=False).ids
    assert tokenizer.decode(token_ids) == text
    assert tokenizer.encode(text) == reference.encode(text).ids
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2


def test_registration_preserves_existing_modes():
    tokenizers = dict(TokenizerRegistry.tokenizers)
    renderers = dict(RENDERER_REGISTRY.renderers)
    register_tokenizers()
    assert TokenizerRegistry.tokenizers == {
        **tokenizers,
        TOKENIZER_MODE: ("vllm_ascend.tokenizers", "DeepSeekR1DistillLlamaTokenizer"),
    }
    assert RENDERER_REGISTRY.renderers == {
        **renderers,
        TOKENIZER_MODE: ("vllm.renderers.hf", "HfRenderer"),
    }


@pytest.mark.parametrize("tokenizer_registered,renderer_registered", [(True, False), (False, True), (True, True)])
def test_registration_preserves_overrides(tokenizer_registered, renderer_registered):
    tokenizer_override = ("custom.tokenizer", "Tokenizer")
    renderer_override = ("custom.renderer", "Renderer")
    if tokenizer_registered:
        TokenizerRegistry.tokenizers[TOKENIZER_MODE] = tokenizer_override
    if renderer_registered:
        RENDERER_REGISTRY.renderers[TOKENIZER_MODE] = renderer_override
    register_tokenizers()
    tokenizers = dict(TokenizerRegistry.tokenizers)
    renderers = dict(RENDERER_REGISTRY.renderers)
    register_tokenizers()
    assert TokenizerRegistry.tokenizers == tokenizers
    assert RENDERER_REGISTRY.renderers == renderers
    if tokenizer_registered:
        assert tokenizers[TOKENIZER_MODE] == tokenizer_override
    else:
        assert tokenizers[TOKENIZER_MODE] == ("vllm_ascend.tokenizers", "DeepSeekR1DistillLlamaTokenizer")
    if renderer_registered:
        assert renderers[TOKENIZER_MODE] == renderer_override
    else:
        assert renderers[TOKENIZER_MODE] == ("vllm.renderers.hf", "HfRenderer")


def test_invalid_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError, match="instantiate the backend tokenizer"):
        DeepSeekR1DistillLlamaTokenizer.from_pretrained(tmp_path, local_files_only=True)
