# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from vllm.tokenizers import get_tokenizer

from vllm_ascend.tokenizers import register_tokenizers


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
