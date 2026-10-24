# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU frontend compatibility with real configs, renderers and HF tokenizers."""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers
from transformers import Qwen3Config, TokenizersBackend
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.renderers import base as base_module
from vllm.renderers import hf as hf_module
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.params import ChatParams, TokenizeParams
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer

NATIVE_CACHE = "tokenizer_cache_gb" in ModelConfig.__dataclass_fields__
BUDGETS = ((0, 0), (1, 0)) + (((0, 0.002), (1, 0.002)) if NATIVE_CACHE else ())


@pytest.fixture(scope="module")
def cache_patch():
    name = "vllm_ascend.patch.platform.patch_tokenizer_cache"
    if name in sys.modules:
        return sys.modules[name]
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_tokenizer_cache.py"
    before_init = base_module.BaseRenderer.__init__
    before_chat = hf_module.safe_apply_chat_template
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert base_module.BaseRenderer.__init__ is not before_init
    if NATIVE_CACHE:
        assert hf_module.safe_apply_chat_template is before_chat
    return module


@pytest.fixture
def local_model(tmp_path):
    Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=400,
        max_position_embeddings=1024,
        architectures=["Qwen3ForCausalLM"],
    ).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["hello world 你好 code unit test"],
        trainers.BpeTrainer(vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    backend.add_special_tokens(["<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>"])
    return TokenizersBackend(tokenizer_object=backend)


def renderer_config(local_model, legacy, native=0):
    kwargs = {"tokenizer_cache_gb": native} if NATIVE_CACHE else {}
    model = ModelConfig(
        model=local_model,
        tokenizer=local_model,
        dtype="float32",
        max_model_len=1024,
        enforce_eager=True,
        **kwargs,
    )
    return VllmConfig(
        model_config=model,
        device_config=DeviceConfig(device="cpu"),
        additional_config={"tokenizer_cache_gb": legacy},
    )


@pytest.mark.parametrize("legacy,native", BUDGETS)
def test_real_renderer_cache_budget_and_signature(cache_patch, local_model, tokenizer, legacy, native):
    config = renderer_config(local_model, legacy, native)
    tokenizer = get_deepseek_v4_tokenizer(tokenizer)
    renderer = DeepseekV4Renderer(config, tokenizer)
    messages = [{"role": "user", "content": "hello 你好"}]
    expected = tokenizer.apply_chat_template(messages)
    try:
        for _ in range(2):
            _, prompt = renderer.render_messages(messages, ChatParams())
            assert prompt["prompt_token_ids"] == expected
        _, prompt = asyncio.run(renderer.render_messages_async(messages, ChatParams()))
        assert prompt["prompt_token_ids"] == expected
        if NATIVE_CACHE:
            assert config.model_config.tokenizer_cache_gb == native
            assert cache_patch._cache_for(renderer.tokenizer) is None
            effective = native if native > 0 else legacy
            if effective:
                assert renderer._tokenizer_cache.capacity == int(effective * (1 << 30))
                assert renderer._tokenizer_cache.hits > 0
            else:
                assert renderer._tokenizer_cache is None
        elif legacy:
            assert cache_patch._cache_for(renderer.tokenizer).stat().hits > 0
        else:
            assert cache_patch._cache_for(renderer.tokenizer) is None
    finally:
        renderer.shutdown()


@pytest.mark.parametrize("invalid", [-1, float("nan")])
def test_invalid_legacy_budget_is_rejected(cache_patch, local_model, invalid):
    config = renderer_config(local_model, invalid)
    with pytest.raises(ValueError, match="finite number >= 0"):
        cache_patch._renderer_capacity_gb(config)


def test_non_mapping_additional_config_keeps_cache_off(cache_patch, local_model, tokenizer):
    class PluginConfig:
        def compute_hash(self):
            return "custom-plugin-config"

    config = renderer_config(local_model, 0)
    # VllmConfig explicitly supports plugin-specific SupportsHash objects.
    config = VllmConfig(
        model_config=config.model_config,
        device_config=DeviceConfig(device="cpu"),
        additional_config=PluginConfig(),
    )
    renderer = hf_module.HfRenderer(config, tokenizer)
    try:
        assert cache_patch._cache_for(renderer.tokenizer) is None
        assert getattr(renderer, "_tokenizer_cache", None) is None
    finally:
        renderer.shutdown()


@pytest.mark.parametrize("options", [{"truncation": True, "max_length": 1}, {"return_tensors": "pt"}])
def test_hf_options_after_warm_cache_keep_regular_result(cache_patch, local_model, tokenizer, options):
    config = renderer_config(local_model, 1)
    renderer = hf_module.HfRenderer(config, tokenizer)
    conversation = [{"role": "user", "content": "hello world"}]
    template = "{% for m in messages %}<｜User｜>{{ m.content }}<｜end▁of▁sentence｜>{% endfor %}"
    try:
        # Populate a plain chat first: the option-bearing call must not inherit
        # an earlier positive chat-identity verdict and lose its options.
        for _ in range(2):
            hf_module.safe_apply_chat_template(
                renderer.model_config, renderer.tokenizer, conversation, chat_template=template
            )
        # safe_apply_chat_template explicitly normalizes the HF5 default to
        # return_dict=False. Match that public API when computing the reference.
        expected = renderer.tokenizer.apply_chat_template(
            conversation, chat_template=template, return_dict=False, **options
        )
        actual = hf_module.safe_apply_chat_template(
            renderer.model_config, renderer.tokenizer, conversation, chat_template=template, **options
        )
        if isinstance(expected, torch.Tensor):
            assert isinstance(actual, torch.Tensor)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected
    finally:
        renderer.shutdown()


def test_context_sensitive_tokenizer_falls_back(cache_patch, local_model, tokenizer):
    # Sort after the three template tokens: a finite startup corpus alone
    # does not exercise this context-sensitive added token.
    tokenizer.add_tokens([AddedToken("zzzzzz", single_word=True, normalized=False)])
    renderer = hf_module.HfRenderer(renderer_config(local_model, 1), tokenizer)
    messages = [{"role": "user", "content": "prezzzzzzpost"}]
    template = "{% for m in messages %}<｜User｜>{{ m.content }}<｜end▁of▁sentence｜>{% endfor %}"
    expected = tokenizer.apply_chat_template(messages, chat_template=template, return_dict=False)
    try:
        text = "zzzzzzpost"
        completion = renderer._tokenize_prompt(
            {"prompt": text}, TokenizeParams(max_total_tokens=1024, add_special_tokens=False)
        )
        assert completion["prompt_token_ids"] == tokenizer.encode(text, add_special_tokens=False)
        _, prompt = renderer.render_messages(messages, ChatParams(chat_template=template))
        assert prompt["prompt_token_ids"] == expected
        assert cache_patch._cache_for(renderer.tokenizer) is None
        native = getattr(renderer, "_tokenizer_cache", None)
        assert native is None or not native.enabled
    finally:
        renderer.shutdown()
