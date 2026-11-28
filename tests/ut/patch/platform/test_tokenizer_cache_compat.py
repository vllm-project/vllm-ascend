# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU frontend compatibility with real configs, renderers and HF tokenizers."""

import asyncio
import importlib
import sys
from contextlib import ExitStack

import pytest
import torch
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers
from transformers import Qwen3Config, TokenizersBackend
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.renderers import base as base_mod
from vllm.renderers import hf as hf_mod
from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.renderers.params import ChatParams, TokenizeParams
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer

NATIVE_CACHE = "tokenizer_cache_gb" in ModelConfig.__dataclass_fields__
BUDGETS = ((0, 0), (1, 0)) + (((0, 0.002), (1, 0.002)) if NATIVE_CACHE else ())
_TPL = "{% for m in messages %}<｜User｜>{{ m.content }}<｜end▁of▁sentence｜>{% endfor %}"


@pytest.fixture(scope="module")
def cache_patch():
    name = "vllm_ascend.patch.platform.patch_tokenizer_cache"
    if name in sys.modules:
        return sys.modules[name]
    before = (base_mod.BaseRenderer.__init__, hf_mod.HfRenderer.__init__, hf_mod.safe_apply_chat_template)
    mod = importlib.import_module(name)
    assert base_mod.BaseRenderer.__init__ is not before[0]
    assert (hf_mod.HfRenderer.__init__ is not before[1]) is (not NATIVE_CACHE and mod._needs_hf_init_patch())
    assert not NATIVE_CACHE or hf_mod.safe_apply_chat_template is before[2]
    return mod


@pytest.fixture
def env(tmp_path):
    qwen = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4}
    Qwen3Config(
        **qwen, num_key_value_heads=2, vocab_size=400, max_position_embeddings=1024, architectures=["Qwen3ForCausalLM"]
    ).save_pretrained(tmp_path)
    b = Tokenizer(models.BPE())
    b.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    alpha = pre_tokenizers.ByteLevel.alphabet()
    b.train_from_iterator(["hello world 你好 code"], trainers.BpeTrainer(vocab_size=300, initial_alphabet=alpha))
    b.add_special_tokens(["<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>"])
    return str(tmp_path), TokenizersBackend(tokenizer_object=b)


def _cfg(model_dir, legacy, native=0):
    extra = {"tokenizer_cache_gb": native} if NATIVE_CACHE else {}
    m = ModelConfig(
        model=model_dir, tokenizer=model_dir, dtype="float32", max_model_len=1024, enforce_eager=True, **extra
    )
    return VllmConfig(
        model_config=m, device_config=DeviceConfig(device="cpu"), additional_config={"tokenizer_cache_gb": legacy}
    )


@pytest.mark.parametrize("legacy,native", BUDGETS)
def test_real_renderers_and_dual_baseline_gate(cache_patch, env, legacy, native):
    from vllm_ascend.utils import vllm_version_is

    assert cache_patch._needs_hf_init_patch() is bool(vllm_version_is("0.34.0"))
    assert (hf_mod.HfRenderer.__init__.__module__ == cache_patch.__name__) is (
        not NATIVE_CACHE and cache_patch._needs_hf_init_patch()
    )
    model_dir, tok = env
    cfg, msgs = _cfg(model_dir, legacy, native), [{"role": "user", "content": "hello 你好"}]
    eff, ds_tok = (native if native > 0 else legacy) if NATIVE_CACHE else legacy, get_deepseek_v4_tokenizer(tok)
    with ExitStack() as stack:
        ds_r, hf_r = DeepseekV4Renderer(cfg, ds_tok), hf_mod.HfRenderer(cfg, tok)
        ds_exp = ds_tok.apply_chat_template(msgs)
        hf_exp = tok.apply_chat_template(msgs, chat_template=_TPL, return_dict=False)
        for r, cp, exp in ((ds_r, ChatParams(), ds_exp), (hf_r, ChatParams(chat_template=_TPL), hf_exp)):
            stack.callback(r.shutdown)
            fn = r.render_messages
            assert fn(msgs, cp)[1]["prompt_token_ids"] == fn(msgs, cp)[1]["prompt_token_ids"] == exp
            assert asyncio.run(r.render_messages_async(msgs, cp))[1]["prompt_token_ids"] == exp
            if NATIVE_CACHE:
                nc = getattr(r, "_tokenizer_cache", None)
                assert cache_patch._cache_for(r.tokenizer) is None
                assert (nc is not None and nc.capacity == int(eff * (1 << 30)) and nc.hits > 0) if eff else (nc is None)
            else:
                c = cache_patch._cache_for(r.tokenizer)
                assert (c is not None and c.stat().hits > 0) if legacy else (c is None)
        ptxt = "<｜User｜>hello world<｜end▁of▁sentence｜>"
        got = hf_r._tokenize_prompt({"prompt": ptxt}, TokenizeParams(max_total_tokens=1024, add_special_tokens=False))
        assert got["prompt_token_ids"] == list(tok(ptxt, add_special_tokens=False)["input_ids"])


def test_guards_and_fallback_boundaries(cache_patch, env):
    model_dir, tok = env
    cfg, pe_cfg = _cfg(model_dir, 0), _cfg(model_dir, 1)
    cfg.additional_config = type("PluginCfg", (), {"compile_factors": lambda self: {"plugin": "x"}})()
    pe_cfg.model_config.enable_prompt_embeds = True
    assert cache_patch._renderer_capacity_gb(cfg) == 0
    with ExitStack() as stack:
        for c in (cfg, pe_cfg):
            r = hf_mod.HfRenderer(c, tok)
            stack.callback(r.shutdown)
            assert cache_patch._cache_for(r.tokenizer) is None and getattr(r, "_tokenizer_cache", None) is None
        opt_r = hf_mod.HfRenderer(_cfg(model_dir, 1), tok)
        stack.callback(opt_r.shutdown)
        conv = [{"role": "user", "content": "hello world"}]
        for _ in range(2):
            hf_mod.safe_apply_chat_template(opt_r.model_config, opt_r.tokenizer, conv, chat_template=_TPL)
        for opts in ({"truncation": True, "max_length": 1}, {"return_tensors": "pt"}):
            exp = opt_r.tokenizer.apply_chat_template(conv, chat_template=_TPL, return_dict=False, **opts)
            act = hf_mod.safe_apply_chat_template(opt_r.model_config, opt_r.tokenizer, conv, chat_template=_TPL, **opts)
            assert torch.equal(act, exp) if isinstance(exp, torch.Tensor) else act == exp
        tok.add_tokens([AddedToken("zzzzzz", single_word=True, normalized=False)])
        cs_r = hf_mod.HfRenderer(_cfg(model_dir, 1), tok)
        stack.callback(cs_r.shutdown)
        tpar = TokenizeParams(max_total_tokens=1024, add_special_tokens=False)
        got = cs_r._tokenize_prompt({"prompt": "zzzzzzpost"}, tpar)["prompt_token_ids"]
        assert got == tok.encode("zzzzzzpost", add_special_tokens=False)
        native = getattr(cs_r, "_tokenizer_cache", None)
        assert cache_patch._cache_for(cs_r.tokenizer) is None and (native is None or not native.enabled)
