import asyncio
import json
import random
import sys
import types
from pathlib import Path

import pytest

from tools import prepare_glm53_flash_rp3 as rp3


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


def test_prompt_length_and_reproducibility():
    words = [str(n) for n in range(100)]
    a = rp3.make_prompt(WordTokenizer(), words, random.Random(1), 16)
    b = rp3.make_prompt(WordTokenizer(), words, random.Random(1), 16)
    c = rp3.make_prompt(WordTokenizer(), words, random.Random(2), 16)
    assert a == b and a != c
    assert len(a.split()) == 16


def test_dataset_sizes_and_prefix_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(rp3, "INPUT_TOKENS", 16)
    result = rp3.prepare_datasets(WordTokenizer(), " ".join(str(n) for n in range(100)), tmp_path)
    for name, count in (("prefill", 280), ("pd", 560), ("prefill_warmup", 8), ("pd_warmup", 8)):
        assert result[name]["requests"] == result[name]["unique_prompts"] == count
    assert result["decode"]["requests"] == 1168
    assert result["decode"]["unique_prompts"] == 1
    cold = lambda name: {
        json.loads(line)["question"] for line in (tmp_path / name / "test.jsonl").read_text().splitlines()
    }
    assert cold("prefill").isdisjoint(cold("prefill_warmup"))
    assert cold("pd").isdisjoint(cold("pd_warmup"))
    assert result == json.loads((tmp_path / "manifest.json").read_text())


def test_reject_empty_source(tmp_path):
    with pytest.raises(ValueError, match="Empty source"):
        rp3.prepare_datasets(WordTokenizer(), "", tmp_path)


def test_client_cache_salts_and_no_retries(tmp_path, monkeypatch):
    module = types.ModuleType("rp3_test_api")

    class API:
        async def get_request_body(self, *args, **kwargs):
            return {"messages": ["unchanged"], "max_tokens": 1024}

    module.API = API
    monkeypatch.setitem(sys.modules, module.__name__, module)
    (tmp_path / "vllm_api_stream_chat.py").write_text(
        "from rp3_test_api import API as VLLMCustomAPIChat\n"
        "models = [dict(type=VLLMCustomAPIChat, retry=2, generation_kwargs=dict(ignore_eos=True))]\n"
    )
    rp3.prepare_request_configs(tmp_path)
    namespace = {}
    exec((tmp_path / "vllm_api_stream_chat_rp3_cold.py").read_text(), namespace)
    cold = namespace["models"][0]
    assert cold["retry"] == 0
    instance = cold["type"]()
    first = asyncio.run(instance.get_request_body())
    second = asyncio.run(instance.get_request_body())
    assert first["cache_salt"] != second["cache_salt"]
    assert first["messages"] == ["unchanged"] and first["max_tokens"] == 1024
    salts = []
    for name in ("low", "high"):
        namespace = {}
        exec((tmp_path / f"vllm_api_stream_chat_rp3_{name}.py").read_text(), namespace)
        warm = namespace["models"][0]
        assert warm["retry"] == 0
        salts.append(warm["generation_kwargs"]["cache_salt"])
    assert len(set(salts)) == 2


def test_reject_changed_aisbench_template(tmp_path: Path):
    (tmp_path / "vllm_api_stream_chat.py").write_text("models = []\n")
    with pytest.raises(ValueError, match="retry"):
        rp3.prepare_request_configs(tmp_path)
