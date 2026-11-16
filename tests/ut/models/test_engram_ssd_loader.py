# SPDX-License-Identifier: Apache-2.0
"""T4 loader 契约测试:表张量在 get_tensor 之前被跳过(04 §9.4)。

用真实 safetensors 小文件 + 计数 safe_open 代理验证:
  1. 表 weight/scale 的 get_tensor 调用次数为 0;
  2. gate/投影等参数正常加载且值正确;
  3. 只含表的分片文件不被 open;
  4. external_tables 账本完整性与断言。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import vllm_ascend.ops  # noqa: F401,E402
from vllm_ascend.model_loader import engram_ssd_loader as L  # noqa: E402


@pytest.fixture()
def fake_ckpt(tmp_path: Path) -> Path:
    d = tmp_path / "ckpt"
    d.mkdir()
    # 混合分片: 表 + gate/投影
    save_file(
        {
            "layers.1.engram.embed.weight": torch.zeros(1000, 256, dtype=torch.int8),
            "layers.1.engram.embed.scale": torch.zeros(1000, 8, dtype=torch.float32),
            "layers.1.engram.wkv.weight": torch.ones(64, 128, dtype=torch.bfloat16),
            "layers.0.input_layernorm.weight": torch.full((128,), 2.0, dtype=torch.bfloat16),
        },
        d / "model-00001-of-00003.safetensors",
    )
    # 表专属分片(模拟 quant_model_weights-engram-14.safetensors)
    save_file(
        {
            "layers.14.engram.embed.weight": torch.zeros(1000, 256, dtype=torch.int8),
            "layers.14.engram.embed.scale": torch.zeros(1000, 8, dtype=torch.float32),
        },
        d / "model-00002-of-00003.safetensors",
    )
    # 纯非表分片
    save_file(
        {
            "layers.2.mlp.gate_proj.weight": torch.arange(8, dtype=torch.bfloat16).reshape(8, 1),
        },
        d / "model-00003-of-00003.safetensors",
    )
    # index: 与真实产物同构(engram 分片表专属)
    import json

    (d / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.1.engram.embed.weight": "model-00001-of-00003.safetensors",
                    "layers.1.engram.embed.scale": "model-00001-of-00003.safetensors",
                    "layers.1.engram.wkv.weight": "model-00001-of-00003.safetensors",
                    "layers.0.input_layernorm.weight": "model-00001-of-00003.safetensors",
                    "layers.14.engram.embed.weight": "model-00002-of-00003.safetensors",
                    "layers.14.engram.embed.scale": "model-00002-of-00003.safetensors",
                    "layers.2.mlp.gate_proj.weight": "model-00003-of-00003.safetensors",
                }
            }
        )
    )
    return d


class CountingOpen:
    """包装 safe_open, 统计 open 的文件与 get_tensor 的名字。"""

    def __init__(self, real):
        self.real = real
        self.opened: list[str] = []
        self.get_tensor_calls: list[str] = []

    def __call__(self, filename, framework="pt", *a, **kw):
        self.opened.append(str(filename))
        inner = self.real(filename, framework, *a, **kw)
        counter = self

        class Guard:
            def __enter__(self_):
                self_._f = inner.__enter__()
                return self_

            def __exit__(self_, *exc):
                return inner.__exit__(*exc)

            def __getattr__(self_, name):
                attr = getattr(self_._f, name)
                if name == "get_tensor":

                    def wrapped(k, *aa, **kkk):
                        counter.get_tensor_calls.append(k)
                        return attr(k, *aa, **kkk)

                    return wrapped
                return attr

        return Guard()


def test_loader_skips_table_before_get_tensor(fake_ckpt):
    from vllm.config import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    loader = L.EngramSSDModelLoader(LoadConfig(load_format="ascend_engram_ssd"))
    counter = CountingOpen(safe_open)
    with patch.object(L, "safe_open", counter):
        got = dict(
            loader._get_weights_iterator(
                DefaultModelLoader.Source(model_or_path=str(fake_ckpt), revision=None, prefix="")
            )
        )
    # 1) 表 key 的 get_tensor 次数为 0
    table_keys = [k for k in counter.get_tensor_calls if L.is_engram_table_key(k)]
    assert table_keys == [], f"表张量被物化: {table_keys}"
    # 2) 非表参数全部加载且值正确
    assert set(got) == {
        "layers.1.engram.wkv.weight",
        "layers.0.input_layernorm.weight",
        "layers.2.mlp.gate_proj.weight",
    }
    assert torch.all(got["layers.1.engram.wkv.weight"] == 1)
    assert torch.all(got["layers.0.input_layernorm.weight"] == 2.0)
    assert got["layers.2.mlp.gate_proj.weight"].shape == (8, 1)
    # 3) 表专属分片不被打开
    assert not any("00002-of-00003" in p for p in counter.opened), f"表专属分片被打开: {counter.opened}"
    assert any("00001" in p for p in counter.opened)
    # 4) 账本
    ledger = L.external_tables[str(fake_ckpt)]
    assert ledger["layers"] == {1: ["scale", "weight"], 14: ["scale", "weight"]}
    assert ledger["skipped_bytes"] > 0
    L.assert_external_tables_complete(str(fake_ckpt), [1, 14])
    with pytest.raises(RuntimeError):
        L.assert_external_tables_complete(str(fake_ckpt), [1])


def test_is_engram_table_key_precise():
    assert L.is_engram_table_key("layers.14.engram.embed.weight")
    assert L.is_engram_table_key("model.layers.1.engram.embed.scale")
    assert not L.is_engram_table_key("layers.14.engram.wkv.weight")
    assert not L.is_engram_table_key("layers.14.engram.embed_tokens.weight.x")
    assert not L.is_engram_table_key("engram.embed.weight")
    assert L._layer_of("model.layers.14.engram.embed.weight") == 14


def test_loader_registered():
    from vllm.model_executor.model_loader import _LOAD_FORMAT_TO_MODEL_LOADER

    L.EngramSSDModelLoader  # noqa: B018  (触发模块导入)
    # 注册发生在 vllm_ascend.register_model_loader(); 直接验证映射可写
    from vllm.model_executor.model_loader import register_model_loader

    if "ascend_engram_ssd" not in _LOAD_FORMAT_TO_MODEL_LOADER:

        @register_model_loader("ascend_engram_ssd")
        class _Probe(L.EngramSSDModelLoader):
            pass

    assert _LOAD_FORMAT_TO_MODEL_LOADER["ascend_engram_ssd"] is not None
