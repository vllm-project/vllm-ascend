# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the real VMM table loader, with only physical mappings replaced.

Standalone: pytest --confcutdir=tests/ut/models tests/ut/models/test_engram_vmm.py
"""

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture
def implementation(monkeypatch):
    root = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41"
    prefix = "_engram_vmm_unit"
    package = ModuleType(prefix)
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, prefix, package)
    child = ModuleType(prefix + ".engram_vmm")
    child.__path__ = [str(root / "engram_vmm")]
    monkeypatch.setitem(sys.modules, child.__name__, child)
    mapping = ModuleType(child.__name__ + ".mapping")

    class Mapping:
        def __init__(self, path, rows, width, dtype, group):
            self.tensor = torch.zeros(rows, width, dtype=dtype)
            self.closed = False

        def publish(self, start, source):
            self.tensor[start : start + len(source)].copy_(source)

        def close(self):
            self.closed = True

    mapping.VmmMapping = Mapping
    monkeypatch.setitem(sys.modules, mapping.__name__, mapping)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    module = importlib.import_module(child.__name__ + ".table")
    yield module
    for name in list(sys.modules):
        if name.startswith(prefix + "."):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("source_dtype", ["int8", "bf16"])
@pytest.mark.parametrize("index_name", ["quant_model_weights.safetensors.index.json", "model.safetensors.index.json"])
def test_checkpoint_publishes_only_owned_rows(implementation, tmp_path, source_dtype, index_name):
    table = implementation.VmmEngram(11, 256, SimpleNamespace(rank=1, size=2), run_id="unit", layer_id=1)
    key = "layers.1.engram.embed.weight"
    scale_key = key.removesuffix(".weight") + ".scale"
    values = (torch.arange(11 * 256).reshape(11, 256) % 255 - 127).to(torch.int8)
    scales = torch.ones(11, 8) / 128
    tensors = {key: values, scale_key: scales}
    if source_dtype == "bf16":
        tensors = {key: (values.float() / 128).bfloat16()}
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / index_name).write_text(json.dumps({"weight_map": dict.fromkeys(tensors, "weights.safetensors")}))
    table.load_checkpoint(tmp_path, key, chunk_rows=2)
    assert table.loaded and (table.start, table.end) == (6, 11)
    assert torch.count_nonzero(table.codes_map.tensor[:6]) == 0
    actual = table.weight.float().reshape(5, 8, 32) * table.weight_scale.unsqueeze(-1)
    expected = values[6:].float().reshape(5, 8, 32) / 128
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="shard"):
        table.set_int8_rows(5, values[:1], scales[:1])
    with pytest.raises(RuntimeError, match="direct lookup"):
        table(torch.tensor([0]))
    table.close()
    table.close()
    assert table.weight is None and table.weight_scale is None
    assert table.codes_map.closed and table.scales_map.closed


def test_unsupported_width_fails_before_mapping(implementation):
    with pytest.raises(ValueError, match="width 256"):
        implementation.VmmEngram(11, 128, SimpleNamespace(rank=0, size=1), run_id="unit", layer_id=1)


def test_parameter_failure_transfers_both_failed_rollbacks(implementation, monkeypatch):
    attempts = []
    mapping_type = implementation.VmmMapping

    class FailingRelease(mapping_type):
        def _rollback(self, original):
            attempts.append(self.tensor.dtype)
            raise RuntimeError("deferred release")

    setattr_original = implementation.VmmEngram.__setattr__

    def attach(self, name, value):
        if name == "weight" and isinstance(value, torch.nn.Parameter) and not value.is_meta:
            raise RuntimeError("parameter attachment failed")
        setattr_original(self, name, value)

    monkeypatch.setattr(implementation, "VmmMapping", FailingRelease)
    monkeypatch.setattr(implementation.VmmEngram, "__setattr__", attach)
    with pytest.raises(RuntimeError, match="construction rollback"):
        implementation.VmmEngram(11, 256, SimpleNamespace(rank=0, size=1), run_id="unit", layer_id=1)
    assert attempts == [torch.float32, torch.int8]
