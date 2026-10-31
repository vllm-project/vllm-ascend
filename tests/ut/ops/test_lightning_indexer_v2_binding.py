# SPDX-License-Identifier: Apache-2.0
"""Exercise the Python bridge against the actual C++ schema on a CPU dispatcher.

This checks argument transport, not CANN numerical correctness (covered on NPU).
No vLLM or torch-npu installation is required.
"""

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def bridge(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "li_v2_bridge", root / "vllm_ascend/ops/lightning_indexer_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = torch.library.Library("li_v2_contract_test", "FRAGMENT")
    calls = []
    source = (root / "csrc/torch_binding.cpp").read_text(encoding="utf-8")
    names = []
    for match in re.finditer(r'ops.def\(\s*((?:"(?:\\.|[^"\\])*"\s*)+)\);', source):
        schema = "".join(json.loads(part) for part in re.findall(r'"(?:\\.|[^"\\])*"', match[1]))
        if not schema.startswith("npu_lightning_indexer_v2"):
            continue
        name = schema.split("(")[0]
        library.define(schema)
        names.append(name)

    def metadata(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.empty(2048, dtype=torch.int32)

    def compute(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.empty(1, dtype=torch.int32), torch.empty(0, dtype=torch.float32)

    library.impl("npu_lightning_indexer_v2_metadata", metadata, "CPU")
    library.impl("npu_lightning_indexer_v2", compute, "CPU")
    monkeypatch.setattr(torch.ops, "_C_ascend", SimpleNamespace(**{
        name: getattr(torch.ops.li_v2_contract_test, name) for name in names
    }))
    yield module, calls
    library._destroy()


def test_metadata_none_scalar_defaults(bridge):
    module, calls = bridge
    cu_q = torch.tensor([0, 1], dtype=torch.int32)
    output = module.lightning_indexer_v2_metadata(
        4, 1, 128, 512, cu_seqlens_q=cu_q, batch_size=None, max_seqlen_q=None,
        max_seqlen_k=None, layout_q=None, layout_k=None, mask_mode=None, cmp_ratio=None,
    )
    assert output.shape == (2048,)
    assert output.dtype == torch.int32
    assert calls[0][0] == (4, 1, 128, 512)
    # Dispatcher omits kwargs equal to their schema defaults.
    assert list(calls[0][1]) == ["cu_seqlens_q"]
    assert calls[0][1]["cu_seqlens_q"] is cu_q


def test_compute_transports_every_optional_input(bridge):
    module, calls = bridge
    q = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 16, 1, 128, dtype=torch.bfloat16)
    w = torch.ones(2, 4, dtype=torch.float32)
    tensors = {name: torch.tensor([index], dtype=torch.int32) for index, name in enumerate([
        "cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k", "cmp_residual_k",
        "block_table", "output_idx_offset", "metadata",
    ])}
    options = dict(max_seqlen_q=2, layout_q="TND", layout_k="PA_BBND",
                   mask_mode=3, cmp_ratio=4, return_value=1)
    module.lightning_indexer_v2(q, k, w, 512, **tensors, **options)
    args, kwargs = calls[0]
    assert all(actual is expected for actual, expected in zip(args[:3], (q, k, w)))
    assert args[3] == 512
    assert all(kwargs[name] is tensor for name, tensor in tensors.items())
    assert all(kwargs[name] == value for name, value in options.items())


def test_metadata_rejects_missing_length_tensors(bridge):
    module, calls = bridge
    with pytest.raises(ValueError, match="at least one NPU length tensor"):
        module.lightning_indexer_v2_metadata(4, 1, 128, 512)
    assert not calls
