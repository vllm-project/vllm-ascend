# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual cache methods without importing the NPU runtime."""

import ast
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import nn


@pytest.fixture(params=["v4", "v41"])
def cache(request):
    path = Path(__file__).resolve().parents[3] / f"vllm_ascend/models/deepseek_{request.param}/vision.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_vision_cos_sin"]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name.endswith("ViT"))
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_get_rope", "_apply", "_log_rope_cache_stats")
    ]
    nodes.append(cls)
    scope = dict(
        torch=torch,
        nn=nn,
        OrderedDict=OrderedDict,
        lru_cache=lru_cache,
        _ROPE_CACHE_LOG_INTERVAL=100,
        _ROPE_CACHE_MAX_ENTRIES=2,
        _ROPE_CACHE_MAX_BYTES=256,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    model = scope[cls.name]()
    model.rope_dim = 4
    model.rope_theta = 10000.0
    model._rope_cache_logger = Mock()
    model._rope_cache_hits = 0
    model._rope_cache_misses = 0
    model._rope_cache_evictions = 0
    model._rope_cache_oversized = 0
    model._rope_cache = OrderedDict()
    model._rope_cache_bytes = 0
    return model, scope


def test_reuse_precision_eviction_and_clear(cache):
    model, scope = cache
    device = torch.device("cpu")
    first = model._get_rope(1, 2, device)
    expected = scope["get_vision_cos_sin"](1, 2, 4, 10000.0)
    for actual, reference in zip(first, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert actual.dtype == torch.float32
    assert model._get_rope(1, 2, device) is first
    model._get_rope(2, 1, device)
    model._get_rope(1, 2, device)  # Touch the oldest entry.
    model._get_rope(1, 3, device)
    assert [key[:2] for key in model._rope_cache] == [(1, 2), (1, 3)]
    assert model._rope_cache_bytes == 160
    model._get_rope(2, 3, device)  # Byte budget evicts both entries.
    assert len(model._rope_cache) == 1
    assert model._rope_cache_bytes == 192
    model._get_rope(10, 10, device)  # Oversized entries are not retained.
    assert model._rope_cache_bytes == 192
    assert not model.state_dict()
    model.to(dtype=torch.bfloat16)
    assert not model._rope_cache
    assert model._rope_cache_bytes == 0
    assert model._get_rope(1, 2, device)[0].dtype == torch.float32


def test_cache_logging_counts_and_interval(cache):
    model, _ = cache
    device = torch.device("cpu")
    for _ in range(100):
        model._get_rope(1, 2, device)
    assert model._rope_cache_hits == 99
    assert model._rope_cache_misses == 1
    assert model._rope_cache_logger.info.call_count == 2
    args = model._rope_cache_logger.info.call_args.args
    assert args[3:7] == (100, 99, 1, 99.0)
    model._get_rope(2, 3, device)
    model._get_rope(1, 3, device)
    assert model._rope_cache_evictions == 2
    model._get_rope(10, 10, device)
    assert model._rope_cache_oversized == 1
    assert model._rope_cache_misses == 4
    assert model._rope_cache_logger.info.call_count == 2
    model.to(dtype=torch.bfloat16)
    assert model._rope_cache_hits == 99
    assert model._rope_cache_misses == 4
