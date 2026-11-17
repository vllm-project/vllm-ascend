# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from vllm.config import SpeculativeConfig

from vllm_ascend.utils import vllm_version_is


class CapturedLLMConfig(BaseException):
    def __init__(self, kwargs: dict[str, Any]):
        self.kwargs = kwargs


@pytest.mark.parametrize("method", ["eagle", "eagle3"])
@pytest.mark.parametrize("disable_padding", [False, True])
def test_dsd_tuner_padding_matches_native_config(method, disable_padding):
    """Exercise the example's LLM boundary without model loading or NPU work."""
    path = Path(__file__).resolve().parents[2] / "examples/offline_DSD_k_tuner/generate_config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("get_acceptance_rate_per_pos", "build_compilation_config")
    ]
    assert len(nodes) == 2
    tokenizer = Mock()

    def capture_llm(**kwargs):
        raise CapturedLLMConfig(kwargs)

    llm = Mock(side_effect=capture_llm)
    scope: dict[str, Any] = {
        "AutoTokenizer": tokenizer,
        "get_samples": lambda *args: [],
        "LLM": llm,
        "vllm_version_is": vllm_version_is,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    args = SimpleNamespace(
        method=method,
        disable_padded_drafter_batch=disable_padding,
        model_dir="target",
        enable_multimodal_chat=False,
        eagle_dir="draft",
        num_spec_tokens=3,
        parallel_drafting=True,
        tp=1,
        enable_chunked_prefill=True,
        enforce_eager=False,
        gpu_memory_utilization=0.8,
        max_model_len=1024,
        max_vllm_batch_size=4,
        allowed_local_media_path="",
        cudagraph_capture_sizes=[1, 4],
    )
    release = vllm_version_is("0.33.0")
    field = "disable_padded_drafter_batch"
    assert (field in SpeculativeConfig.__annotations__) == release
    if disable_padding and not release:
        with pytest.raises(ValueError, match="supported only with vLLM v0.33.0"):
            scope["get_acceptance_rate_per_pos"](args)
        tokenizer.from_pretrained.assert_not_called()
        llm.assert_not_called()
    else:
        with pytest.raises(CapturedLLMConfig) as captured:
            scope["get_acceptance_rate_per_pos"](args)
        expected = {
            "method": method,
            "model": "draft",
            "num_speculative_tokens": 3,
            "parallel_drafting": True,
        }
        if release:
            expected[field] = disable_padding
        assert captured.value.kwargs["speculative_config"] == expected
        assert captured.value.kwargs["compilation_config"] == {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 4],
        }
