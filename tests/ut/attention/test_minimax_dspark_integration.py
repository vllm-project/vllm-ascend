# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Exercise MiniMax Flash draft selection on CPU without NPU imports."""

import __future__

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def selector_runtime(monkeypatch):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/platform.py"
    tree = ast.parse(source.read_text())
    names = {"_is_supported_flash_attn_draft", "_validate_flash_attn_backend", "_validate_minimax_dspark_flash_attn"}
    nodes: list[ast.stmt] = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    nodes.extend(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "_MINIMAX_M3_ARCHITECTURES" for t in node.targets)
    )
    runtime = SimpleNamespace(config=None, tag="dspark_head", supported=True)
    backend = SimpleNamespace(AttentionType=SimpleNamespace(DECODER="decoder"))
    # Keep selector contracts independent of native attention imports.
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.fa_v1",
        SimpleNamespace(AscendFlashAttentionBackend=SimpleNamespace(get_supported_head_sizes=lambda: [128])),
    )
    config_module = SimpleNamespace(get_current_vllm_config_or_none=lambda: runtime.config)
    backends = SimpleNamespace(model_tag="dspark_head")
    monkeypatch.setitem(sys.modules, "vllm.compilation", SimpleNamespace(backends=backends))
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backend", backend)
    namespace = {
        "torch": torch,
        "HardwareCapability": SimpleNamespace(DEVICE_LENGTH_FLASH_ATTN=object()),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: runtime.supported),
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
    runtime.select = namespace["_validate_flash_attn_backend"]
    runtime.backends = backends
    runtime.config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        speculative_config=SimpleNamespace(
            method="dspark",
            enforce_eager=True,
            enable_adaptive_verification=False,
            target_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["MiniMaxM3SparseForConditionalGeneration"]),
                hf_text_config=SimpleNamespace(model_type="minimax_m3_text"),
            ),
            draft_model_config=SimpleNamespace(architectures=["DSparkMiniMaxDraftModel"]),
        ),
    )
    runtime.selector = SimpleNamespace(
        attn_type="decoder",
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="bfloat16",
        use_pcp=False,
        use_dcp=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        use_batch_invariant=False,
        use_rswa=False,
        use_adaptive_verification=False,
    )
    return runtime


@pytest.mark.parametrize("architecture", ["MiniMaxM31FlashA3DSparkModel", "DSparkMiniMaxDraftModel"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_draft_selection(selector_runtime, architecture, dtype):
    r = selector_runtime
    r.config.speculative_config.draft_model_config.architectures = [architecture]
    r.selector.dtype = dtype
    assert r.select((False, False, False), r.selector)
