# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Execute the production draft backend selection logic on CPU without NPU imports."""

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
    config_module = SimpleNamespace(get_current_vllm_config_or_none=lambda: runtime.config)
    backends = SimpleNamespace(model_tag="dspark_head")
    monkeypatch.setitem(sys.modules, "vllm.compilation", SimpleNamespace(backends=backends))
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backend", backend)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.attention.fa_v1",
        SimpleNamespace(AscendFlashAttentionBackend=SimpleNamespace(get_supported_head_sizes=lambda: [128, 192, 256])),
    )
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
@pytest.mark.parametrize("eager", [False, True])
def test_flash_draft_selection(selector_runtime, architecture, dtype, eager):
    r = selector_runtime
    r.config.speculative_config.draft_model_config.architectures = [architecture]
    r.selector.dtype = dtype
    r.config.speculative_config.enforce_eager = eager
    assert r.select((False, False, False), r.selector)


@pytest.mark.parametrize(
    "fault",
    [
        "no_config",
        "no_spec",
        "no_draft",
        "hardware",
        "tag",
        "target",
        "method",
        "architecture",
        "adaptive",
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
        "dtype",
        "kv_cache_dtype",
        "attn_type",
        "use_pcp",
        "use_dcp",
        "use_mm_prefix",
        "use_per_head_quant_scales",
        "use_batch_invariant",
        "use_rswa",
        "use_adaptive_verification",
        "sparse_backend",
    ],
)
@pytest.mark.parametrize("eager", [False, True])
def test_flash_draft_unsupported_paths_fall_back(selector_runtime, fault, eager):
    r = selector_runtime
    spec = r.config.speculative_config
    spec.enforce_eager = eager
    key = (False, False, False)
    if fault == "no_config":
        r.config = None
    elif fault == "no_spec":
        r.config.speculative_config = None
    elif fault == "no_draft":
        spec.draft_model_config = None
    elif fault == "hardware":
        r.supported = False
    elif fault == "tag":
        r.backends.model_tag = "backbone"
    elif fault == "target":
        spec.target_model_config.hf_text_config.model_type = "other"
    elif fault == "method":
        spec.method = "eagle3"
    elif fault == "architecture":
        spec.draft_model_config.architectures = ["OtherDraft"]
    elif fault == "adaptive":
        spec.enable_adaptive_verification = True
    elif fault.endswith("parallel_size"):
        setattr(r.config.parallel_config, fault, 2)
    elif fault == "dtype":
        r.selector.dtype = torch.float32
    elif fault == "kv_cache_dtype":
        r.selector.kv_cache_dtype = "fp8"
    elif fault == "attn_type":
        r.selector.attn_type = "encoder"
    elif fault == "sparse_backend":
        key = (True, True, False)
    else:
        setattr(r.selector, fault, True)
    assert not r.select(key, r.selector)


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("architecture", ["MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"])
def test_original_qwen3_dspark_graph_and_eager_support_is_retained(selector_runtime, eager, architecture):
    r = selector_runtime
    spec = r.config.speculative_config
    spec.enforce_eager = eager
    spec.target_model_config.hf_config.architectures = [architecture]
    spec.target_model_config.hf_text_config.model_type = "minimax_m3"
    spec.draft_model_config.architectures = ["Qwen3DSparkModel"]
    assert r.select((False, False, False), r.selector)
    r.backends.model_tag = "backbone"
    assert not r.select((False, False, False), r.selector)


def test_glm_dflash_selection_is_retained(selector_runtime):
    r = selector_runtime
    spec = r.config.speculative_config
    spec.method = "dflash"
    spec.target_model_config.hf_text_config.model_type = "glm_moe_dsa"
    spec.draft_model_config.architectures = ["DFlash2DraftModel"]
    r.backends.model_tag = "dflash_head"
    assert r.select((False, False, False), r.selector)
    spec.enforce_eager = False
    assert not r.select((False, False, False), r.selector)


@pytest.mark.parametrize("eager", [False, True])
def test_flash_draft_unsupported_head_size_falls_back(selector_runtime, eager):
    r = selector_runtime
    r.config.speculative_config.enforce_eager = eager
    r.selector.head_size = 72
    assert not r.select((False, False, False), r.selector)
