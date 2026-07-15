# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_glm47_patch_is_noop_without_legacy_private_method(monkeypatch):
    parser_module_name = "vllm.tool_parsers.glm47_moe_tool_parser"
    parser_module = ModuleType(parser_module_name)

    class ParserWithoutLegacyMethod:
        pass

    parser_module.Glm47MoeModelToolParser = ParserWithoutLegacyMethod
    monkeypatch.setitem(sys.modules, parser_module_name, parser_module)

    patch_path = REPO_ROOT / "vllm_ascend" / "patch" / "platform" / "patch_glm47_tool_call_parser.py"
    _load_module_from_path("test_glm47_patch_noop", patch_path)

    assert not hasattr(
        ParserWithoutLegacyMethod,
        "_ascend_original_extract_tool_call_regions",
    )


def test_glm47_patch_still_wraps_legacy_private_method(monkeypatch):
    parser_module_name = "vllm.tool_parsers.glm47_moe_tool_parser"
    parser_module = ModuleType(parser_module_name)

    class LegacyParser:
        arg_key_start = "<arg_key>"

        def _extract_tool_call_regions(self, text):
            return [(text, True)]

    parser_module.Glm47MoeModelToolParser = LegacyParser
    monkeypatch.setitem(sys.modules, parser_module_name, parser_module)

    patch_path = REPO_ROOT / "vllm_ascend" / "patch" / "platform" / "patch_glm47_tool_call_parser.py"
    _load_module_from_path("test_glm47_patch_legacy", patch_path)

    assert LegacyParser()._extract_tool_call_regions("get_time") == [("get_time\n", True)]


def test_rejection_sampler_helper_resolution_supports_vllm_024_names():
    legacy_global_lse = object()
    legacy_block_stats = object()
    insert_resampled = object()
    upstream_module = SimpleNamespace(
        _compute_global_lse=legacy_global_lse,
        _compute_block_stats_kernel=legacy_block_stats,
        _insert_resampled_kernel=insert_resampled,
    )

    compat_path = REPO_ROOT / "vllm_ascend" / "worker" / "v2" / "spec_decode" / "compat.py"
    compat = _load_module_from_path("test_rejection_sampler_compat", compat_path)

    assert compat.resolve_rejection_sampler_helpers(upstream_module) == (
        legacy_global_lse,
        legacy_block_stats,
        insert_resampled,
    )


def test_rejection_sampler_helper_resolution_supports_new_main_names():
    main_global_lse = object()
    main_block_stats = object()
    insert_resampled = object()
    upstream_module = SimpleNamespace(
        _compute_global_logsumexp=main_global_lse,
        _compute_local_logits_stats_kernel=main_block_stats,
        _insert_resampled_kernel=insert_resampled,
    )

    compat_path = REPO_ROOT / "vllm_ascend" / "worker" / "v2" / "spec_decode" / "compat.py"
    compat = _load_module_from_path("test_rejection_sampler_compat_main", compat_path)

    assert compat.resolve_rejection_sampler_helpers(upstream_module) == (
        main_global_lse,
        main_block_stats,
        insert_resampled,
    )


def _load_dflash_patch(monkeypatch, causal_lm_cls):
    torch_module = ModuleType("torch")
    torch_module.__path__ = []
    torch_module.Tensor = object

    torch_nn_module = ModuleType("torch.nn")
    torch_nn_module.__path__ = []
    torch_functional_module = ModuleType("torch.nn.functional")
    torch_nn_module.functional = torch_functional_module
    torch_module.nn = torch_nn_module

    qwen_module_name = "vllm.model_executor.models.qwen3_dflash"
    qwen_module = ModuleType(qwen_module_name)

    class DFlashModel:
        pass

    qwen_module.DFlashQwen3ForCausalLM = causal_lm_cls
    qwen_module.DFlashQwen3Model = DFlashModel

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn_module)
    monkeypatch.setitem(
        sys.modules,
        "torch.nn.functional",
        torch_functional_module,
    )
    monkeypatch.setitem(sys.modules, qwen_module_name, qwen_module)

    patch_path = REPO_ROOT / "vllm_ascend" / "patch" / "worker" / "patch_qwen3_dflash.py"
    _load_module_from_path("test_qwen3_dflash_patch", patch_path)


def test_dflash_patch_is_noop_without_optional_mask_embedding_method(monkeypatch):
    class DFlashWithoutMaskEmbedding:
        pass

    _load_dflash_patch(monkeypatch, DFlashWithoutMaskEmbedding)

    assert not hasattr(DFlashWithoutMaskEmbedding, "_read_mask_embedding")


def test_dflash_patch_wraps_optional_mask_embedding_method(monkeypatch):
    class DFlashWithMaskEmbedding:
        def _read_mask_embedding(self):
            raise RuntimeError("optional mask embedding is unavailable")

    _load_dflash_patch(monkeypatch, DFlashWithMaskEmbedding)

    assert DFlashWithMaskEmbedding()._read_mask_embedding() is None
