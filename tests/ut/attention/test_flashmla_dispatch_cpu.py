# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the default CANN package exports without NPU dependencies."""

import ast
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]

# Model the serving-compatible form of the dcp.py API. No legacy Python
# wrappers or optional window arguments are needed for modes 0 and 3.
ATTENTION_SCHEMA = """flash_mla(Tensor q, Tensor k_cache, *,
    Tensor? block_table=None, Tensor? cache_seqlens=None, Tensor? cu_seqlens_q=None,
    Tensor? seqused_q=None, Tensor? attn_mask=None, Tensor? metadata=None,
    int head_dim_v=512, float softmax_scale=1.0, int mask_mode=0,
    int max_seqlen_q=-1, int max_seqlen_kv=-1, str layout_q="TND",
    str layout_kv="PA_BBND", str layout_out="TND", bool return_softmax_lse=False
) -> (Tensor, Tensor)"""
METADATA_SCHEMA = """flash_mla_metadata(Tensor cache_seqlens, int num_heads_q, int num_heads_kv, *,
    Tensor? cu_seqlens_q=None, Tensor? seqused_q=None,
    int max_seqlen_q=-1, int max_seqlen_kv=-1, int head_dim_qk=576, int head_dim_v=512,
    int mask_mode=0, str layout_q="TND", bool enable_dcp=False, int cp_world=1, int cp_rank=0,
    int cp_kv_cache_interleave_size=128, Tensor? casusal_seqlens_kv_global=None
) -> Tensor"""


@pytest.fixture
def loader():
    # Extract the production functions so importing unrelated vLLM/NPU code
    # cannot hide registration errors in this CPU-only contract check.
    tree = ast.parse((ROOT / "vllm_ascend/attention/utils.py").read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_has_flashmla_module", "get_flashmla_ops")
    ]
    module = ModuleType("flashmla_dispatch_test")
    module.__dict__.update(
        Callable=Callable,
        lru_cache=lru_cache,
        find_spec=Mock(return_value=object()),
        import_module=Mock(),
        torch=SimpleNamespace(ops=SimpleNamespace(cann_ops_transformer=SimpleNamespace())),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<flashmla-loader>", "exec"), module.__dict__)
    return module


def registered_ops(metadata_schema=METADATA_SCHEMA):
    attention = Mock(_schema=torch._C.parse_schema(ATTENTION_SCHEMA))
    metadata = Mock(_schema=torch._C.parse_schema(metadata_schema))
    return SimpleNamespace(
        flash_mla=SimpleNamespace(default=attention),
        flash_mla_metadata=SimpleNamespace(default=metadata),
    )


def test_default_package_absent_skips_import_and_operator_lookup(loader):
    loader.find_spec.return_value = None
    assert loader.get_flashmla_ops() is None
    loader.find_spec.assert_called_once_with("cann_ops_transformer")
    loader.import_module.assert_not_called()


def test_default_package_does_not_require_custom_distribution(loader):
    loader.find_spec.side_effect = lambda name: object() if name == "cann_ops_transformer" else None
    namespace = registered_ops()
    loader.import_module.return_value = namespace
    assert loader.get_flashmla_ops() == (namespace.flash_mla.default, namespace.flash_mla_metadata.default)
    loader.find_spec.assert_called_once_with("cann_ops_transformer")
    loader.import_module.assert_called_once_with("cann_ops_transformer.ops")


def test_default_package_public_exports_are_used(loader):
    namespace = registered_ops()
    # A dispatcher handle from another import must not override the exports
    # selected by the default package (including its entry-point policy).
    loader.torch.ops.cann_ops_transformer = registered_ops()

    def register(name):
        assert name == "cann_ops_transformer.ops"
        return namespace

    loader.import_module.side_effect = register
    attention, metadata = loader.get_flashmla_ops()
    assert attention is namespace.flash_mla.default
    assert metadata is namespace.flash_mla_metadata.default
    loader.import_module.assert_called_once_with("cann_ops_transformer.ops")


def test_default_package_missing_during_import_returns_none(loader):
    loader.import_module.side_effect = ModuleNotFoundError("missing default package", name="cann_ops_transformer")
    assert loader.get_flashmla_ops() is None


def test_default_package_internal_import_failure_is_not_hidden(loader):
    loader.import_module.side_effect = ModuleNotFoundError("missing internal_dependency", name="internal_dependency")
    with pytest.raises(ModuleNotFoundError, match="internal_dependency"):
        loader.get_flashmla_ops()


def test_installed_package_missing_ops_module_is_not_silently_disabled(loader):
    loader.import_module.side_effect = ModuleNotFoundError("missing ops module", name="cann_ops_transformer.ops")
    with pytest.raises(ModuleNotFoundError, match="missing ops module"):
        loader.get_flashmla_ops()


def test_incompatible_package_exports_do_not_use_stale_dispatcher(loader):
    loader.torch.ops.cann_ops_transformer = registered_ops()
    loader.import_module.return_value = SimpleNamespace()
    with pytest.raises(RuntimeError, match="must export.*flash_mla"):
        loader.get_flashmla_ops()


@pytest.mark.parametrize("missing", ["flash_mla", "flash_mla_metadata"])
def test_installed_package_requires_both_exported_operators(loader, missing):
    namespace = registered_ops()
    delattr(namespace, missing)
    loader.import_module.return_value = namespace
    with pytest.raises(RuntimeError, match=f"must export.*{missing}"):
        loader.get_flashmla_ops()


@pytest.mark.parametrize("argument", ["enable_dcp", "cp_rank", "casusal_seqlens_kv_global", "seqused_q"])
def test_rejects_old_metadata_schema_before_cache_allocation(loader, argument):
    schema = torch._C.parse_schema(METADATA_SCHEMA)
    namespace = registered_ops()
    namespace.flash_mla_metadata.default._schema = SimpleNamespace(
        arguments=[item for item in schema.arguments if item.name != argument]
    )
    loader.import_module.return_value = namespace
    with pytest.raises(RuntimeError, match=f"incompatible flash_mla_metadata.*{argument}"):
        loader.get_flashmla_ops()


def test_rejects_incompatible_attention_schema(loader):
    namespace = registered_ops()
    namespace.flash_mla.default._schema = torch._C.parse_schema("flash_mla(Tensor q, Tensor k_cache) -> Tensor")
    loader.import_module.return_value = namespace
    with pytest.raises(RuntimeError, match="incompatible flash_mla schema"):
        loader.get_flashmla_ops()
