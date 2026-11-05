# SPDX-License-Identifier: Apache-2.0

"""Import-compatibility guards for the context-parallel refactor.

The PCP/DCP decoupling moves shared symbols out of the GQA v1 backend module
and out of the worker layer while keeping their historical import paths
working. These tests pin that contract and the resulting dependency direction;
follow-up changes of the same refactor extend this file.
"""

import ast
from dataclasses import MISSING, fields
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

from vllm_ascend.attention.attention_v1 import AscendAttentionState as FromV1
from vllm_ascend.attention.context_parallel import common_cp, metadata
from vllm_ascend.attention.context_parallel.context import (
    AscendPCPAttentionContext as FromContext,
)
from vllm_ascend.attention.context_parallel.context import (
    PCPAttentionContext,
)
from vllm_ascend.attention.state import AscendAttentionState as FromState
from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext as FromManager


def test_attention_v1_reexports_attention_state() -> None:
    # The historical GQA v1 import path must hand out the very same enum object
    # that is now defined in the shared, backend-agnostic state module.
    assert FromV1 is FromState
    assert issubclass(FromState, Enum)
    assert [(member.name, member.value) for member in FromState] == [
        ("PrefillNoCache", 0),
        ("PrefillCacheHit", 1),
        ("DecodeOnly", 2),
        ("ChunkedPrefill", 3),
        ("SpecDecoding", 4),
    ]
    assert FromState.__module__ == "vllm_ascend.attention.state"


def test_pcp_manager_reexports_attention_context() -> None:
    # The worker PCP manager must re-export the very same dataclass that now
    # lives in the attention layer, so historical imports keep working.
    assert FromManager is FromContext
    assert FromContext.__module__ == "vllm_ascend.attention.context_parallel.context"
    field_names = [field.name for field in fields(FromContext)]
    assert field_names == [
        "global_batch",
        "global_block_tables",
        "global_slot_mappings",
        "hidden_restore_idx",
        "padded_gather_idx",
        "gathered_kv_write_mask",
        "global_block_table_num_blocks",
        "local_to_global_req_indices",
        "local_token_indices",
    ]
    default_by_name = {field.name: field.default for field in fields(FromContext)}
    required, optional = field_names[:4], field_names[4:]
    assert all(default_by_name[name] is MISSING for name in required)
    assert all(default_by_name[name] is None for name in optional)


def test_context_contract_module_does_not_import_worker() -> None:
    module_path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/context_parallel/context.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))

    # Only top-level statements are runtime imports; imports nested in an
    # `if TYPE_CHECKING:` block (an ast.If node) are not executed at runtime.
    runtime_imports: list[str] = []
    type_checking_imports: list[str] = []
    has_future_annotations = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            runtime_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            runtime_imports.append(module)
            has_future_annotations = has_future_annotations or (
                module == "__future__" and any(alias.name == "annotations" for alias in node.names)
            )
        elif isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            type_checking_imports.extend(child.module or "" for child in node.body if isinstance(child, ast.ImportFrom))

    # The attention layer must not import the worker layer (or vLLM) at
    # runtime; the worker-only type is a type-checking-only dependency and the
    # module postpones annotations so that it stays that way.
    assert not any(module.startswith("vllm_ascend.worker") for module in runtime_imports)
    assert not any(module == "vllm" or module.startswith(("vllm.", "vllm_ascend.")) for module in runtime_imports)
    assert has_future_annotations
    assert "vllm_ascend.worker.v2.input_batch" in type_checking_imports

    assert PCPAttentionContext.__name__ == "PCPAttentionContext"
    # `runtime_checkable` protocols with data members support isinstance() but
    # not issubclass(); exercise both directions of the former.
    matching = SimpleNamespace(
        global_batch=object(),
        global_block_tables=(),
        global_slot_mappings=None,
        hidden_restore_idx=None,
        padded_gather_idx=None,
        gathered_kv_write_mask=None,
        global_block_table_num_blocks=None,
        local_to_global_req_indices=None,
        local_token_indices=None,
    )
    assert isinstance(matching, PCPAttentionContext)
    assert not isinstance(SimpleNamespace(global_batch=object()), PCPAttentionContext)


def test_common_cp_reexports_moved_symbols_are_identical() -> None:
    # The policy module must keep re-exporting every metadata symbol that moved
    # to `metadata`, so historical imports from `common_cp` keep working.
    moved_symbols = [
        "CPKVScope",
        "DCPImplMixin",
        "DCPMetadataBuilderMixin",
        "_npu_attention_update",
        "_npu_attn_out_lse_update",
        "_out_lse_reshape",
        "_process_attn_out_lse",
        "_update_out_and_lse",
        "build_pcp_ordered_slot_mapping",
        "use_history_current_split_decode",
    ]
    for name in moved_symbols:
        assert getattr(common_cp, name) is getattr(metadata, name), name
    assert [(member.name, member.value) for member in common_cp.CPKVScope] == [
        ("HISTORY", "history"),
        ("CURRENT", "current"),
        ("FULL", "full"),
    ]
