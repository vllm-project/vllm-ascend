# SPDX-License-Identifier: Apache-2.0
"""CPU regression for Flash draft metadata in PP prefill MoE selection."""

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def classification():
    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    scope = {
        "torch": torch,
        "envs": SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        "is_pd_decode_recompute_scheduler_enabled": lambda: False,
        "_build_flash_attention_metadata": lambda *args, **kwargs: None,
        "get_ascend_config": lambda: SimpleNamespace(enable_fused_mc2=1),
        "is_mega_moe_supported": lambda: True,
        "MoECommType": SimpleNamespace(FUSED_MC2="fused", MC2="mc2", ALLGATHER="ag", ALLTOALL="a2a"),
    }

    def execute(path, nodes):
        tree = ast.Module(body=nodes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec", __future__.annotations.compiler_flag), scope)

    # Run the real CPU classification and builder; only Flash's device metadata
    # construction is stubbed, so this needs neither vLLM imports nor an NPU.
    path = root / "attention/utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    execute(path, [node for node in tree.body if getattr(node, "name", None) == "split_decodes_and_prefills"])
    path = root / "attention/attention_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if getattr(node, "name", None) == "AscendAttentionMetadataBuilder")
    cls.bases = []
    cls.body = [node for node in cls.body if getattr(node, "name", None) in {"build", "_split_decodes_and_prefills"}]
    execute(path, [cls])

    path = root / "ascend_forward_context.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    execute(
        path,
        [
            node
            for node in tree.body
            if getattr(node, "name", None) == "_select_capacity_and_world_size_moe_comm_method"
        ],
    )
    # Exercise both MRv1 and MRv2's actual pure-prefill predicates.
    predicates = []
    for filename in ("ascend_forward_context.py", "platform.py"):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "is_pure_prefill" for target in node.targets)
        )
        predicates.append(compile(ast.Expression(assignment.value), filename, "eval"))
    return SimpleNamespace(**scope), predicates


@pytest.mark.parametrize("dp_size", [1, 8])
@pytest.mark.parametrize(
    "query_lens,threshold,expected",
    [
        ([4096, 8192], 4, (0, 2, 0)),  # Pure prefill, including the PP4 draft group.
        ([1, 1], 4, (2, 0, 2)),  # Ordinary decode.
        ([4, 4], 4, (2, 0, 8)),  # DSpark3 verification.
        ([8, 8], 8, (2, 0, 16)),  # DSpark7 verification.
        ([4, 4096], 4, (1, 1, 4)),  # Mixed decode/prefill.
        ([1, 4], 4, (2, 0, 5)),  # Preserve FIA's short-extend classification.
        ([], 4, (0, 0, 0)),  # Empty/dummy batch must not select MegaMoE.
    ],
)
def test_flash_draft_counts_preserve_prefill_only_megamoe(classification, query_lens, threshold, expected, dp_size):
    module, predicates = classification
    starts = torch.tensor([0, *query_lens], dtype=torch.int32).cumsum(0)
    common = SimpleNamespace(
        num_reqs=len(query_lens),
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens, default=0),
        query_start_loc_cpu=starts,
        query_start_loc=starts,
        context_parallel_metadata=None,
        seq_lens=None,
        block_table_tensor=None,
        slot_mapping=None,
        causal=True,
        attn_state=None,
    )
    builder = module.AscendAttentionMetadataBuilder()
    builder.decode_threshold = threshold
    builder.pcp_enabled = False
    builder.metadata_cls = lambda **kwargs: SimpleNamespace(
        **({"num_decodes": 0, "num_prefills": 0, "num_decode_tokens": 0} | kwargs)
    )
    draft = builder.build(0, common)
    assert (draft.num_decodes, draft.num_prefills, draft.num_decode_tokens) == expected
    target = SimpleNamespace(num_decodes=expected[0], num_prefills=expected[1])
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_experts_per_tok=8)),
        parallel_config=SimpleNamespace(data_parallel_size=dp_size, world_size_across_dp=32),
    )
    for predicate in predicates:
        for metadata in ({"target": target}, {"target": target, "draft": draft}):
            pure_prefill = eval(predicate, {"attn_metadata": metadata})
            selected = module._select_capacity_and_world_size_moe_comm_method(
                common.num_actual_tokens,
                config,
                512,
                cann_mega_moe_supported=True,
                is_pure_prefill=pure_prefill,
            )
            assert (selected == "fused") is (expected[1] > 0 and expected[0] == 0 and dp_size == 1)
