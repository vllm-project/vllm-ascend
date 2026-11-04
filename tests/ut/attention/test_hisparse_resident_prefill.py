# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing contracts; native output equivalence requires NPU qualification."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


def load_route(context, capturing=False):
    path = Path(__file__).parents[3] / "vllm_ascend/attention/sfa_hisparse_optimized.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendHiSparseImpl")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_try_resident_c8_prefill"
    )
    namespace = {
        "torch": NS(npu=NS(is_current_stream_capturing=lambda: capturing)),
        "CUDAGraphMode": NS(NONE="NONE"),
        "get_forward_context": lambda: context,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


def fixture():
    table = torch.zeros((8, 16), dtype=torch.int32)
    states = torch.tensor([6, 1], dtype=torch.int32)
    handles = [
        NS(
            all_context_pages_resident=True,
            dummy_batch=False,
            block_table=table.view_as(table),
            runtime=NS(max_swap_rows=128, request_state_indices=states),
            view=NS(block_size=128, attention_block_stride=640, cache=torch.empty((32, 128, 656))),
        )
        for _ in range(2)
    ]
    calls = []

    def convert(layer, logical, metadata, **kwargs):
        # An independent current-state oracle exposes stale-order mistakes.
        calls.append((layer, logical, metadata, kwargs))
        return states[metadata.req_id_per_token.long(), None] * 640 + logical, None

    def attention(query, rope, cache, physical, **kwargs):
        calls.append((query, rope, cache, physical, kwargs))
        return physical

    owner = NS(
        enable_sparse_sfa_c8=True,
        hisparse_cache=handles[1],
        index_group_index=1,
        index_group=NS(caches=handles, cache=lambda index: handles[index], convert_logical_to_physical_topk=convert),
        vllm_config=NS(kv_transfer_config=NS(kv_connector="HiSparseConnector")),
        _attention_physical=attention,
    )
    metadata = NS(num_prefills=1, num_actual_tokens=129, num_input_tokens=129, req_id_per_token=torch.arange(129) % 2)
    return owner, metadata, handles, states, calls


def test_current_request_order_and_encoded_rows():
    owner, metadata, handles, states, calls = fixture()
    route = load_route(NS(cudagraph_runtime_mode="NONE"))
    logical = torch.arange(129, dtype=torch.int32)[:, None]
    query, rope = object(), object()
    first = route(owner, query, rope, logical, metadata)
    torch.testing.assert_close(first, states[metadata.req_id_per_token.long(), None] * 640 + logical)
    assert calls[-1][2] is handles[1].view.cache
    assert calls[-1][-1] == {"_c8_physical_rows": True}
    assert calls[0][-1] == {"block_stride_rows": 640, "return_valid_counts": True}
    states.copy_(torch.tensor([3, 7]))
    metadata.req_id_per_token.copy_(1 - metadata.req_id_per_token)
    second = route(owner, query, rope, logical, metadata)
    torch.testing.assert_close(second, states[metadata.req_id_per_token.long(), None] * 640 + logical)
    assert not torch.equal(first, second)


@pytest.mark.parametrize(
    "change",
    [
        "bf16",
        "decode",
        "padding_actual",
        "padding_input",
        "small",
        "graph",
        "capture",
        "composite",
        "nonresident",
        "dummy",
        "unbound_view",
        "unbound_table",
        "unbound_states",
        "new_table",
        "block_size",
        "stride",
        "cache_shape",
    ],
)
def test_unsupported_routes_keep_original_path(change):
    owner, metadata, handles, _, calls = fixture()
    context = NS(cudagraph_runtime_mode="NONE")
    if change == "bf16":
        owner.enable_sparse_sfa_c8 = False
    elif change == "decode":
        metadata.num_prefills = 0
    elif change == "padding_actual":
        metadata.num_actual_tokens -= 1
    elif change == "padding_input":
        metadata.num_input_tokens += 1
    elif change == "small":
        handles[1].runtime.max_swap_rows = 129
    elif change == "graph":
        context.cudagraph_runtime_mode = "FULL"
    elif change == "composite":
        owner.vllm_config.kv_transfer_config.kv_connector = "MultiConnector"
    elif change == "nonresident":
        handles[1].all_context_pages_resident = False
    elif change == "dummy":
        handles[1].dummy_batch = True
    elif change == "unbound_view":
        handles[0].view = None
    elif change == "unbound_table":
        handles[0].block_table = None
    elif change == "unbound_states":
        handles[0].runtime.request_state_indices = None
    elif change == "new_table":
        handles[1].block_table = handles[1].block_table.clone()
    elif change == "block_size":
        handles[1].view.block_size = 64
    elif change == "stride":
        handles[1].view.attention_block_stride += 128
    elif change == "cache_shape":
        handles[1].view.cache = handles[1].view.cache[:16]
    route = load_route(context, capturing=change == "capture")
    assert route(owner, object(), object(), torch.zeros((129, 1), dtype=torch.int32), metadata) is None
    assert not calls
