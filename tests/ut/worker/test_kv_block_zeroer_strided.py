# SPDX-License-Identifier: Apache-2.0
"""CPU check of VA zeroer metadata and scheduler-to-kernel page mapping."""

import ast
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import torch


class FullAttentionSpec:
    block_size = 32


class RecordingZeroKernel:
    def __init__(self, allocations):
        self.allocations = allocations

    def __getitem__(self, grid):
        assert 0 < grid[0] <= 8

        def launch(addresses, block_ids, n_blocks, strides, sizes, **constants):
            assert constants["BLOCK_SIZE"] <= 8192
            assert constants["PAGE_SIZE_EL"] % constants["BLOCK_SIZE"] == 0
            for index, address in enumerate(addresses.tolist()):
                pitch = strides[index].item() if constants["PAGE_STRIDED"] else constants["PAGE_SIZE_EL"]
                payload = sizes[index].item() if constants["PAGE_STRIDED"] else constants["PAGE_SIZE_EL"]
                assert payload <= constants["PAGE_SIZE_EL"]
                for block in block_ids[:n_blocks].tolist():
                    start, length = address + block * pitch * 4, payload * 4
                    for allocation in self.allocations:
                        offset = start - allocation.data_ptr()
                        if offset >= 0 and offset + length <= allocation.numel() * allocation.element_size():
                            allocation.view(torch.uint8).reshape(-1)[offset : offset + length].zero_()
                            break
                    else:
                        raise AssertionError("zeroing interval exceeds its backing allocation")

        return launch


def load_zeroer(allocations):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definition = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendKVBlockZeroer"
    )
    namespace = {
        "torch": torch,
        "KVBlockZeroer": object,
        "FullAttentionSpec": FullAttentionSpec,
        "iprod": product,
        "largest_power_of_2_divisor": lambda value: value & -value,
        "get_vectorcore_num": lambda: 8,
        "_zero_kv_blocks_kernel": RecordingZeroKernel(allocations),
    }
    exec(compile("from __future__ import annotations\n" + ast.unparse(definition), str(source), "exec"), namespace)
    return namespace["AscendKVBlockZeroer"](torch.device("cpu"), pin_memory=False)


def test_strided_payloads_preserve_layer_gaps_and_legacy_tuple_mapping():
    # One logical scheduler block spans two kernel pages. The MLA and GQA
    # layers deliberately have different payloads, pitches, and offsets.
    allocations, caches, expected = [], {}, []
    for name, shape, gap, offset in (
        ("mla", (6, 16, 576), 64, 32),
        ("gqa", (6, 4, 16, 64), 128, 64),
    ):
        payload = torch.Size(shape[1:]).numel()
        pitch = 3 * payload + gap
        raw = torch.full((offset + 6 * pitch + 64,), 91, dtype=torch.bfloat16)
        dense_strides = torch.empty(shape).stride()
        cache = raw.as_strided(shape, (pitch, *dense_strides[1:]), offset)
        reference = raw.clone()
        reference.as_strided(shape, cache.stride(), offset)[2:4].zero_()
        allocations.append(raw)
        expected.append(reference)
        caches[name] = SimpleNamespace(kv_cache=cache)
    group = SimpleNamespace(kv_cache_spec=FullAttentionSpec(), kv_cache_group_id=0, layer_names=list(caches))
    zeroer = load_zeroer(allocations)
    zeroer.init_meta([group], [[16]], "auto", set(), caches, page_strided=True)
    ids_pinned, ids_device = zeroer._ids_pinned, zeroer._ids_gpu
    zeroer.zero_block_ids([1])
    zeroer.zero_block_ids([])
    assert zeroer._ids_pinned is ids_pinned and zeroer._ids_gpu is ids_device
    for actual, reference in zip(allocations, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    # The default tuple route retains its original contiguous-page behavior.
    kv = tuple(torch.full((6, 2, 16, 64), 73, dtype=torch.bfloat16) for _ in range(2))
    expected_kv = tuple(tensor.clone() for tensor in kv)
    for tensor in expected_kv:
        tensor[2:4].zero_()
    zeroer = load_zeroer(kv)
    group.layer_names = ["legacy"]
    zeroer.init_meta([group], [[16]], "auto", set(), {"legacy": SimpleNamespace(kv_cache=kv)})
    zeroer.zero_block_ids([1])
    for actual, reference in zip(kv, expected_kv):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_replicated_draft_zeroes_all_dcp_lanes_of_only_the_selected_manager_block():
    class ReplicatedDraftSpec(FullAttentionSpec):
        block_size = 768
        dcp_replication_size = 8

    blocks, ratio, replication = 3, 6, 8
    shape = (blocks * replication * ratio, 4, 128, 64)
    draft = torch.full(shape, 91, dtype=torch.bfloat16)
    target = torch.full((blocks * ratio, 128, 576), 73, dtype=torch.bfloat16)
    expected_draft, expected_target = draft.clone(), target.clone()
    expected_draft[replication * ratio : 2 * replication * ratio].zero_()
    expected_target[ratio : 2 * ratio].zero_()
    target_spec = FullAttentionSpec()
    target_spec.block_size = 768
    groups = [
        SimpleNamespace(kv_cache_spec=target_spec, kv_cache_group_id=0, layer_names=["target"]),
        SimpleNamespace(kv_cache_spec=ReplicatedDraftSpec(), kv_cache_group_id=0, layer_names=["draft"]),
    ]
    zeroer = load_zeroer([target, draft])
    zeroer.init_meta(
        groups,
        [[128]],
        "auto",
        set(),
        {"target": SimpleNamespace(kv_cache=target), "draft": SimpleNamespace(kv_cache=draft)},
        page_strided=True,
    )
    zeroer.zero_block_ids([1])
    torch.testing.assert_close(draft, expected_draft, rtol=0, atol=0)
    torch.testing.assert_close(target, expected_target, rtol=0, atol=0)
