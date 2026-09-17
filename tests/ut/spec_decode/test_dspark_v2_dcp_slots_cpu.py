# SPDX-License-Identifier: Apache-2.0
"""Exercise the NPU input kernel's scalar control flow without an NPU import.

The CP helper is emulated by enumerating each rank's resident positions. These
tests cover the caller's global block lookup, rejected rows and graph reuse;
device compilation and the upstream helper itself require the NPU runtime.
"""

import ast
from pathlib import Path

import numpy as np
import pytest


class Scalar(int):
    def to(self, _dtype):
        return self


class Pointer:
    def __init__(self, values, offset=0):
        self.values = values
        self.offset = offset

    def __add__(self, offset):
        return Pointer(self.values, self.offset + int(offset))

    def __sub__(self, offset):
        return self + -int(offset)


class ScalarTL:
    int32 = int64 = int

    def __init__(self, num_reqs):
        self.num_reqs = num_reqs
        self.req = 0

    def program_id(self, axis):
        return self.req if axis == 0 else 0

    def num_programs(self, _axis):
        return self.num_reqs

    @staticmethod
    def load(ptr, mask=True, other=0):
        return Scalar(ptr.values[ptr.offset] if mask else other)

    @staticmethod
    def store(ptr, value):
        ptr.values[ptr.offset] = value

    minimum = staticmethod(min)
    where = staticmethod(lambda condition, yes, no: yes if condition else no)


def rank_positions(block_size, cp_size, interleave, rank):
    return [position for position in range(block_size * cp_size) if position // interleave % cp_size == rank]


def emulate_cp_slot(position, block_id, block_size, rank, cp_size, interleave, pad):
    positions = rank_positions(block_size, cp_size, interleave, rank)
    offset = int(position) % (block_size * cp_size)
    return block_id * block_size + positions.index(offset) if offset in positions else pad


def load_kernel(tl):
    source = Path(__file__).parents[3] / "vllm_ascend/worker/v2/spec_decode/dflash/speculator.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    # The second version is the pinned main2main signature with DCP inputs.
    definition = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare_dflash_inputs_kernel_ascend"
    ][-1]
    definition.decorator_list = []
    namespace = {"tl": tl, "cp_local_slot": emulate_cp_slot}
    exec(compile("from __future__ import annotations\n" + ast.unparse(definition), str(source), "exec"), namespace)
    return namespace[definition.name]


def make_buffers(cp_size, interleave):
    virtual_block = 128 * cp_size
    # Request 0 crosses a global block boundary; request 1's accepted context
    # and initial queries hit an evicted block (physical block 0).
    positions = np.array(
        list(range(virtual_block - 2, virtual_block + 3))
        + list(range(2 * virtual_block + interleave - 2, 2 * virtual_block + interleave + 2))
    )
    sizes = {
        "input_ids": 28,
        "query_positions": 28,
        "query_start_loc": 5,
        "seq_lens": 4,
        "query_slot_mapping": 28,
        "context_positions": len(positions),
        "context_slot_mapping": len(positions),
        "sample_indices": 28,
        "sample_pos": 28,
        "sample_idx_mapping": 28,
        "temperature": 4,
        "seeds": 4,
    }
    outputs = {f"out_{name}_ptr": np.full(size, 9999, dtype=np.int64) for name, size in sizes.items()}
    inputs = {
        "target_positions_ptr": positions,
        "target_query_start_loc_ptr": np.array([0, 5, 9]),
        "idx_mapping_ptr": np.array([2, 0]),
        "last_sampled_ptr": np.array([42, 0, 73, 0]),
        "next_prefill_tokens_ptr": np.array([29, 0, 31, 0]),
        "num_sampled_ptr": np.array([1, 0]),
        "num_rejected_ptr": np.array([2, 1]),
        "temperature_ptr": np.array([1, 2, 3, 4]),
        "seeds_ptr": np.array([11, 12, 13, 14]),
        "block_table_ptr": np.array([[3, 7, 13, 17], [19, 23, 0, 31]]).ravel(),
    }
    return outputs, inputs


def run_kernel(tl, outputs, inputs, cp_size, interleave, rank):
    kernel = load_kernel(tl)
    kwargs = {name: Pointer(value) for name, value in {**outputs, **inputs}.items()}
    kwargs.update(
        block_table_stride=4,
        parallel_drafting_token_id=99,
        block_size=128,
        num_query_per_req=7,
        num_speculative_steps=7,
        max_num_reqs=4,
        max_num_tokens=28,
        max_model_len=4096,
        cp_rank=rank,
        SAMPLE_FROM_ANCHOR=True,
        PAD_SLOT_ID=-1,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=interleave,
        BLOCK_SIZE=128,
    )
    for req in range(tl.num_reqs):
        tl.req = req
        kernel(**kwargs)


@pytest.mark.parametrize("cp_size,interleave", [(1, 1), (2, 1), (4, 1), (2, 128), (4, 128)])
def test_context_and_query_have_exactly_one_resident_owner(cp_size, interleave):
    results = []
    for rank in range(cp_size):
        outputs, inputs = make_buffers(cp_size, interleave)
        run_kernel(ScalarTL(2), outputs, inputs, cp_size, interleave, rank)
        results.append(outputs)

    context_slots = np.stack([result["out_context_slot_mapping_ptr"] for result in results])
    query_slots = np.stack([result["out_query_slot_mapping_ptr"] for result in results])
    positions = inputs["target_positions_ptr"]
    tables = inputs["block_table_ptr"].reshape(2, 4)
    for req, start, accepted in [(0, 0, 3), (1, 5, 3)]:
        query_positions = list(
            range(int(positions[start + accepted - 1]) + 1, int(positions[start + accepted - 1]) + 8)
        )
        for offsets, slots, slot_start in [
            (positions[start : start + accepted], context_slots, start),
            (query_positions, query_slots, req * 7),
        ]:
            for index, position in enumerate(offsets):
                block_id = tables[req, position // (128 * cp_size)]
                owners = np.flatnonzero(slots[:, slot_start + index] != -1)
                if block_id == 0:
                    assert owners.size == 0
                else:
                    owner = position // interleave % cp_size
                    assert owners.tolist() == [owner]
                    resident = rank_positions(128, cp_size, interleave, owner)
                    local_offset = resident.index(position % (128 * cp_size))
                    assert slots[owner, slot_start + index] == block_id * 128 + local_offset

    # Rejected tokens are fully initialized, including their unused positions.
    assert np.all(context_slots[:, [3, 4, 8]] == -1)
    for outputs in results:
        assert outputs["out_context_positions_ptr"][[3, 4, 8]].tolist() == [0, 0, 0]
        assert outputs["out_input_ids_ptr"][:14].tolist() == [73] + [99] * 6 + [29] + [99] * 6
        assert outputs["out_query_start_loc_ptr"].tolist() == [0, 7, 14, 14, 14]
        assert outputs["out_seq_lens_ptr"][2:].tolist() == [0, 0]
        assert np.all(outputs["out_query_slot_mapping_ptr"][14:] == -1)
        assert np.all(outputs["out_sample_idx_mapping_ptr"][14:] == -1)


def test_smaller_replay_clears_previous_request_slots():
    outputs, inputs = make_buffers(2, 1)
    run_kernel(ScalarTL(2), outputs, inputs, 2, 1, 0)
    run_kernel(ScalarTL(1), outputs, inputs, 2, 1, 0)
    assert outputs["out_query_start_loc_ptr"].tolist() == [0, 7, 7, 7, 7]
    assert outputs["out_seq_lens_ptr"][1:].tolist() == [0, 0, 0]
    assert np.all(outputs["out_query_slot_mapping_ptr"][7:] == -1)
    assert np.all(outputs["out_sample_idx_mapping_ptr"][7:] == -1)
    assert np.all(outputs["out_sample_indices_ptr"][7:] == 0)
