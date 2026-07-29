"""DSA 动态 eager 生命周期计划与满块复制任务构造器。

本模块从 ``DSAInputBatchState`` 读取已经确定的列式语义，在 model-forward
控制面完成 prefill/decode 满块目的地址预留，并把
request/hash/logical-block 行展平为 layer 可直接消费的物理 src/dst copy jobs。

single-token decode（包括首次进入 sparse 的 ENTER step）的 eager/graph 物理
镜像由 ``dsa_row_mode_runtime.py`` 负责。本模块不拥有图固定地址，也不维护
第二套 decode metadata adapter。
"""

from __future__ import annotations

import numpy as np
import torch

from vllm_ascend.dsa_sparse.dsa_forward_batch import (
    DSAForwardLayerHookPlan,
    DSAFullBlockDumpBatch,
)
from vllm_ascend.dsa_sparse.dsa_hot_kv_store_core import DSAHotKVStore
from vllm_ascend.dsa_sparse.dsa_input_batch_state import DSAInputBatchState
from vllm_ascend.dsa_sparse.dsa_types import (
    KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID,
    ReqType,
)


def _tensor_from_numpy(
    values: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Copy one contiguous NumPy column to the requested execution device."""
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    if tensor.device == device and tensor.dtype == dtype:
        return tensor
    return tensor.to(device=device, dtype=dtype)


def reserve_decode_full_block_dumps_from_input_state(
    input_state: DSAInputBatchState,
    *,
    dram_store: DSAHotKVStore | None,
) -> None:
    """Resolve decode dump destinations before row tensors are materialized.

    ``DSAInputBatchState.refresh`` has already detected boundary rows while
    walking the native batch. The common no-dump path is one integer check;
    Python lists are created only for the compact set of dump rows. Every
    eager/graph adapter calls this helper before reading the fixed src/dst
    columns, keeping the canonical sidecar as the shared semantic source.
    """
    if input_state.decode_dump_reservations_ready:
        return
    dump_count = int(input_state.decode_dump_row_count)
    if dump_count == 0:
        input_state.decode_dump_reservations_ready = True
        return
    if dram_store is None:
        raise RuntimeError("DSA decode full-block dump has no DRAM store")

    dump_rows = input_state.decode_dump_row_indices[:dump_count]
    request_ids: list[ReqType] = []
    request_pool_indices: list[int] = []
    block_hash_rows: list[list] = []
    logical_block_index_rows: list[list[int]] = []
    for raw_row in dump_rows:
        row = int(raw_row)
        req_id = input_state.request_ids[row]
        pool_idx = int(input_state.resident_pool_indices[row])
        logical_idx = int(
            input_state.decode_dump_logical_block_indices[row])
        hashes = input_state.context_full_block_hash_rows[row]
        if req_id is None or pool_idx < 0 or logical_idx < 0 or hashes is None or logical_idx >= len(hashes):
            raise RuntimeError(
                "DSA decode dump reservation metadata is incomplete: "
                f"row={row}, request={req_id!r}, pool_idx={pool_idx}, "
                f"logical_idx={logical_idx}"
            )
        request_ids.append(req_id)
        request_pool_indices.append(pool_idx)
        block_hash_rows.append([hashes[logical_idx]])
        logical_block_index_rows.append([logical_idx])

    destination_rows = dram_store.reserve_blocks_for_requests(
        request_ids=request_ids,
        request_pool_indices=request_pool_indices,
        block_hash_rows=block_hash_rows,
        logical_block_index_rows=logical_block_index_rows,
    )
    if len(destination_rows) != dump_count:
        raise RuntimeError(
            "DSA decode dump reservation returned the wrong row count: "
            f"expected={dump_count}, actual={len(destination_rows)}"
        )
    physical_destinations: list[int] = []
    for raw_row, destinations in zip(dump_rows, destination_rows):
        if len(destinations) != 1:
            raise RuntimeError(
                f"DSA single-token decode dump must reserve exactly one logical block, got {len(destinations)}"
            )
        destination = int(destinations[0])
        if destination < KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID:
            raise RuntimeError(
                "DSA decode dump destination must be -1 or a valid block "
                f"id, got {destination}")
        if destination >= 0:
            physical_destinations.append(destination)
        input_state.decode_dump_dst_dram_block_ids[int(raw_row)] = destination
    if len(set(physical_destinations)) != len(physical_destinations):
        raise RuntimeError(
            "DSA decode full-block dump has duplicate physical destinations")
    input_state.decode_dump_reservations_ready = True


def build_forward_layer_hook_plan_from_input_state(
    input_state: DSAInputBatchState,
    *,
    dram_store: DSAHotKVStore | None,
    include_decode_dump: bool = True,
    tensor_device: torch.device | str | None = None,
    empty_plan: DSAForwardLayerHookPlan | None = None,
) -> DSAForwardLayerHookPlan:
    """Build eager-only lifecycle work from the canonical row projection.

    Variable-length request/hash rows exist only while the DRAM allocator
    resolves ownership and physical destinations.  The returned layer plan
    retains only flat src/dst tensors, so every layer can invoke the same dump
    operator without re-walking requests, hashes or logical block tables.
    Steady graph replay uses the same data-plane class over persistent padded
    row-mode buffers and does not call this dynamic adapter.
    """
    device = torch.device("cpu") if tensor_device is None else torch.device(tensor_device)
    row_count = int(input_state.row_count)
    active_slice = slice(0, row_count)
    prefill_dump_mask = (input_state.num_output_tokens[active_slice] == 0) & input_state.last_prefill_chunk_mask[
        active_slice
    ]
    decode_dump_mask = (input_state.num_output_tokens[active_slice] > 0) & input_state.full_block_dump_mask[
        active_slice
    ]
    lifecycle_mask = prefill_dump_mask
    if include_decode_dump:
        lifecycle_mask = lifecycle_mask | decode_dump_mask
    lifecycle_rows = np.flatnonzero(lifecycle_mask)
    if int(lifecycle_rows.size) == 0 and empty_plan is not None:
        # Optimized eager decode has no lifecycle work. Reuse the worker-lifetime
        # empty object instead of allocating empty device tensors every step.
        return empty_plan

    dump_request_ids: list[ReqType] = []
    dump_request_pool_indices: list[int] = []
    dump_block_hash_rows: list[list] = []
    dump_block_id_rows: list[list[int]] = []
    dump_logical_block_index_rows: list[list[int]] = []
    dump_destination_block_id_rows: list[list[int]] = []
    dump_input_rows: list[int] = []
    reservation_plan_indices: list[int] = []
    for row in lifecycle_rows.tolist():
        req_id = input_state.request_ids[row]
        block_hashes = input_state.context_full_block_hash_rows[row]
        block_ids = input_state.budget_block_id_rows[row]
        if req_id is None or block_hashes is None or block_ids is None:
            raise RuntimeError(f"DSA InputBatch row {row} is incomplete")
        pool_idx = int(input_state.resident_pool_indices[row])
        if pool_idx < 0:
            raise RuntimeError(
                "DSA lifecycle row has no resident pool binding: "
                f"row={row}, request={req_id!r}")

        dump_hashes: list = []
        dump_block_ids: list[int] = []
        logical_block_indices: list[int] = []
        if bool(prefill_dump_mask[row]):
            num_full_blocks = len(block_hashes)
            if num_full_blocks > 0:
                dump_hashes = list(block_hashes)
                dump_block_ids = [
                    int(block_id)
                    for block_id in block_ids[:num_full_blocks]
                ]
                logical_block_indices = list(range(num_full_blocks))
        elif block_hashes:
            logical_block_idx = len(block_hashes) - 1
            dump_hashes = [block_hashes[-1]]
            dump_block_ids = [int(block_ids[-1])]
            logical_block_indices = [logical_block_idx]
        if not dump_block_ids:
            continue
        if dram_store is None:
            raise RuntimeError(f"DSA full-block dump request {req_id} has no DRAM block manager")
        dump_request_ids.append(req_id)
        dump_request_pool_indices.append(pool_idx)
        dump_block_hash_rows.append(dump_hashes)
        dump_block_id_rows.append(dump_block_ids)
        dump_logical_block_index_rows.append(logical_block_indices)
        dump_input_rows.append(row)
        plan_idx = len(dump_request_ids) - 1
        if bool(prefill_dump_mask[row]):
            dump_destination_block_id_rows.append([])
            reservation_plan_indices.append(plan_idx)
        elif input_state.decode_dump_reservations_ready:
            expected_source = int(input_state.decode_dump_src_hbm_block_ids[row])
            expected_logical = int(input_state.decode_dump_logical_block_indices[row])
            if dump_block_ids != [expected_source] or logical_block_indices != [expected_logical]:
                raise RuntimeError(
                    "DSA decode dump plan disagrees with canonical row "
                    f"metadata: row={row}, source={dump_block_ids}, "
                    f"logical={logical_block_indices}"
                )
            dump_destination_block_id_rows.append([int(input_state.decode_dump_dst_dram_block_ids[row])])
        else:
            # Direct/general eager adapters may call this builder without the
            # optimized manager boundary. Reserve only those decode rows here
            # and publish the result back to the canonical row projection.
            dump_destination_block_id_rows.append([])
            reservation_plan_indices.append(plan_idx)

    if reservation_plan_indices:
        if dram_store is None:
            raise RuntimeError("DSA full-block dump has no DRAM block manager")
        reserved_rows = dram_store.reserve_blocks_for_requests(
            request_ids=[
                dump_request_ids[idx] for idx in reservation_plan_indices
            ],
            request_pool_indices=[
                dump_request_pool_indices[idx]
                for idx in reservation_plan_indices
            ],
            block_hash_rows=[
                dump_block_hash_rows[idx] for idx in reservation_plan_indices
            ],
            logical_block_index_rows=[
                dump_logical_block_index_rows[idx]
                for idx in reservation_plan_indices
            ],
        )
        if len(reserved_rows) != len(reservation_plan_indices):
            raise RuntimeError(
                "DSA full-block reservation returned the wrong row count")
        for plan_idx, destinations in zip(
                reservation_plan_indices, reserved_rows):
            dump_destination_block_id_rows[plan_idx] = destinations
            input_row = dump_input_rows[plan_idx]
            if bool(decode_dump_mask[input_row]):
                if len(destinations) != 1:
                    raise RuntimeError(
                        "DSA decode dump must reserve exactly one block")
                input_state.decode_dump_dst_dram_block_ids[input_row] = int(
                    destinations[0])
        input_state.decode_dump_reservations_ready = True

    # Request ownership and hash bookkeeping stop at the model-forward
    # boundary. Layer hooks only consume aligned physical copy pairs, so the
    # same compact payload works for eager execution and graph replay.
    flat_source_block_ids: list[int] = []
    flat_destination_block_ids: list[int] = []
    for source_row, destination_row in zip(
            dump_block_id_rows, dump_destination_block_id_rows):
        if len(source_row) != len(destination_row):
            raise RuntimeError(
                "DSA full-block dump source/destination counts differ")
        for source_id, destination_id in zip(source_row, destination_row):
            destination_id = int(destination_id)
            if destination_id < KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID:
                raise RuntimeError(
                    "DSA full-block dump destination must be -1 or a valid "
                    f"block id, got {destination_id}")
            if destination_id == KV_CACHE_FULL_BLOCK_DUMP_NOOP_DST_BLOCK_ID:
                continue
            flat_source_block_ids.append(int(source_id))
            flat_destination_block_ids.append(destination_id)

    if (len(set(flat_destination_block_ids))
            != len(flat_destination_block_ids)):
        raise RuntimeError(
            "DSA full-block dump has duplicate physical destinations")

    if flat_source_block_ids:
        source_ids_np = np.asarray(flat_source_block_ids, dtype=np.int32)
        destination_ids_np = np.asarray(
            flat_destination_block_ids, dtype=np.int32)
        full_block_dump_batch = DSAFullBlockDumpBatch(
            src_hbm_block_ids_tensor=_tensor_from_numpy(
                source_ids_np, dtype=torch.int32, device=device),
            dst_dram_block_ids_tensor=_tensor_from_numpy(
                destination_ids_np, dtype=torch.int32, device=device),
        )
    else:
        full_block_dump_batch = DSAFullBlockDumpBatch.empty(
            tensor_device=device)

    return DSAForwardLayerHookPlan(
        full_block_dump_batch=full_block_dump_batch,
    )
