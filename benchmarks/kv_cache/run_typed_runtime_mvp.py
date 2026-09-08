#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate typed KV cache allocation, mapping, and NPU tensor binding.

The script consumes a profile captured from a real vLLM-Ascend model.  It uses
the experimental small pages, allocates exact-LCM backing storage, passes block
IDs through group-local block tables and slot mappings, then verifies the final
physical byte addresses through Attention and Mamba tensor views.

This is a runtime plumbing benchmark, not an end-to-end model throughput claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm_ascend.core.typed_kv_cache import (
    TypedKVCachePlan,
    TypedPageSpec,
    TypedSuperpagePool,
    make_group_byte_view,
)


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _dtype(name: str) -> torch.dtype:
    value = name.removeprefix("torch.")
    try:
        return getattr(torch, value)
    except AttributeError as exc:
        raise ValueError(f"unsupported profile dtype {name!r}") from exc


def _build_plan(
    profile: dict[str, object], num_superpages: int | None
) -> tuple[TypedKVCachePlan, list[dict[str, object]]]:
    groups = profile["groups"]
    assert isinstance(groups, list)
    specs = tuple(
        TypedPageSpec(
            group_id=int(group["group_id"]),
            page_size_bytes=int(group["experimental_small_page_size_bytes"]),
            block_size_tokens=int(group["experimental_block_size_tokens"]),
        )
        for group in groups
    )
    raw_tensors = profile["raw_tensors"]
    assert isinstance(raw_tensors, list) and raw_tensors
    per_tensor_budget = int(raw_tensors[0]["size_bytes"])
    probe = TypedKVCachePlan.exact_lcm(specs, per_tensor_budget)
    if num_superpages is None:
        return probe, groups
    if num_superpages < 2:
        raise ValueError("--num-superpages must be at least two")
    return (
        TypedKVCachePlan(
            specs=probe.specs,
            superpage_size_bytes=probe.superpage_size_bytes,
            num_superpages=num_superpages,
        ),
        groups,
    )


def _mamba_views(
    group_view: torch.Tensor,
    group: dict[str, object],
) -> list[torch.Tensor]:
    shapes = group["state_shapes"]
    dtypes = group["state_dtypes"]
    assert isinstance(shapes, list) and isinstance(dtypes, list)
    page_size = group_view.shape[1]
    raw = group_view.flatten()
    views: list[torch.Tensor] = []
    component_offset_bytes = 0
    for shape_value, dtype_name in zip(shapes, dtypes):
        assert isinstance(shape_value, list) and isinstance(dtype_name, str)
        dtype = _dtype(dtype_name)
        dtype_size = torch.empty((), dtype=dtype).element_size()
        shape = tuple(int(dim) for dim in shape_value)
        elements_per_state = 1
        for dim in shape:
            elements_per_state *= dim
        component_bytes = elements_per_state * dtype_size
        if component_offset_bytes + component_bytes > page_size:
            raise ValueError("Mamba state components exceed the typed page")
        views.append(
            torch.as_strided(
                raw.view(dtype),
                size=(group_view.shape[0], *shape),
                stride=(page_size // dtype_size, *torch.empty(shape).stride()),
                storage_offset=component_offset_bytes // dtype_size,
            )
        )
        component_offset_bytes += component_bytes
    return views


def _time_dense_writes(
    view: torch.Tensor,
    block_ids: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> tuple[float, float]:
    payload = (
        torch.arange(block_ids.numel() * view.shape[1], dtype=torch.int64, device=device)
        .remainder_(251)
        .to(torch.uint8)
    )
    payload = payload.view(block_ids.numel(), view.shape[1])
    samples_ms: list[float] = []
    for _ in range(iterations):
        _sync(device)
        start = time.perf_counter()
        view.index_copy_(0, block_ids, payload)
        _sync(device)
        samples_ms.append((time.perf_counter() - start) * 1000)
    median_ms = statistics.median(samples_ms)
    gib_per_second = payload.numel() / (1024**3) / (median_ms / 1000)
    return median_ms, gib_per_second


def _attention_kv_views(
    raw: torch.Tensor,
    plan: TypedKVCachePlan,
    group_id: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build PR #14340's per-block interleaved K/V views."""

    spec = plan.spec(group_id)
    dtype = torch.bfloat16
    elements_per_component = spec.block_size_tokens * num_kv_heads * head_size
    expected_page_bytes = 2 * elements_per_component * 2
    if spec.page_size_bytes != expected_page_bytes:
        raise ValueError(
            "profile Attention page does not match bf16 K/V dimensions: "
            f"{spec.page_size_bytes} != {expected_page_bytes}"
        )
    num_blocks = plan.total_managed_bytes // spec.page_size_bytes
    kv_cache = raw.view(dtype).view(
        2,
        num_blocks,
        spec.block_size_tokens,
        num_kv_heads,
        head_size,
    )
    kv_cache.as_strided_(
        size=kv_cache.shape,
        stride=(
            elements_per_component,
            2 * elements_per_component,
            num_kv_heads * head_size,
            head_size,
            1,
        ),
    )
    return kv_cache[0], kv_cache[1]


def _run_reshape_and_cache_kernel(
    raw: torch.Tensor,
    plan: TypedKVCachePlan,
    group_id: int,
    block_ids: list[int],
    iterations: int,
    device: torch.device,
    num_kv_heads: int = 2,
    head_size: int = 256,
) -> dict[str, float | str]:
    if device.type != "npu":
        return {"status": "skipped: reshape_and_cache is an NPU kernel"}

    import torch_npu

    key_cache, value_cache = _attention_kv_views(raw, plan, group_id, num_kv_heads, head_size)
    num_tokens = min(plan.spec(group_id).block_size_tokens, 64)
    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    slot_mapping = torch.arange(
        block_ids[0] * plan.spec(group_id).block_size_tokens,
        block_ids[0] * plan.spec(group_id).block_size_tokens + num_tokens,
        dtype=torch.int32,
        device=device,
    )

    # Warm up both compilation and workspace allocation before timing.
    torch_npu.npu_scatter_pa_kv_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_mapping=slot_mapping,
        cache_mode="Norm",
    )
    _sync(device)
    if not torch.equal(key_cache[block_ids[0], :num_tokens], key):
        raise AssertionError("reshape_and_cache wrote K to the wrong typed block")
    if not torch.equal(value_cache[block_ids[0], :num_tokens], value):
        raise AssertionError("reshape_and_cache wrote V to the wrong typed block")

    samples_ms: list[float] = []
    for _ in range(iterations):
        _sync(device)
        start = time.perf_counter()
        torch_npu.npu_scatter_pa_kv_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
            cache_mode="Norm",
        )
        _sync(device)
        samples_ms.append((time.perf_counter() - start) * 1000)
    median_ms = statistics.median(samples_ms)
    payload_bytes = (key.numel() + value.numel()) * key.element_size()
    return {
        "status": "passed",
        "median_ms": median_ms,
        "payload_gib_per_second": payload_bytes / (1024**3) / (median_ms / 1000),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    plan, groups = _build_plan(profile, args.num_superpages)
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

    device = torch.device(args.device)
    raw = torch.zeros(plan.total_managed_bytes, dtype=torch.uint8, device=device)
    pool = TypedSuperpagePool(plan)

    allocations: dict[int, list] = {}
    for group in groups:
        group_id = int(group["group_id"])
        count = args.attention_blocks if not group["contains_mamba"] else args.mamba_blocks
        allocations[group_id] = pool.for_group(group_id).get_new_blocks(count)

    address_checks: list[dict[str, object]] = []
    group_views: dict[int, torch.Tensor] = {}
    for group in groups:
        group_id = int(group["group_id"])
        view = make_group_byte_view(raw, plan, group_id)
        group_views[group_id] = view
        block_ids = [block.block_id for block in allocations[group_id]]
        for sequence_idx, block_id in enumerate(block_ids):
            typed_id = plan.from_group_local_id(group_id, block_id)
            marker = (17 + group_id * 31 + sequence_idx) % 251
            view[block_id, 0] = marker
            _sync(device)
            physical_offset = plan.byte_offset(typed_id)
            actual = int(raw[physical_offset].cpu())
            if actual != marker:
                raise AssertionError(
                    f"group {group_id} block {block_id} bound to wrong byte: expected {marker}, got {actual}"
                )
            address_checks.append(
                {
                    "group_id": group_id,
                    "group_local_block_id": block_id,
                    "superpage_id": typed_id.superpage_id,
                    "slot": typed_id.slot,
                    "physical_byte_offset": physical_offset,
                }
            )

        # Exercise the same block-table -> slot-mapping formula consumed by
        # Attention kernels. Mamba consumes block_ids[:, 0] as state indices.
        if not group["contains_mamba"]:
            positions = torch.arange(
                len(block_ids) * plan.spec(group_id).block_size_tokens,
                dtype=torch.int64,
                device=device,
            )
            table = torch.tensor(block_ids, dtype=torch.int64, device=device)
            logical_indices = positions // plan.spec(group_id).block_size_tokens
            offsets = positions % plan.spec(group_id).block_size_tokens
            slot_mapping = table[logical_indices] * plan.spec(group_id).block_size_tokens + offsets
            expected_last = plan.slot_mapping(
                group_id,
                block_ids[-1],
                plan.spec(group_id).block_size_tokens - 1,
            )
            if int(slot_mapping[-1].cpu()) != expected_last:
                raise AssertionError("Attention slot mapping differs from typed plan")
        else:
            state_views = _mamba_views(view, group)
            block_id = block_ids[0]
            state_views[0][block_id].flatten()[0] = 7
            state_views[1][block_id].flatten()[0] = 11
            _sync(device)
            if int(state_views[0][block_id].flatten()[0].cpu()) != 7:
                raise AssertionError("Mamba Conv state bind failed")
            if int(state_views[1][block_id].flatten()[0].cpu()) != 11:
                raise AssertionError("Mamba SSM state bind failed")

    attention_group = next(group for group in groups if not group["contains_mamba"])
    attention_group_id = int(attention_group["group_id"])
    attention_ids = torch.tensor(
        [block.block_id for block in allocations[attention_group_id]],
        dtype=torch.int64,
        device=device,
    )
    median_ms, gib_per_second = _time_dense_writes(
        group_views[attention_group_id],
        attention_ids,
        args.iterations,
        device,
    )
    reshape_and_cache = _run_reshape_and_cache_kernel(
        raw,
        plan,
        attention_group_id,
        [block.block_id for block in allocations[attention_group_id]],
        args.iterations,
        device,
    )
    uniform_first_kernel_block = int(attention_group.get("uniform_slots_per_page", 1))
    reshape_and_cache_uniform = _run_reshape_and_cache_kernel(
        raw,
        plan,
        attention_group_id,
        [uniform_first_kernel_block],
        args.iterations,
        device,
    )
    kernel_latency_ratio = None
    if reshape_and_cache.get("status") == "passed" and reshape_and_cache_uniform.get("status") == "passed":
        kernel_latency_ratio = float(reshape_and_cache["median_ms"]) / float(reshape_and_cache_uniform["median_ms"])

    return {
        "device": str(device),
        "model": profile.get("model"),
        "superpage_size_bytes": plan.superpage_size_bytes,
        "num_superpages": plan.num_superpages,
        "managed_bytes_per_raw_tensor": plan.total_managed_bytes,
        "group_capacities": {str(spec.group_id): plan.capacity(spec.group_id) for spec in plan.specs},
        "free_superpages_after_allocation": pool.get_num_free_superpages(),
        "address_checks": address_checks,
        "correctness": "passed",
        "dense_attention_write_median_ms": median_ms,
        "dense_attention_write_gib_per_second": gib_per_second,
        "reshape_and_cache_kernel_typed": reshape_and_cache,
        "reshape_and_cache_kernel_uniform_baseline": reshape_and_cache_uniform,
        "typed_over_uniform_kernel_latency_ratio": kernel_latency_ratio,
        "note": "Runtime mapping microbenchmark; not end-to-end model throughput.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--num-superpages", type=int, default=8)
    parser.add_argument("--attention-blocks", type=int, default=8)
    parser.add_argument("--mamba-blocks", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = args.profile.parent / "typed-runtime-mvp.json"
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    result = run(cli_args)
    cli_args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
