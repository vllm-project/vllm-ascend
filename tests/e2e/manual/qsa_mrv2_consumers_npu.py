# SPDX-License-Identifier: Apache-2.0
"""Exact-tree MRV2 allocation plus real QSA stores and historical compression."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm_ascend.models.qwen4_exp import qsa
from vllm_ascend.models.qwen4_exp.cache_config import get_qwen4_exp_kv_cache_config
from vllm_ascend.worker.v2 import attn_utils

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--fixture-file")
args = parser.parse_args()
torch.npu.set_device(0)
torch.manual_seed(21945)
fixture_path = Path(args.fixture_file) if args.fixture_file else Path(__file__).with_name("qsa_mrv2_fixture.py")
spec = importlib.util.spec_from_file_location("layout_fixture", fixture_path)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
results = {
    "source": attn_utils.__file__,
    "consumer": qsa.__file__,
    "qsa_sha256": hashlib.sha256(Path(qsa.__file__).read_bytes()).hexdigest(),
    "ascend_head": subprocess.check_output(
        ["git", "-C", str(Path(qsa.__file__).resolve().parents[3]), "rev-parse", "HEAD"], text=True
    ).strip(),
    "cases": [],
    "threshold": {"atol": 0.04, "rtol": 0.02},
}


def publish():
    p = Path(args.output)
    p.with_suffix(".tmp").write_text(json.dumps(results, indent=2))
    p.with_suffix(".tmp").replace(p)


def setup():
    groups = fixture._groups()
    raw_group = groups[1]
    narrow = raw_group.layer_names[1]
    wide = [name for name in raw_group.layer_names if name != narrow]
    groups[1:2] = [
        KVCacheGroupSpec([narrow], replace(raw_group.kv_cache_spec, head_size=128)),
        KVCacheGroupSpec(wide, replace(raw_group.kv_cache_spec, head_size=140)),
    ]
    for i, g in enumerate(groups):
        if g.layer_names[0].endswith("compressed_key_cache"):
            groups[i] = KVCacheGroupSpec(g.layer_names, replace(g.kv_cache_spec, head_size=128))
    config = get_qwen4_exp_kv_cache_config(
        NS(cache_config=NS(num_gpu_blocks_override=3, prefix_cache_retention_interval=None)), groups, 10**9
    )
    monkey = pytest.MonkeyPatch()
    # Reuse repository fixture runtime config only; planner, allocator,
    # reshape and all device consumers are their real implementations.
    raw_cpu, _, layout = fixture._allocate_mrv2_kv_cache(config, monkey)
    copies = {}
    raw = {}
    for name, tensor in raw_cpu.items():
        key = (tensor.data_ptr(), tensor.numel())
        if key not in copies:
            copies[key] = tensor.to("npu")
        raw[name] = copies[key]
    backend = NS(full_cls_name=lambda: "qwen-cache-test")
    attention = [
        NS(kv_cache_group_id=i, kv_cache_spec=g.kv_cache_spec, layer_names=g.layer_names, backend=backend)
        for i, g in enumerate(config.kv_cache_groups)
    ]
    caches = attn_utils._reshape_kv_cache_v2(attention, raw, "auto", [128] * len(groups), {}, config)
    view = caches[narrow]
    region = layout.region("r4")
    assert tuple(view.shape) == (3, 8, 1, 128)
    assert region.block_stride == region.page_size_bytes == 2240
    assert view.stride(0) == 1120 and not view.is_contiguous()
    assert view[1].data_ptr() - view[0].data_ptr() == 2240
    backing = raw[narrow]
    start = layout.owner(narrow).slot * layout.slot_backing_size + region.offset
    return monkey, raw, caches, view, backing, start, narrow, layout


def row_values(request, positions):
    # Independent logical token history; never derive expected values from a
    # physical cache view. Each request/position/channel has a distinct value.
    return ((request * 100 + positions[:, None] * 2 + torch.arange(128)[None, :] / 128) / 32).bfloat16().unsqueeze(1)


def norm_rope_reference(rows, norm, rot, first):
    x = rows.float()
    w = norm.weight.detach().float().cpu()
    x = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + w)).bfloat16().float()
    # Every axis is the same logical first position in this raw-key-only case.
    cos, sin = rot.cos_sin_cache.detach().float().cpu()[first].chunk(2, -1)
    a, b = x[..., :32], x[..., 32:64]
    cos = cos[:, None]
    sin = sin[:, None]
    return torch.cat([a * cos - b * sin, b * cos + a * sin, x[..., 64:]], -1).bfloat16()


def execute(negative=False):
    monkey, raw, caches, good, backing, start, name, layout = setup()
    try:
        cache = torch.as_strided(good, good.shape, (1024, 128, 128, 1)) if negative else good
        backing.fill_(0x42)
        expected = backing.cpu().clone()
        slots = torch.tensor([7, 8, 9, -1, 24, 2**40], device="npu", dtype=torch.int64)
        values = (torch.arange(6 * 128).reshape(6, 1, 128).float() / 64).bfloat16()
        qsa._store_qsa_cache_rows(cache, slots, values.to("npu"))
        torch.npu.synchronize()
        for index, slot in enumerate([7, 8, 9]):
            block, row = divmod(slot, 8)
            offset = start + block * 2240 + row * 256
            expected[offset : offset + 256] = values[index].contiguous().view(torch.int8).reshape(-1)
        # Compare the whole shared backing, including adjacent owners and raw
        # page padding, against byte offsets fixed by the physical contract.
        actual_backing = backing.cpu()
        if negative:
            mismatched_bytes = int(torch.count_nonzero(actual_backing != expected))
            assert mismatched_bytes > 0, "negative control failed to detect compact stride"
            return {"detected_mismatched_bytes": mismatched_bytes, "boundary_slots": [7, 8, 9]}
        torch.testing.assert_close(actual_backing, expected, rtol=0, atol=0)
        backing.fill_(0x42)
        output_name = name.replace("raw_key_cache", "compressed_key_cache")
        compressed = caches[output_name]
        obj = qsa.AscendQSAIndexer.__new__(qsa.AscendQSAIndexer)
        torch.nn.Module.__init__(obj)
        obj.index_head_dim = 128
        obj.compress_ratio = 4
        obj.raw_key_cache = NS(key_cache=cache, rope_position_cache=None)
        obj.compressed_key_cache = NS(kv_cache=compressed)
        obj.rotary_emb = get_rope(
            head_size=256,
            max_position=128,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
            },
        ).to(device="npu", dtype=torch.bfloat16)
        obj.k_layernorm = GemmaRMSNorm(128, eps=1e-6).to(device="npu", dtype=torch.bfloat16)
        obj.k_layernorm.weight.data.copy_(torch.linspace(-0.2, 0.4, 128, device="npu", dtype=torch.bfloat16))
        table = torch.tensor([[2], [1]], device="npu", dtype=torch.int32)
        compression = qsa._compress_qsa_groups
        recorded = []

        def observe(*a, **kw):
            pooled, first = compression(*a, **kw)
            recorded.append((pooled, first))
            return pooled, first

        qsa._compress_qsa_groups = observe
        try:
            rounds = []
            for positions in [torch.arange(0, 3), torch.arange(3, 10), torch.arange(10, 12)]:
                count = len(positions)
                logical = positions.repeat(2)
                values = torch.cat([row_values(req, positions) for req in range(2)])
                raw_slots = torch.cat([physical * 8 + positions.remainder(8) for physical in [2, 1]])
                destination = torch.cat(
                    [torch.where((positions + 1) % 4 == 0, physical * 64 + positions // 4, -1) for physical in [2, 1]]
                )
                metadata = NS(
                    num_actual_tokens=2 * count,
                    logical_positions=logical.to("npu"),
                    token_to_req=torch.arange(2, dtype=torch.int32).repeat_interleave(count).to("npu"),
                    query_start_loc=torch.tensor([0, count, 2 * count], device="npu", dtype=torch.int32),
                    block_table=table,
                    slot_mapping=raw_slots.to("npu"),
                )
                compressed_metadata = NS(slot_mapping=destination.to("npu"))
                before = backing.cpu().clone()
                obj._update_and_compress(values.to("npu"), logical.to("npu"), metadata, compressed_metadata)
                torch.npu.synchronize()
                pooled, first = recorded[-1]
                pooled = pooled.cpu()
                first = first.cpu()
                max_error = 0.0
                for request, physical in enumerate([2, 1]):
                    for local, position in enumerate(positions.tolist()):
                        index = request * count + local
                        offset = start + physical * 2240 + (position % 8) * 256
                        before[offset : offset + 256] = values[index].contiguous().view(torch.int8).reshape(-1)
                        if (position + 1) % 4:
                            continue
                        independent = (
                            row_values(request, torch.arange(position - 3, position + 1))
                            .float()
                            .mean(0, keepdim=True)
                            .bfloat16()
                        )
                        torch.testing.assert_close(pooled[index : index + 1], independent, atol=0.04, rtol=0.02)
                        assert torch.equal(first[index].reshape(-1), torch.full((3,), position - 3, dtype=first.dtype))
                        reference = norm_rope_reference(
                            independent, obj.k_layernorm, obj.rotary_emb, torch.tensor([position - 3])
                        )
                        actual = compressed[physical * 2, (position // 4)]
                        torch.testing.assert_close(actual.cpu(), reference[0], atol=0.04, rtol=0.02)
                        max_error = max(max_error, (actual.float().cpu() - reference[0].float()).abs().max().item())
                # The compressed owner has legitimate writes. Compare only all
                # raw physical pages in every owner slot, preserving padding.
                for owner_name in [n for n in caches if n.endswith("raw_key_cache")]:
                    owner = layout.owner(owner_name)
                    base = owner.slot * layout.slot_backing_size + layout.region("r4").offset
                    size = 3 * 2240
                    torch.testing.assert_close(
                        raw[owner_name].cpu()[base : base + size], before[base : base + size], rtol=0, atol=0
                    )
                rounds.append(
                    {
                        "positions": positions.tolist(),
                        "historical_rows_read": int(positions[0] > 0),
                        "physical_blocks": [2, 1],
                        "compressed_rows": int((destination >= 0).sum()),
                        "max_abs": max_error,
                    }
                )
            return {
                "shape": list(good.shape),
                "stride": list(good.stride()),
                "physical_page_bytes": 2240,
                "logical_page_bytes": 2048,
                "direct_boundary_slots": [7, 8, 9],
                "rounds": rounds,
                "padding_and_adjacent_raw_owners": "byte-exact",
                "reference": "independent logical history, FP32 norm/RoPE",
            }
        finally:
            qsa._compress_qsa_groups = compression
    finally:
        monkey.undo()


with set_current_vllm_config(VllmConfig()), torch.inference_mode():
    for negative in [False, True]:
        case = {"name": "compact-stride-negative-control" if negative else "mrv2-history-compression"}
        try:
            case["result"] = execute(negative)
            case["status"] = "EXPECTED_FAILURE" if negative else "PASS"
        except Exception as error:
            case.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        results["cases"].append(case)
        publish()
        print(case["name"], case["status"], flush=True)
raise SystemExit(int(any(case["status"] == "FAIL" for case in results["cases"])))
