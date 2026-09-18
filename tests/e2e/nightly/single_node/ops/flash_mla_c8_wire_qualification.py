# SPDX-License-Identifier: Apache-2.0
"""Run only inside the isolated native + framework candidate environment.

python qualify_npu.py --device 0 --json qualification.json [--native]

No benchmark or production deployment is performed. The native option needs
flash_mla_c8_qualification.py on PYTHONPATH and the matching compiled binding.
"""

import argparse
import inspect
import json
import os
from pathlib import Path

_CUSTOM_OPP_PATH = os.environ.get("ASCEND_CUSTOM_OPP_PATH")

import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.ops.triton import sfa_dcp_exchange as exchange
from vllm_ascend.ops.triton.sfa_dcp_merge import merge_raw_dcp_output_lse

if _CUSTOM_OPP_PATH is not None:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _CUSTOM_OPP_PATH


def bit_equal(a, b):
    return torch.equal(a.detach().cpu().contiguous().view(torch.uint8), b.detach().cpu().contiguous().view(torch.uint8))


def pattern_wire(tokens, seed):
    rows = 96 * tokens
    words = (torch.arange(rows * 256, dtype=torch.int64) * 65537 + seed) % (1 << 32)
    wire = torch.zeros((8, 12, tokens, 272), dtype=torch.int32)
    wire[..., :256] = words.to(torch.int32).reshape(8, 12, tokens, 256)
    special = torch.tensor([0, -(1 << 31), 0x7F800000, -8388608, 0x7FC01234, 0x3F800000], dtype=torch.int32)
    wire[..., 256] = special.roll(seed % 6).repeat((rows + 5) // 6)[:rows].reshape(8, 12, tokens)
    return wire


def logical_views(wire):
    tokens = wire.shape[2]
    output = wire.view(torch.bfloat16).reshape(96, tokens, 544)[..., :512].transpose(0, 1)
    lse = wire[..., 256].reshape(96, tokens).view(torch.float32).transpose(0, 1).unsqueeze(-1).clone()
    return output, lse


def qualify_pack(tokens):
    wire_cpu = pattern_wire(tokens, 0x7FC1)
    wire = wire_cpu.npu()
    output, lse = logical_views(wire)
    plain = output.contiguous()
    direct = exchange.pack_raw_dcp_output_lse(output, lse, 272)
    assert direct.data_ptr() == wire.data_ptr()
    # This is a dispatch assertion, not a CPU substitute for the fallback kernel.
    original_kernel = exchange._pack
    try:
        class NoLaunch:
            def __getitem__(self, grid):
                raise AssertionError("NTD_DCP fast path launched pack")
        exchange._pack = NoLaunch()
        assert bit_equal(exchange.pack_raw_dcp_output_lse(output, lse, 272), wire)
    finally:
        exchange._pack = original_kernel

    def run():
        return (
            exchange.pack_raw_dcp_output_lse(output, lse, 272),
            exchange.pack_raw_dcp_output_lse(plain, lse, 272),
            exchange.pack_raw_dcp_output_lse(plain, lse),
        )

    for _ in range(3):
        actual = run()
    torch.npu.synchronize()
    assert bit_equal(actual[0], wire_cpu) and bit_equal(actual[1], wire_cpu)
    assert bit_equal(actual[2], wire_cpu[..., :257])
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for seed in (123, 0xFFC1, 331):
        changed = pattern_wire(tokens, seed)
        changed_o, changed_lse = logical_views(changed)
        wire.copy_(changed)
        plain.copy_(changed_o)
        lse.copy_(changed_lse)
        graph.replay()
        torch.npu.synchronize()
        assert bit_equal(captured[0], changed) and bit_equal(captured[1], changed)
        assert bit_equal(captured[2], changed[..., :257])
    return {"tokens": tokens, "direct_view": "bitwise/no pack", "fallback257_272": "bitwise", "changed_input_graph": 3}


def merge_inputs(tokens, seed):
    generator = torch.Generator().manual_seed(seed)
    values = (torch.randn(8, 12, tokens, 512, generator=generator) * 0.1).bfloat16()
    stats = torch.randn(8, 12, tokens, generator=generator)
    local = (torch.randn(tokens, 12, 512, generator=generator) * 0.1).bfloat16()
    local_stats = torch.randn(tokens, 12, 1, generator=generator)
    stats[:, 0, 0] = -float("inf")
    values[:, 0, 0] = float("nan")
    local_stats[0, 0] = -float("inf")
    local[0, 0] = float("nan")
    stats[2, 2, 1] = float("nan")
    stats[3, 2, 1] = float("inf")
    values[2:4, 2, 1] = float("nan")
    raw = torch.zeros(8, 12, tokens, 272, dtype=torch.int32)
    raw[..., :256] = values.view(torch.int32)
    raw[..., 256] = stats.view(torch.int32)
    # The consumer must ignore padding even if a producer-test poisons it.
    raw[..., 257:] = -123456789
    finite, local_finite = torch.isfinite(stats), torch.isfinite(local_stats.squeeze(-1).transpose(0, 1))
    local_stat_ht = local_stats.squeeze(-1).transpose(0, 1)
    maximum = torch.maximum(stats.masked_fill(~finite, -float("inf")).amax(0), local_stat_ht.masked_fill(~local_finite, -float("inf")))
    maximum = maximum.masked_fill(~torch.isfinite(maximum), 0)
    weights = torch.where(finite, torch.exp(stats - maximum), 0)
    local_weights = torch.where(local_finite, torch.exp(local_stat_ht - maximum), 0)
    numerator = (values.float().masked_fill(~finite[..., None], 0) * weights[..., None]).sum(0)
    numerator += local.transpose(0, 1).float().masked_fill(~local_finite[..., None], 0) * local_weights[..., None]
    denominator = weights.sum(0) + local_weights
    expected = (numerator / denominator.masked_fill(denominator == 0, 1)[..., None]).transpose(0, 1).bfloat16()
    return raw, local, local_stats, expected


def qualify_merge(tokens):
    raw_cpu, local_cpu, stats_cpu, expected = merge_inputs(tokens, 1)
    raw, legacy = raw_cpu.npu(), raw_cpu[..., :257].contiguous().npu()
    local, stats = local_cpu.npu(), stats_cpu.npu()

    def run():
        return (
            merge_raw_dcp_output_lse(raw, 512, 1, local, stats),
            merge_raw_dcp_output_lse(legacy, 512, 1, local, stats),
        )

    for _ in range(3):
        actual = run()
    torch.npu.synchronize()
    assert bit_equal(*actual)
    torch.testing.assert_close(actual[0].cpu(), expected, atol=0.002, rtol=0.01)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for seed in (2, 3, 4):
        changed, local_cpu, stats_cpu, expected = merge_inputs(tokens, seed)
        raw.copy_(changed)
        legacy.copy_(changed[..., :257])
        local.copy_(local_cpu)
        stats.copy_(stats_cpu)
        graph.replay()
        torch.npu.synchronize()
        assert bit_equal(*captured)
        torch.testing.assert_close(captured[0].cpu(), expected, atol=0.002, rtol=0.01)
        assert torch.equal(captured[0][0, 0].cpu(), torch.zeros(512, dtype=torch.bfloat16))
    return {"tokens": tokens, "pitch257_vs272": "bitwise", "invalid_and_current": "reference pass", "changed_input_graph": 3}


def qualify_native(checkpoint=None):
    from flash_mla_c8_qualification import C8Case

    reports = []
    fd_seen = False
    for lengths in ([16384] * 16, [0, 1, 127, 128, 129, 1023, 2049, 16384] * 2, [32768]):
        case = C8Case(lengths, 96, qlen=4)
        metadata = case.metadata()
        # Test-only read proves the FD path is exercised; never used for dispatch.
        fd = int(metadata[1].cpu()) != 0
        fd_seen |= fd

        def run():
            normal, normal_lse = case.run(metadata, layout="TND")
            direct, direct_lse = case.run(metadata, layout="NTD_DCP")
            wire = exchange.pack_raw_dcp_output_lse(direct, direct_lse.transpose(0, 1).unsqueeze(-1), 272)
            return normal, normal_lse, direct, direct_lse, wire

        def check(results):
            normal, normal_lse, direct, direct_lse, wire = results
            assert direct.stride() == (544, direct.shape[0] * 544, 1)
            assert wire.data_ptr() == direct.data_ptr()
            assert bit_equal(normal, direct)
            assert bit_equal(normal_lse, direct_lse)
            embedded = wire[..., 256].reshape(96, direct.shape[0])
            assert bit_equal(embedded, direct_lse.contiguous().view(torch.int32))
            assert torch.count_nonzero(wire[..., 257:].cpu()) == 0

        for _ in range(3):
            eager = run()
        check(eager)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = run()
        for i in range(3):
            case.q.copy_(case.q_cpu.roll(i + 1, -1))
            case.qr.copy_(case.qr_cpu.roll(i + 1, -1))
            graph.replay()
            torch.npu.synchronize()
            check(captured)
        reports.append({"tokens": case.q.shape[0], "lengths": lengths, "fd": fd, "output_lse_wire_padding": "bitwise/zero", "changed_input_graph": 3})
        if checkpoint is not None:
            checkpoint(reports)
    assert fd_seen, "No native FlashDecode case ran; choose a schedule with metadata[1] != 0."
    return reports


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    assert "raw_row_words" in inspect.signature(exchange.pack_raw_dcp_output_lse).parameters, "Framework candidate is not installed."
    torch.npu.set_device(args.device)
    torch.set_num_threads(4)
    result = {"status": "running", "pack": [], "merge": [], "native": []}

    def write_result():
        args.json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json.with_suffix(args.json.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.json)

    def native_checkpoint(reports):
        result["native"] = reports
        write_result()

    write_result()
    try:
        if not args.native_only:
            for tokens in (4, 8, 16, 32, 64):
                result["pack"].append(qualify_pack(tokens))
                write_result()
                result["merge"].append(qualify_merge(tokens))
                write_result()
        if args.native or args.native_only:
            result["native"] = qualify_native(native_checkpoint)
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        write_result()
        raise
    write_result()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
