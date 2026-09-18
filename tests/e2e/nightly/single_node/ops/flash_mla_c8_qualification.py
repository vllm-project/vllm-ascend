#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C8 FlashMLA operator gate; no model/server or hidden cache dequantization.

Run after building/installing the C8 extension:
  python flash_mla_c8_qualification.py --json result.json [--benchmark]

Correctness compares identical quantized inputs, independently gathering logical
pages on CPU. Benchmarks cover the current C8-per-DP/DCP1 and C16-per-DP/DCP8
workloads with three speculative tokens and 99% shared prefix pages. Kernel-only
latency is labelled explicitly; it is not an attention-module or ITL result.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from flash_mla_qualification import cpu_reference


def error(actual, expected):
    if actual.numel() == 0:
        return {"max_abs": 0.0, "normalized_rmse": 0.0}
    delta = actual.float() - expected.float()
    return {
        "max_abs": delta.abs().max().item(),
        "normalized_rmse": (delta.square().mean() / expected.float().square().mean().clamp_min(1e-12)).sqrt().item(),
    }


class C8Case:
    def __init__(self, lengths, heads, *, qlen=4, gap=2, scale_k=0.015625, seed=16468, shared=False):
        gen = torch.Generator().manual_seed(seed)
        self.lengths_cpu = lengths
        self.max_length = max(lengths)
        self.heads, self.qlen, self.batch = heads, qlen, len(lengths)
        self.scale = 1 / math.sqrt(576)
        total = qlen * self.batch
        page_counts = [max(1, math.ceil(n / 128)) for n in lengths]
        common = int(min(lengths) * 0.99) // 128 if shared else 0
        pages = common + sum(n - common for n in page_counts)
        blocks = torch.zeros((self.batch, max(page_counts)), dtype=torch.int32)
        cursor = common
        for b, count in enumerate(page_counts):
            blocks[b, :common] = torch.arange(common)
            blocks[b, common:count] = torch.arange(cursor, cursor + count - common)
            cursor += count - common
        # Shuffle physical pages to ensure the golden does not assume identity.
        permutation = torch.randperm(pages, generator=gen)
        self.blocks_cpu = permutation[blocks.long()].to(torch.int32)
        latent = (torch.randn((pages, 128, 1, 512), generator=gen) * 0.5 / scale_k).clamp(-448, 448)
        self.k_cpu = latent.to(torch.float8_e4m3fn)
        self.kr_cpu = (torch.randn((pages, 128, 1, 64), generator=gen) * 0.5).bfloat16()
        # Different byte/element strides are intentional.
        self.k_backing = torch.zeros(pages * gap * 128 * 512 + 128, dtype=torch.uint8, device="npu")
        self.kr_backing = torch.full((pages * (gap + 1) * 128 * 64 + 64,), -13, dtype=torch.bfloat16, device="npu")
        self.k = self.k_backing.view(torch.float8_e4m3fn).as_strided(
            self.k_cpu.shape, (gap * 128 * 512, 512, 512, 1), 64
        )
        self.kr = self.kr_backing.as_strided(self.kr_cpu.shape, ((gap + 1) * 128 * 64, 64, 64, 1), 32)
        self.k.copy_(self.k_cpu)
        self.kr.copy_(self.kr_cpu)
        self.sq_cpu = torch.linspace(0.005, 0.035, total * heads).reshape(total, heads)
        self.sk = torch.tensor([scale_k], dtype=torch.float32, device="npu")
        self.q_cpu = (
            (torch.randn((total, heads, 512), generator=gen) * 0.5 / self.sq_cpu[..., None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        raw_qr = torch.randn((total, heads, 64), generator=gen) * 0.5
        self.qr_cpu = (raw_qr / self.sq_cpu[..., None] / scale_k).bfloat16()
        self.q, self.qr, self.sq = (x.to("npu") for x in (self.q_cpu, self.qr_cpu, self.sq_cpu))
        self.cu_cpu = list(range(0, total + 1, qlen))
        self.cu = torch.tensor(self.cu_cpu, dtype=torch.int32, device="npu")
        self.lengths = torch.tensor(lengths, dtype=torch.int32, device="npu")
        self.used_cpu = [qlen] * self.batch
        self.used = torch.full((self.batch,), qlen, dtype=torch.int32, device="npu")
        self.blocks = self.blocks_cpu.to("npu")
        self.scale_k = scale_k
        self.mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8), diagonal=1).to("npu")

    def metadata(self, causal=False, *, is_c8=True):
        return torch.ops._C_ascend.flash_mla_with_kvcache_metadata(
            self.lengths,
            self.heads,
            1,
            cu_seqlens_q=self.cu,
            seqused_q=self.used,
            max_seqlen_q=self.qlen,
            max_seqlen_kv=self.max_length,
            head_dim_qk=576,
            head_dim_v=512,
            mask_mode=3 if causal else 0,
            layout_q="TND",
            is_c8=is_c8,
        )

    def run(self, metadata, *, causal=False, layout="TND"):
        return torch.ops._C_ascend.flash_mla_with_kvcache(
            self.q,
            self.k,
            block_table=self.blocks,
            cache_seqlens=self.lengths,
            cu_seqlens_q=self.cu,
            seqused_q=self.used,
            attn_mask=self.mask if causal else None,
            metadata=metadata,
            head_dim_v=512,
            softmax_scale=self.scale,
            mask_mode=3 if causal else 0,
            max_seqlen_q=self.qlen,
            max_seqlen_kv=self.max_length,
            layout_q="TND",
            layout_kv="PA_BBND",
            layout_out=layout,
            return_softmax_lse=True,
            query_rope=self.qr,
            key_rope=self.kr,
            dequant_scale_query=self.sq,
            dequant_scale_key=self.sk,
        )

    def run_fia(self, causal=False):
        import torch_npu

        # FIA PA uses [pages, heads, block, dim]; permute is a view and keeps
        # the same independently strided physical allocations as FlashMLA.
        key = self.k.permute(0, 2, 1, 3)
        key_rope = self.kr.permute(0, 2, 1, 3)
        return torch_npu.npu_fused_infer_attention_score_v2(
            self.q,
            key,
            key,
            query_rope=self.qr,
            key_rope=key_rope,
            num_query_heads=self.heads,
            num_key_value_heads=1,
            input_layout="TND",
            actual_seq_qlen=self.cu_cpu[1:],
            actual_seq_kvlen=self.lengths_cpu,
            block_table=self.blocks,
            block_size=128,
            softmax_scale=self.scale,
            atten_mask=self.mask if causal else None,
            sparse_mode=3 if causal else 0,
            query_quant_mode=3,
            key_quant_mode=0,
            value_quant_mode=0,
            dequant_scale_query=self.sq,
            dequant_scale_key=self.sk,
            dequant_scale_value=self.sk,
            return_softmax_lse=True,
        )

    def prepare_bf16(self):
        # Same physical pages and quantized values as C8, widened before timing.
        # This is an operator baseline, not a cache-conversion implementation.
        self.bf16_q = (
            torch.cat(
                (
                    self.q_cpu.float() * self.sq_cpu[..., None],
                    self.qr_cpu.float() * self.sq_cpu[..., None] * self.scale_k,
                ),
                -1,
            )
            .bfloat16()
            .to("npu")
        )
        values = torch.cat((self.k_cpu.float() * self.scale_k, self.kr_cpu.float()), -1).bfloat16()
        pages = values.shape[0]
        gap = self.k.stride(0) // (128 * 512)
        self.bf16_backing = torch.empty(pages * gap * 128 * 576 + 128, dtype=torch.bfloat16, device="npu")
        self.bf16_k = self.bf16_backing.as_strided(values.shape, (gap * 128 * 576, 576, 576, 1), 64)
        self.bf16_k.copy_(values)
        self.bf16_metadata = self.metadata(is_c8=False)

    def run_bf16(self):
        return torch.ops._C_ascend.flash_mla_with_kvcache(
            self.bf16_q,
            self.bf16_k,
            block_table=self.blocks,
            cache_seqlens=self.lengths,
            cu_seqlens_q=self.cu,
            seqused_q=self.used,
            metadata=self.bf16_metadata,
            head_dim_v=512,
            softmax_scale=self.scale,
            max_seqlen_q=self.qlen,
            max_seqlen_kv=self.max_length,
            layout_q="TND",
            layout_kv="PA_BBND",
            layout_out="TND",
            return_softmax_lse=True,
        )

    def reference(self, causal=False):
        q = torch.cat(
            (self.q_cpu.float() * self.sq_cpu[..., None], self.qr_cpu.float() * self.sq_cpu[..., None] * self.scale_k),
            dim=-1,
        )
        kv = torch.cat((self.k_cpu.float() * self.scale_k, self.kr_cpu.float()), dim=-1).squeeze(2)
        out, lse, live = cpu_reference(
            q, kv, self.blocks_cpu, self.cu_cpu, self.used_cpu, self.lengths_cpu, causal, self.scale
        )
        lse[:, ~live] = -torch.inf
        return out, lse, live

    def check(self, *, causal=False, layout="TND"):
        out, lse = self.run(self.metadata(causal), causal=causal, layout=layout)
        return self.check_result(out, lse, causal=causal, layout=layout)

    def check_result(self, out, lse, *, causal=False, layout="TND"):
        out, lse = out.cpu(), lse.cpu()
        if layout == "NTD":
            out = out.transpose(0, 1)
        ref, ref_lse, live = self.reference(causal)
        assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out[~live].float(), ref[~live], rtol=0, atol=0)
        assert torch.isneginf(lse[:, ~live]).all()
        # FP8 P rounding differs from the FP32 reference. Report its error
        # explicitly, while checking LSE independently of the P/V path.
        torch.testing.assert_close(out[live].float(), ref[live], rtol=0.05, atol=0.03)
        torch.testing.assert_close(lse[:, live], ref_lse[:, live], rtol=0.0002, atol=0.002)
        return {
            "lengths": self.lengths_cpu,
            "used_q": self.used_cpu,
            "heads": self.heads,
            "causal": causal,
            "layout": layout,
            "output_error": error(out[live], ref[live]),
            "lse_error": error(lse[:, live], ref_lse[:, live]),
        }


def graph_replay_check(heads):
    case = C8Case([129, 1025], heads, seed=38)
    metadata = case.metadata()
    case.run(metadata)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out, lse = case.run(metadata)
    results = []
    for lengths in ([0, 127], [0, 0], [128, 1024], [129, 1025]):
        case.lengths_cpu = list(lengths)
        case.lengths.copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
        # AICPU schedule remains outside the graph, as in the model runner.
        metadata.copy_(case.metadata())
        case.q_cpu = (-case.q_cpu.float()).to(torch.float8_e4m3fn)
        case.q.copy_(case.q_cpu)
        graph.replay()
        torch.npu.synchronize()
        results.append(case.check_result(out, lse))
    return {"gate": "graph_replay_dynamic_lengths_schedule_and_query", "heads": heads, "replays": results}


def bench_graph(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(10):
            fn()
    for _ in range(5):
        graph.replay()
    trials = []
    for _ in range(7):
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        trials.append(begin.elapsed_time(end) * 10)  # milliseconds / 100 calls -> microseconds
    return {"median_us": statistics.median(trials), "trials_us": trials}


def dcp_merge_check():
    # One short request gives several empty shards and verifies their neutral
    # contribution; the other crosses all ranks and a partial physical page.
    full = C8Case([127, 2179], 96, seed=72)
    reference, reference_lse, _ = full.reference()
    outs, lses = [], []
    for rank in range(8):
        local = []
        local_kr = []
        lens = []
        for b, length in enumerate(full.lengths_cpu):
            keys = full.k_cpu[full.blocks_cpu[b].long()].reshape(-1, 1, 512)[:length]
            rope = full.kr_cpu[full.blocks_cpu[b].long()].reshape(-1, 1, 64)[:length]
            select = (torch.arange(length) // 128) % 8 == rank
            local.append(keys[select])
            local_kr.append(rope[select])
            lens.append(int(select.sum()))
        case = C8Case(lens, 96, seed=72)
        case.q.copy_(full.q)
        case.qr.copy_(full.qr)
        case.sq.copy_(full.sq)
        for b, length in enumerate(lens):
            for i in range(math.ceil(length / 128)):
                count = min(128, length - i * 128)
                page = int(case.blocks_cpu[b, i])
                case.k[page, :count].copy_(local[b][i * 128 : i * 128 + count])
                case.kr[page, :count].copy_(local_kr[b][i * 128 : i * 128 + count])
        out, lse = case.run(case.metadata())
        outs.append(out.cpu().float())
        lses.append(lse.cpu().T)
    lses = torch.stack(lses)
    maximum = lses.max(0).values
    weights = (lses - maximum).exp()
    summed = weights.sum(0)
    combined = (torch.stack(outs) * weights[..., None]).sum(0) / summed[..., None]
    combined_lse = maximum + summed.log()
    torch.testing.assert_close(combined, reference, rtol=0.05, atol=0.03)
    torch.testing.assert_close(combined_lse.T, reference_lse, rtol=0.0002, atol=0.002)
    return {
        "gate": "dcp8_local_attention_and_lse_merge",
        "output_error": error(combined, reference),
        "lse_error": error(combined_lse.T, reference_lse),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--baselines", action="store_true", help="Compare installed FIA C8 and BF16 FlashMLA")
    args = parser.parse_args()
    import torch_npu  # noqa: F401
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.npu.set_device(args.device)
    torch.set_num_threads(8)
    results = []

    def record(result):
        results.append(result)
        args.json.write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)

    for heads in (12, 96):
        case = C8Case([127, 128, 129, 1025, 0], heads)
        for causal, layout in ((False, "TND"), (True, "NTD")):
            record(case.check(causal=causal, layout=layout))
        ragged = C8Case([127, 128, 129, 1025], heads)
        ragged.used_cpu = [1, 3, 4, 0]
        ragged.used.copy_(torch.tensor(ragged.used_cpu, dtype=torch.int32, device="npu"))
        for causal in (False, True):
            record({"gate": "speculative_used_q_and_padding", **ragged.check(causal=causal)})
        record(graph_replay_check(heads))
        if args.baselines:
            # Avoid asserting a convention for the installed FIA's empty rows.
            fia_case = C8Case([127, 128, 129, 1025], heads)
            for causal in (False, True):
                output, lse = fia_case.run_fia(causal)
                # FIA exposes token-major LSE, FlashMLA uses head-major LSE.
                if lse.shape == (fia_case.batch * fia_case.qlen, heads, 1):
                    lse = lse.squeeze(-1).T
                elif lse.shape == (fia_case.batch * fia_case.qlen, heads):
                    lse = lse.T
                record({"gate": "installed_fia_c8_reference", **fia_case.check_result(output, lse, causal=causal)})
    record(dcp_merge_check())
    if args.benchmark:
        for batch, heads, length, dcp in ((8, 12, 131072, 1), (16, 96, 16384, 8)):
            case = C8Case([length] * batch, heads, shared=True)
            meta = case.metadata()
            record(
                {
                    "gate": "kernel_only_graph_latency",
                    "dcp": dcp,
                    "batch": batch,
                    "heads": heads,
                    "tokens": batch * 4,
                    "local_kv_length": length,
                    "prefix_fraction": 0.99,
                    "implementation": "flash_mla_c8",
                    **bench_graph(lambda case=case, meta=meta: case.run(meta)),
                }
            )
            if args.baselines:
                for implementation, fn in (("fia_c8", case.run_fia), ("flash_mla_bf16", case.run_bf16)):
                    if implementation == "flash_mla_bf16":
                        case.prepare_bf16()
                    record(
                        {
                            "gate": "kernel_only_graph_latency",
                            "dcp": dcp,
                            "batch": batch,
                            "heads": heads,
                            "tokens": batch * 4,
                            "local_kv_length": length,
                            "prefix_fraction": 0.99,
                            "implementation": implementation,
                            **bench_graph(fn),
                        }
                    )


if __name__ == "__main__":
    main()
