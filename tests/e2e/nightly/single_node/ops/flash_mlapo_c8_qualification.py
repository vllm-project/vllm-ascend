#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A5 MXFP8 MLAPO -> C8 FlashMLA direct operator-chain qualification."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from flash_mla_c8_qualification import C8Case, bench_graph, error
from singlecard_ops.test_mlapo_flash_c8 import (
    make_mxfp8_prolog_inputs,
    test_mxfp8_prolog_c8_zero_hidden_stays_finite,
)

from vllm_ascend.worker.flash_kv_cache import split_flash_mla_c8_cache


class MLAPOChain:
    def __init__(self, lengths, shared=False):
        self.case = case = C8Case(lengths, 12, scale_k=0.02, shared=shared)
        pages = case.k.shape[0]
        self.storage = torch.full((pages, 2, 128, 640), 42, dtype=torch.uint8, device="npu")
        case.k, case.kr = split_flash_mla_c8_cache(self.storage[:, 0].view(torch.float8_e4m3fn))
        case.k.copy_(case.k_cpu)
        case.kr.copy_(case.kr_cpu)
        self.bf16_storage = torch.empty(pages, 2, 128, 576, dtype=torch.bfloat16, device="npu")
        self.bf16_cache = self.bf16_storage[:, 0].unsqueeze(2)
        self.bf16_cache[..., :512].copy_(case.k_cpu.float().mul(case.scale_k).bfloat16())
        self.bf16_cache[..., 512:].copy_(case.kr_cpu)
        self.kwargs, _ = make_mxfp8_prolog_inputs(case.batch * case.qlen, case.heads)
        slots = []
        for b, length in enumerate(lengths):
            for token in range(length - case.qlen, length):
                slots.append(int(case.blocks_cpu[b, token // 128]) * 128 + token % 128)
        self.kwargs["cache_index"] = torch.tensor(slots, dtype=torch.int64, device="npu")
        self.quant_scale = torch.tensor([1 / case.scale_k], device="npu")
        self.c8_meta = case.metadata()
        self.bf16_meta = case.metadata(is_c8=False)

    def run_c8(self):
        case = self.case
        q, qr, scale, _, _ = torch.ops._C_ascend.npu_mla_prolog_v3(
            kv_cache=case.k,
            kr_cache=case.kr,
            kv_cache_quant_mode=1,
            query_quant_mode=1,
            quant_scale_ckv=self.quant_scale,
            **self.kwargs,
        )
        # Pass the native three-dimensional descale directly; no hidden Q/K
        # layout copy, QR rescaling, cache widening, or separate KV scatter.
        out, lse = torch.ops._C_ascend.flash_mla_with_kvcache(
            q,
            case.k,
            query_rope=qr,
            key_rope=case.kr,
            dequant_scale_query=scale,
            dequant_scale_key=case.sk,
            block_table=case.blocks,
            cache_seqlens=case.lengths,
            cu_seqlens_q=case.cu,
            seqused_q=case.used,
            metadata=self.c8_meta,
            head_dim_v=512,
            softmax_scale=case.scale,
            max_seqlen_q=case.qlen,
            max_seqlen_kv=case.max_length,
            layout_q="TND",
            layout_kv="PA_BBND",
            layout_out="TND",
            return_softmax_lse=True,
        )
        return out, lse, q, qr, scale

    def run_bf16(self):
        case = self.case
        q, qr, *_ = torch.ops._C_ascend.npu_mla_prolog_v3(
            kv_cache=self.bf16_cache[..., :512],
            kr_cache=self.bf16_cache[..., 512:],
            **self.kwargs,
        )
        return torch.ops._C_ascend.flash_mla_with_kvcache(
            torch.cat((q, qr), dim=-1),
            self.bf16_cache,
            block_table=case.blocks,
            cache_seqlens=case.lengths,
            cu_seqlens_q=case.cu,
            seqused_q=case.used,
            metadata=self.bf16_meta,
            head_dim_v=512,
            softmax_scale=case.scale,
            max_seqlen_q=case.qlen,
            max_seqlen_kv=case.max_length,
            layout_q="TND",
            layout_kv="PA_BBND",
            layout_out="TND",
            return_softmax_lse=True,
        )

    def check(self, cpu=True):
        case = self.case
        out, lse, q, qr, scale = self.run_c8()
        assert scale.shape == (case.batch * case.qlen, case.heads, 1)
        case.q, case.qr, case.sq = q, qr, scale.squeeze(-1)
        fia_out, fia_lse = case.run_fia()
        fia_lse = fia_lse.squeeze(-1).T
        torch.testing.assert_close(out, fia_out, rtol=0.005, atol=0.0005)
        torch.testing.assert_close(lse, fia_lse, rtol=0.0002, atol=0.002)
        record = {
            "lengths": case.lengths_cpu,
            "q_scale_shape": list(scale.shape),
            "flash_vs_fia_output": error(out, fia_out),
            "flash_vs_fia_lse": error(lse, fia_lse),
        }
        if cpu:
            case.q_cpu, case.qr_cpu, case.sq_cpu = q.cpu(), qr.cpu(), scale.squeeze(-1).cpu()
            case.k_cpu, case.kr_cpu = case.k.cpu(), case.kr.cpu()
            record["cpu_reference"] = case.check_result(out, lse)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            replayed = self.run_c8()
        graph.replay()
        torch.npu.synchronize()
        for actual, expected in zip(replayed[:2], (out, lse)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.all(self.storage[:, 1].cpu() == 42)
        record["graph"] = "PASS"
        return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    torch.npu.set_device(0)
    torch.set_num_threads(8)
    results = []

    def record(value):
        results.append(value)
        args.json.write_text(json.dumps(results, indent=2))
        print(json.dumps(value), flush=True)

    with torch.inference_mode():
        test_mxfp8_prolog_c8_zero_hidden_stays_finite()
        record({"gate": "zero_hidden_active_token", "status": "PASS"})
        short = MLAPOChain([129, 257, 513, 1025, 129, 257, 513, 1025])
        record({"gate": "mlapo_c8_flashmla_direct_chain", **short.check()})
        short.kwargs["token_x"].zero_()
        record({"gate": "zero_hidden_mlapo_c8_flashmla_direct_chain", **short.check()})
        if args.benchmark:
            large = MLAPOChain([131072] * 8, shared=True)
            record({"gate": "mlapo_c8_flashmla_128k_99pct", **large.check(cpu=False)})
            for round_id in range(3):
                kinds = ("bf16", "c8") if round_id % 2 == 0 else ("c8", "bf16")
                for kind in kinds:
                    record(
                        {
                            "gate": "same_graph_chain_benchmark",
                            "round": round_id,
                            "cache": kind,
                            **bench_graph(getattr(large, "run_" + kind)),
                        }
                    )


if __name__ == "__main__":
    main()
