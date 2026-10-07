#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""C8 non-absorbed MLA attention qualification, with a bounded CPU oracle.

--shape-audit requires only Python. Native runs require the matching FlashAttn
build. Timings name attention, operand preparation, or their combined scope;
projection, metadata, cache writes, communication and O projection are excluded. A TP8 rank has
12 of the model's 96 heads; PP4 does not divide a layer's sequence length.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path


def shape_audit(query_lengths=(131072,), kv_lengths=(131072,), heads=12, causal=True):
    if len(query_lengths) != len(kv_lengths) or any(q < 1 or k < q for q, k in zip(query_lengths, kv_lengths)):
        raise ValueError("require matching nonempty requests with 0 < query_length <= kv_length")
    pairs = sum(q * (k - q) + q * (q + 1) // 2 if causal else q * k for q, k in zip(query_lengths, kv_lengths))
    return dict(
        scope="one native attention call on one TP8 rank, not module latency or PP4 model TTFT",
        pipeline_parallel_size=4,
        tensor_parallel_size=8,
        local_heads=heads,
        query_lengths=list(query_lengths),
        kv_lengths=list(kv_lengths),
        call_history_lengths=[k - q for q, k in zip(query_lengths, kv_lengths)],
        mask_mode=3 if causal else 0,
        q_nope=[sum(query_lengths), heads, 128],
        q_rope=[sum(query_lengths), heads, 64],
        k_nope=[sum(kv_lengths), heads, 128],
        k_rope=[sum(kv_lengths), heads, 64],
        value=[sum(kv_lengths), heads, 128],
        causal_pairs_per_head=pairs,
        useful_fp8_cube_flops=2 * heads * pairs * (128 + 128),
        useful_bf16_rope_cube_flops=2 * heads * pairs * 64,
        note="Useful FLOPs exclude masked tile padding, softmax, data movement, and scale operations.",
    )


if __name__ == "__main__" and "--shape-audit" in sys.argv:
    print(json.dumps(shape_audit(), indent=2))
    sys.exit(0)

import torch  # noqa: E402


def cumulative(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def sampled_rows(length):
    """All short rows; long cases cover both tails and tile/chunk boundaries."""
    if length <= 257:
        return list(range(length))
    indices = {0, 1, length - 2, length - 1}
    for boundary in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
        indices.update(i for i in (boundary - 1, boundary, boundary + 1) if i < length)
    indices.update(i * (length - 1) // 16 for i in range(17))
    return sorted(indices)


def streaming_reference(case, *, key_tile=2048):
    """FP64 online softmax over every key for selected queries, never S by S.

    The oracle consumes the actual FP8 codes and the actual BF16 rope operands.
    It measures kernel error independently from input quantization error. Its
    largest score buffer is [heads, sampled_queries, key_tile].
    """
    outputs, lses, global_rows = [], [], []
    q_offsets, k_offsets = cumulative(case.query_lengths), cumulative(case.kv_lengths)
    for batch, (qlen, klen) in enumerate(zip(case.query_lengths, case.kv_lengths)):
        rows = torch.tensor(sampled_rows(qlen), dtype=torch.long)
        absolute_rows = rows + q_offsets[batch]
        q = case.q_cpu[absolute_rows].double().transpose(0, 1)
        qr = case.qr_cpu[absolute_rows].double().transpose(0, 1)
        sq = case.sq_cpu[absolute_rows].double().transpose(0, 1)
        maximum = torch.full((case.heads, len(rows)), -torch.inf, dtype=torch.float64)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros((case.heads, len(rows), 128), dtype=torch.float64)
        for start in range(0, klen, key_tile):
            stop = min(start + key_tile, klen)
            key_slice = slice(k_offsets[batch] + start, k_offsets[batch] + stop)
            key = case.k_cpu[key_slice].double().permute(1, 2, 0)
            rope = case.kr_cpu[key_slice].double().permute(1, 2, 0)
            value = case.v_cpu[key_slice].double().transpose(0, 1) * case.scale_v[:, None, None]
            descale = sq * case.scale_k[:, None] * case.softmax_scale
            logits = (torch.bmm(q, key) + torch.bmm(qr, rope)) * descale[..., None]
            if case.causal:
                live = torch.arange(start, stop)[None, :] <= (rows + klen - qlen)[:, None]
                logits.masked_fill_(~live[None], -torch.inf)
            new_maximum = torch.maximum(maximum, logits.amax(-1))
            rescale = torch.exp(maximum - new_maximum)
            probability = torch.exp(logits - new_maximum[..., None])
            numerator = numerator * rescale[..., None] + torch.bmm(probability, value)
            denominator = denominator * rescale + probability.sum(-1)
            maximum = new_maximum
        outputs.append((numerator / denominator[..., None]).transpose(0, 1))
        lses.append((maximum + denominator.log()).transpose(0, 1))
        global_rows.append(absolute_rows)
    return torch.cat(outputs), torch.cat(lses), torch.cat(global_rows)


def error_metrics(actual, expected):
    difference = actual.double() - expected.double()
    return dict(
        max_abs=float(difference.abs().amax()),
        normalized_rmse=float((difference.square().mean() / expected.double().square().mean().clamp_min(1e-12)).sqrt()),
    )


class C8PrefillCase:
    def __init__(self, query_lengths, kv_lengths=None, heads=12, *, causal=True, seed=20260921, zero_nope=False):
        self.query_lengths = tuple(query_lengths)
        self.kv_lengths = tuple(kv_lengths or query_lengths)
        shape_audit(self.query_lengths, self.kv_lengths, heads, causal)
        self.heads, self.causal = heads, causal
        self.scale_k = torch.linspace(0.01, 0.025, heads)
        self.scale_v = torch.linspace(0.018, 0.036, heads)
        self.softmax_scale = 192**-0.5
        generator = torch.Generator().manual_seed(seed)
        qtokens, ktokens = sum(self.query_lengths), sum(self.kv_lengths)

        def random(shape):
            return torch.randn(shape, generator=generator) * 0.5

        def fp8(value):
            return value.clamp(-448, 448).to(torch.float8_e4m3fn)

        self.sq_cpu = torch.linspace(0.003, 0.017, qtokens * heads).reshape(qtokens, heads)
        self.q_cpu = fp8(random((qtokens, heads, 128)) / self.sq_cpu[..., None])
        if zero_nope:
            self.q_cpu.zero_()
        self.qr_cpu = (random((qtokens, heads, 64)) / self.sq_cpu[..., None] / self.scale_k[None, :, None]).bfloat16()
        self.k_cpu = fp8(random((ktokens, heads, 128)) / self.scale_k[None, :, None])
        self.kr_cpu = random((ktokens, heads, 64)).bfloat16()
        self.v_cpu = fp8(random((ktokens, heads, 128)) / self.scale_v[None, :, None])

    def to_npu(self):
        for name in ("q", "qr", "k", "kr", "v", "sq"):
            setattr(self, name, getattr(self, name + "_cpu").to("npu"))
        from cann_ops_transformer.ops import flash_attn_metadata

        self.sk = self.scale_k.to("npu")
        self.sv = self.scale_v.to("npu")
        self.mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8), diagonal=1).to("npu")
        self.arguments = dict(
            cu_seqlens_q=torch.tensor(cumulative(self.query_lengths), dtype=torch.int32, device="npu"),
            cu_seqlens_kv=torch.tensor(cumulative(self.kv_lengths), dtype=torch.int32, device="npu"),
            seqused_q=torch.tensor(self.query_lengths, dtype=torch.int32, device="npu"),
            seqused_kv=torch.tensor(self.kv_lengths, dtype=torch.int32, device="npu"),
            max_seqlen_q=max(self.query_lengths),
            max_seqlen_kv=max(self.kv_lengths),
            mask_mode=3 if self.causal else 0,
            layout_q="TND",
            layout_kv="TND",
            layout_out="TND",
        )
        self.bf16_metadata = flash_attn_metadata(
            self.heads, self.heads, 192, head_dim_v=128, batch_size=len(self.query_lengths), **self.arguments
        )
        # Native C8 uses M128/N128. The generic scheduler's short-Q branch
        # otherwise emits M64/N256, so only its maximum-shape hint is widened.
        c8_arguments = self.arguments | {"max_seqlen_q": max(65, max(self.query_lengths))}
        self.metadata = flash_attn_metadata(
            self.heads, self.heads, 192, head_dim_v=128, batch_size=len(self.query_lengths), **c8_arguments
        )
        header = self.metadata[:6].cpu().tolist()
        if header[2:4] != [128, 128]:
            raise RuntimeError(f"C8 native kernel requires metadata M128/N128, got header {header}")
        return self

    def run(self, operands=None):
        if operands is None:
            operands = (self.q, self.k, self.v, self.qr, self.kr, self.sq, self.sk, self.sv)
        return torch.ops._C_ascend.flash_attn_c8(
            *operands,
            self.arguments["cu_seqlens_q"],
            self.arguments["cu_seqlens_kv"],
            self.metadata,
            softmax_scale=self.softmax_scale,
            mask_mode=self.arguments["mask_mode"],
            max_seqlen_q=max(self.query_lengths),
            max_seqlen_kv=max(self.kv_lengths),
            seqused_q=self.arguments["seqused_q"],
            attn_mask=self.mask if self.causal else None,
            return_softmax_lse=True,
        )

    def prepare_bf16(self):
        self.raw_query = torch.cat(
            (
                self.q.float() * self.sq[..., None],
                self.qr.float() * self.sq[..., None] * self.sk[None, :, None],
            ),
            -1,
        ).bfloat16()
        projected = torch.cat(
            (self.k.float() * self.sk[None, :, None], self.v.float() * self.sv[None, :, None]), -1
        ).bfloat16()
        self.raw_key, self.raw_value = projected.split(128, dim=-1)
        self.raw_rope = self.kr
        return self.raw_query, self.raw_key, self.raw_value, self.raw_rope

    def run_bf16(self, operands):
        from cann_ops_transformer.ops import flash_attn

        return flash_attn(
            *operands,
            metadata=self.bf16_metadata,
            softmax_scale=self.softmax_scale,
            attn_mask=self.mask if self.causal else None,
            return_softmax_lse=True,
            **self.arguments,
        )

    def check(self, output, lse):
        assert output.dtype == torch.bfloat16 and lse.dtype == torch.float32
        assert tuple(output.shape) == (sum(self.query_lengths), self.heads, 128)
        assert tuple(lse.shape) == (self.heads, sum(self.query_lengths))
        assert torch.isfinite(output).all() and torch.isfinite(lse).all()
        expected, expected_lse, rows = streaming_reference(self)
        actual = output[rows.to(output.device)].cpu().float()
        actual_lse = lse[:, rows.to(lse.device)].T.cpu()
        # As in the absorbed C8 qualification, P is rounded to FP8 for P @ V.
        # LSE independently checks the logits/rope/descale/causal path.
        torch.testing.assert_close(actual.double(), expected, rtol=0.05, atol=0.03)
        torch.testing.assert_close(actual_lse.double(), expected_lse, rtol=0.0002, atol=0.002)
        return dict(
            **shape_audit(self.query_lengths, self.kv_lengths, self.heads, self.causal),
            checked_queries=rows.tolist(),
            checked_query_count=len(rows),
            checked_keys="all permitted keys for every checked query",
            output_error=error_metrics(actual, expected),
            lse_error=error_metrics(actual_lse, expected_lse),
        )


def benchmark(function, *, warmup=5, trials=9, repeats=1, graph=False):
    """Completed device-event samples; warmup and graph capture are excluded."""
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    if graph:
        captured = torch.npu.NPUGraph()
        with torch.npu.graph(captured):
            captured_output = function()
        function = captured.replay
        for _ in range(warmup):
            function()
        torch.npu.synchronize()
    samples = []
    for _ in range(trials):
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(repeats):
            function()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / repeats)
    if graph:
        del captured_output
    return dict(
        mode="graph" if graph else "eager",
        warmup_calls=warmup,
        calls_per_sample=repeats,
        median_us=statistics.median(samples),
        minimum_us=min(samples),
        maximum_us=max(samples),
        samples_us=samples,
        timing_scope="device events around the named callable, includes dispatch gaps in eager mode",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--query-length", type=int, default=8192)
    parser.add_argument("--kv-length", type=int)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    import torch_npu  # noqa: F401
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    from vllm_ascend.ops.flash_attn_c8_quant import fake_quant_flash_attn_c8, quantize_flash_attn_c8

    torch.npu.set_device(args.device)
    torch.set_num_threads(8)
    records = []

    def record(result):
        records.append(result)
        args.json.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)

    with torch.inference_mode():
        case = C8PrefillCase((args.query_length,), (args.kv_length or args.query_length,), heads=args.heads).to_npu()
        record({"gate": "native_c8_accuracy", **case.check(*case.run())})
        if args.benchmark:
            inputs = case.prepare_bf16()
            quantized = quantize_flash_attn_c8(*inputs)
            fake = fake_quant_flash_attn_c8(*inputs)
            prepared_reference = copy.copy(case)
            for name, tensor in zip(
                ("q_cpu", "k_cpu", "v_cpu", "qr_cpu", "kr_cpu", "sq_cpu", "scale_k", "scale_v"), quantized
            ):
                setattr(prepared_reference, name, tensor.cpu())
            record(
                {"gate": "native_preparation_and_attention_accuracy", **prepared_reference.check(*case.run(quantized))}
            )
            partial = torch.ops._C_ascend.flash_attn_c8_quant_stats(inputs[1], inputs[2])
            # This is an equal-BF16-input subchain comparison. Projection and
            # permanent cache write are not represented by these measurements.
            functions = (
                ("prequantized_native_c8_attention", lambda: case.run(quantized)),
                ("quantize_stats_only", lambda: torch.ops._C_ascend.flash_attn_c8_quant_stats(inputs[1], inputs[2])),
                (
                    "quantize_prepare_given_stats",
                    lambda: torch.ops._C_ascend.flash_attn_c8_prepare(*inputs, partial, False),
                ),
                ("quantize_prepare_only", lambda: quantize_flash_attn_c8(*inputs)),
                ("fake_quant_prepare_only", lambda: fake_quant_flash_attn_c8(*inputs)),
                ("prepare_and_native_c8_attention", lambda: case.run(quantize_flash_attn_c8(*inputs))),
                ("prepare_and_fake_quant_bf16_attention", lambda: case.run_bf16(fake_quant_flash_attn_c8(*inputs))),
                ("prepared_fake_quant_bf16_attention", lambda: case.run_bf16(fake)),
            )
            for name, function in functions:
                record(
                    dict(
                        gate="prepare_attention_subchain_latency",
                        implementation=name,
                        query_length=args.query_length,
                        kv_length=args.kv_length or args.query_length,
                        heads=args.heads,
                        excludes=(
                            "projection, RMSNorm, RoPE rotation, persistent cache writes, "
                            "O projection, PP/TP collectives"
                        ),
                        **benchmark(function, warmup=args.warmup, trials=args.trials, graph=args.graph),
                    )
                )


if __name__ == "__main__":
    main()
