#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-layer TP8/DCP8 decode forwards with synthetic, shared MXFP8 weights.

Run --shape-audit locally without importing NPU dependencies. Native execution:
torchrun --standalone --nproc_per_node=8 flash_mla_c8_forward_benchmark.py --json result.json

This is a module measurement, not model ITL. It invokes the production BF16/C8
forward methods, including projections, cache stores, native history/current
attention, real HCCL merge, V-up, output gate, and O-projection/ReduceScatter.
Both paths retain their current production MLAPO and communication overlap.
Metadata construction, synthetic weight initialization and cache initialization
are outside timing, matching steady-state decode. No full model is constructed.
"""

import argparse
import ast
import json
import math
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

CONTEXTS = (8192, 32768, 131072)
BATCH, QLEN, TP = 16, 4, 8
TOKENS, HEADS, LOCAL_HEADS = BATCH * QLEN, 96, 12
HIDDEN, Q_RANK, LATENT, ROPE, QK_NOPE, VALUE = 7168, 1536, 512, 64, 128, 128
PAGE = 128


def shape_audit():
    cases = []
    for context in CONTEXTS:
        global_pages = context // PAGE
        common_global_pages = math.floor(context * 0.99) // PAGE
        local_pages = global_pages // TP
        shared = [sum(p % TP == rank for p in range(common_global_pages)) for rank in range(TP)]
        physical = [s + BATCH * (local_pages - s + 1) for s in shared]
        cases.append(
            dict(
                global_history=context,
                local_history=context // TP,
                requested_shared_fraction=0.99,
                actual_page_shared_fraction=common_global_pages / global_pages,
                common_pages_per_rank=shared,
                allocated_pages_per_rank=physical,
                current_owner=(context // PAGE) % TP,
            )
        )
    return dict(
        topology="one DP on one TP8/DCP8 machine; EP/MoE are outside an MLA layer",
        per_dp_concurrency=BATCH,
        dspark_tokens=3,
        query_tokens=TOKENS,
        hidden=[TOKENS, HIDDEN],
        fused_qkv_down_weight=[HIDDEN, Q_RANK + LATENT + ROPE],
        q_up_weight=[Q_RANK, HEADS * (QK_NOPE + ROPE)],
        k_absorption_weight=[HEADS, QK_NOPE, LATENT],
        replicated_q=[TOKENS, HEADS, LATENT + ROPE],
        dcp_merged=[TOKENS, LOCAL_HEADS, LATENT],
        v_up_weight=[LOCAL_HEADS, LATENT, VALUE],
        gate_weight=[HIDDEN, LOCAL_HEADS * VALUE],
        o_weight=[LOCAL_HEADS * VALUE, HIDDEN],
        output_after_rs=[TOKENS // TP, HIDDEN],
        kernel_page_size=PAGE,
        weight_source="deterministic synthetic MXFP8 input/Q projections, BF16 UK/UV/gate/O",
        cases=cases,
    )


if __name__ == "__main__" and "--shape-audit" in sys.argv:
    print(json.dumps(shape_audit(), indent=2))
    sys.exit(0)

_NATIVE_OPP_PATH = os.environ.get("ASCEND_CUSTOM_OPP_PATH")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_npu  # noqa: E402
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401
from vllm.distributed.parallel_state import (  # noqa: E402
    destroy_distributed_environment,
    init_distributed_environment,
    init_model_parallel_group,
)

from vllm_ascend.attention.context_parallel.common_cp import (  # noqa: E402
    _dcp_mtp_comm_stream,
    combine_flash_attention_output,
    exchange_flash_attention_output,
    merge_flash_attention_output,
)
from vllm_ascend.ops.triton.copy_mla_kv import copy_mla_kv  # noqa: E402
from vllm_ascend.ops.triton.flash_attention_output import flash_attention_gate, flash_attention_output  # noqa: E402
from vllm_ascend.ops.triton.mla_query_rope import (  # noqa: E402
    quantize_mla_query,
    quantize_mla_query_and_kv,
    scale_mla_query_rope,
)
from vllm_ascend.ops.triton.quantize_mla_kv import quantize_mla_kv  # noqa: E402
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton  # noqa: E402
from vllm_ascend.worker.flash_kv_cache import split_flash_mla_c8_cache  # noqa: E402

if _NATIVE_OPP_PATH is not None:
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = _NATIVE_OPP_PATH


def load_method(path, class_name, method_name, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method_name]


ROOT = Path(__file__).resolve().parents[5]


class SyntheticMXLinear:
    """The actual VA MXFP8 linear scheme, using deterministic synthetic weights."""

    def __init__(self, raw):
        self.raw = raw
        self.quantized, scale = torch_npu.npu_dynamic_mx_quant(raw, dst_type=torch.float8_e4m3fn)
        self.prolog_weight = torch_npu.npu_format_cast(self.quantized.T.contiguous(), 29)
        self.prolog_scale = scale.flatten(1).view(torch.float8_e8m0fnu)
        self.weight = self.quantized.T.contiguous()
        self.weight_scale = scale.reshape(raw.shape[0], -1, 2).transpose(0, 1).contiguous().view(torch.uint8)
        self.scheme = SimpleNamespace(dynamic_mx_quant_scale_alg=0, group_size=32)
        self.apply = load_method(
            ROOT / "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py",
            "AscendW8A8MXFP8DynamicLinearMethod",
            "apply",
            dict(torch=torch, torch_npu=torch_npu),
        )

    def __call__(self, x):
        return self.apply(self.scheme, self, x), None


class SyntheticOProjection:
    """Production decode matmul/RS method with synthetic BF16 O weights."""

    def __init__(self, weight, group, rank):
        self.weight = weight
        self.layer = self
        self.comm_group = group
        self.tp_rank, self.tp_size = rank, TP
        self.custom_op = self
        self.get_input_parallel = lambda x: x
        self.quant_method = SimpleNamespace(apply=lambda layer, x, bias: torch.mm(x, layer.weight.T))
        self._apply_into = load_method(
            ROOT / "vllm_ascend/ops/linear_op.py",
            "KimiOProjMMReduceScatterOp",
            "apply_into",
            dict(torch=torch, dist=dist),
        )

    def apply_into(self, x, output):
        return self._apply_into(self, x, output)

    def __call__(self, x, **kwargs):
        output = torch.empty(x.shape[0] // TP, HIDDEN, device=x.device, dtype=x.dtype)
        return self.apply_into(x, output), None


class SyntheticRMSNorm:
    def __init__(self, weight):
        self.weight = weight
        self.variance_epsilon = 1e-6

    def __call__(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


class ForwardPair:
    def __init__(self, context, group, rank):
        self.context, self.group, self.rank = context, group, rank
        torch.manual_seed(16464)
        self.hidden = torch.randn(TOKENS, HIDDEN, device="npu", dtype=torch.bfloat16)

        def raw_weight(n, k):
            return torch.randn(n, k, device="npu", dtype=torch.bfloat16) * 0.02

        self.wdq = SyntheticMXLinear(raw_weight(Q_RANK, HIDDEN))
        self.wdkv = SyntheticMXLinear(raw_weight(LATENT + ROPE, HIDDEN))
        self.wuq = SyntheticMXLinear(raw_weight(HEADS * (QK_NOPE + ROPE), Q_RANK))
        self.wqkv = SyntheticMXLinear(torch.cat((self.wdq.raw, self.wdkv.raw), 0))
        self.uk = raw_weight(HEADS * QK_NOPE, LATENT).view(HEADS, QK_NOPE, LATENT)
        self.gamma_q = torch.ones(Q_RANK, dtype=torch.bfloat16, device="npu")
        self.gamma_kv = torch.ones(LATENT, dtype=torch.bfloat16, device="npu")
        # Output projections are TP-local slices of the same global matrices.
        torch.manual_seed(16465 + rank)
        self.uv = raw_weight(LOCAL_HEADS * LATENT, VALUE).view(LOCAL_HEADS, LATENT, VALUE)
        self.gate = raw_weight(LOCAL_HEADS * VALUE, HIDDEN)
        self.o_proj = SyntheticOProjection(raw_weight(HIDDEN, LOCAL_HEADS * VALUE), group, rank)
        self.empty_rope = torch.empty(0, ROPE, device="npu", dtype=torch.bfloat16)
        self.sk = torch.tensor([0.02], device="npu", dtype=torch.float32)
        self.inverse_sk = self.sk.reciprocal()
        self._allocate_cache()
        self.scope = dict(
            torch=torch,
            torch_npu=torch_npu,
            envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
            get_dynamic_mx_quant_scale_alg=lambda _: 0,
            _npu_mla_prolog_v3_no_rope=self._checked_prolog,
            DecodeMLAPreprocessResult=lambda q, qr, k, kr, **kwargs: SimpleNamespace(ql_nope=q, q_pe=qr, **kwargs),
            MLAPO_MAX_SUPPORTED_TOKENS=128,
            FLASH_DCP_QUERY_PREP_MAX_TOKENS=16,
            DeviceMetadataStage=SimpleNamespace(ATTENTION=0),
            wait_for_device_metadata=lambda *args: None,
            wait_for_kv_layer_from_connector=lambda *args: None,
            notify_kv_cache_written=lambda *args: None,
            copy_mla_kv=copy_mla_kv,
            quantize_mla_kv=quantize_mla_kv,
            scale_mla_query_rope=scale_mla_query_rope,
            quantize_mla_query=quantize_mla_query,
            quantize_mla_query_and_kv=quantize_mla_query_and_kv,
            _dcp_mtp_comm_stream=_dcp_mtp_comm_stream,
            exchange_flash_attention_output=exchange_flash_attention_output,
            combine_flash_attention_output=combine_flash_attention_output,
            merge_flash_attention_output=merge_flash_attention_output,
            flash_attention_gate=flash_attention_gate,
            flash_attention_output=flash_attention_output,
            KimiOProjMMReduceScatterOp=SyntheticOProjection,
        )
        path = ROOT / "vllm_ascend/attention/mla_v1.py"
        self.bf16_forward = load_method(path, "AscendMLAImpl", "_forward_flash", self.scope)
        self.c8_forward = load_method(path, "AscendMLAImpl", "_forward_flash_c8", self.scope)
        self.v_up = load_method(path, "AscendMLAImpl", "_v_up_proj_batch_major", self.scope)
        self.preprocess = load_method(path, "AscendMLAImpl", "mla_preprocess_only_decode", self.scope)
        self.output = {
            name: torch.empty(TOKENS // TP, HIDDEN, dtype=torch.bfloat16, device="npu") for name in ("bf16", "c8")
        }
        self.impl = {name: self._implementation(name) for name in self.output}

    def _allocate_cache(self):
        local_pages = self.context // TP // PAGE
        prefix_pages = math.floor(self.context * 0.99) // PAGE
        common = sum(p % TP == self.rank for p in range(prefix_pages))
        pages = common + BATCH * (local_pages - common + 1)
        table = torch.empty(BATCH, local_pages + 1, dtype=torch.int32)
        cursor = common
        for request in range(BATCH):
            table[request, :common] = torch.arange(common, dtype=torch.int32)
            private_count = local_pages - common + 1
            table[request, common:] = torch.arange(cursor, cursor + private_count, dtype=torch.int32)
            cursor += private_count
        gen = torch.Generator().manual_seed(16467 + self.rank)
        permutation = torch.randperm(pages, generator=gen)
        table = permutation[table.long()].int()
        slots = torch.full((TOKENS,), -1, dtype=torch.int64)
        if (self.context // PAGE) % TP == self.rank:
            for request in range(BATCH):
                slots[request * QLEN : (request + 1) * QLEN] = table[request, -1] * PAGE + torch.arange(QLEN)
        self.cache_bf16_storage = torch.empty(pages, 2, PAGE, LATENT + ROPE, dtype=torch.bfloat16, device="npu")
        self.cache_bf16 = self.cache_bf16_storage[:, 0]
        self.cache_c8_storage = torch.empty(pages, 2, PAGE, LATENT + 2 * ROPE, dtype=torch.uint8, device="npu")
        self.cache_c8 = split_flash_mla_c8_cache(self.cache_c8_storage[:, 0].view(torch.float8_e4m3fn))
        # Same quantized historical values in both formats, prepared once.
        # Widening here is only a benchmark input constructor, never timed.
        latent = torch.randn(pages, PAGE, 1, LATENT, device="npu", dtype=torch.float32) * 0.5
        latent = (latent / self.sk).clamp(-448, 448).to(torch.float8_e4m3fn)
        rope = torch.randn(pages, PAGE, 1, ROPE, device="npu", dtype=torch.bfloat16) * 0.5
        self.cache_c8[0].copy_(latent)
        self.cache_c8[1].copy_(rope)
        self.cache_bf16[..., :LATENT].copy_((latent.float() * self.sk).squeeze(2))
        self.cache_bf16[..., LATENT:].copy_(rope.squeeze(2))
        self.cu = torch.arange(0, TOKENS + 1, QLEN, dtype=torch.int32, device="npu")
        self.used = torch.full((BATCH,), QLEN, dtype=torch.int32, device="npu")
        self.lengths = torch.full((BATCH,), self.context // TP, dtype=torch.int32, device="npu")
        self.table, self.slots = table.to("npu"), slots.to("npu")
        self.current_blocks = torch.arange(BATCH, dtype=torch.int32, device="npu").view(BATCH, 1)
        self.current_slots = torch.tensor(
            [b * PAGE + row for b in range(BATCH) for row in range(QLEN)], dtype=torch.int64, device="npu"
        )
        self.mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1)
        self.meta = {}
        for name in ("bf16", "c8"):
            current_cache = torch.empty(BATCH, 1, PAGE, LATENT + ROPE, dtype=torch.bfloat16, device="npu")
            history_schedule = self._metadata(self.lengths, HEADS, self.context // TP, is_c8=name == "c8")
            current_schedule = self._metadata(self.used, LOCAL_HEADS, QLEN, is_c8=False, current=True)
            bundle = SimpleNamespace(
                query=torch.empty(TOKENS, HEADS, LATENT + ROPE, dtype=torch.bfloat16, device="npu"),
                split_kv=True,
                unabsorbed=False,
                dcp_size=TP,
                is_prefill=False,
                causal=True,
                schedule=history_schedule,
                current_schedule=current_schedule,
                current_cache=current_cache,
                current_block_table=self.current_blocks,
                current_slots=self.current_slots,
                slots=self.slots,
                block_table=self.table,
                cache_lens=self.lengths,
                used_q=self.used,
                cu=self.cu,
                max_query_len=QLEN,
                max_seq_len=self.context // TP,
                token_live=torch.ones(TOKENS, dtype=torch.bool, device="npu"),
                attn_mask=self.mask,
            )
            self.meta[name] = SimpleNamespace(flash=bundle, causal=True)

    def _metadata(self, lengths, heads, maximum, *, is_c8, current=False):
        return torch.ops._C_ascend.flash_mla_with_kvcache_metadata(
            lengths,
            heads,
            1,
            cu_seqlens_q=self.cu,
            seqused_q=self.used,
            max_seqlen_q=QLEN,
            max_seqlen_kv=maximum,
            head_dim_qk=LATENT + ROPE,
            head_dim_v=LATENT,
            mask_mode=3 if current else 0,
            layout_q="TND",
            is_c8=is_c8,
        )

    def _query_projection(self, q):
        projected = self.wuq(q)[0].view(TOKENS, HEADS, QK_NOPE + ROPE)
        q_nope, q_rope = projected.split((QK_NOPE, ROPE), -1)
        q_latent = torch.bmm(q_nope.transpose(0, 1), self.uk).transpose(0, 1)
        return q_latent, q_rope

    def _checked_prolog(self, **kwargs):
        # Validate the real production wrapper selected mode 7 for both DCP
        # paths, instead of relying on a test-only hand-written prolog call.
        assert kwargs["kv_cache_quant_mode"] == kwargs["query_quant_mode"] == 0
        assert kwargs["quant_scale_ckv"] is None
        assert kwargs["kv_cache"].dtype == kwargs["kr_cache"].dtype == torch.bfloat16
        assert kwargs["cache_index"].data_ptr() == self.current_slots.data_ptr()
        current_ptrs = [meta.flash.current_cache.data_ptr() for meta in self.meta.values()]
        assert kwargs["kv_cache"].data_ptr() in current_ptrs
        return torch.ops._C_ascend.npu_mla_prolog_v3(**kwargs)

    def _implementation(self, name):
        impl = SimpleNamespace(
            fa_quant_layer=name == "c8",
            enable_mlapo=True,
            mlapo_weight_quant_mode=3,
            support_fp8_attention=True,
            vllm_config=None,
            weight_dq=self.wdq.prolog_weight,
            weight_dkv_kr=self.wdkv.prolog_weight,
            weight_uq_qr=self.wuq.prolog_weight,
            dequant_scale_w_dq=self.wdq.prolog_scale,
            dequant_scale_w_dkv_kr=self.wdkv.prolog_scale,
            dequant_scale_w_uq_qr=self.wuq.prolog_scale,
            mlapo_W_UK_T=self.uk,
            mlapo_num_heads=HEADS,
            reorg_decode_q=lambda q, qr: (q, qr),
            _mlapo_empty_rope=self.empty_rope,
            num_heads=LOCAL_HEADS,
            kv_lora_rank=LATENT,
            v_head_dim=VALUE,
            W_UV=self.uv,
            fused_qkv_a_proj=self.wqkv,
            q_lora_rank=Q_RANK,
            q_a_layernorm=SyntheticRMSNorm(self.gamma_q),
            kv_a_layernorm=SyntheticRMSNorm(self.gamma_kv),
            _q_proj_and_k_up_proj=self._query_projection,
            q_proj=SimpleNamespace(
                qrep_active=True,
                rank_in_group=self.rank,
                _local_view=lambda x: x[:, self.rank * LOCAL_HEADS : (self.rank + 1) * LOCAL_HEADS].contiguous(),
            ),
            use_mla_rope=False,
            layerwise_kv_cache_hook=None,
            flash_dcp_overlap=True,
            flash_dcp_preprocess_overlap=True,
            flash_dcp_query_prep=True,
            dcp_group=self.group,
            fak_descale_float=self.sk,
            fak_descale_reciprocal=self.inverse_sk,
            scale=(QK_NOPE + ROPE) ** -0.5,
            use_output_gate=True,
            g_proj=lambda x: (torch.mm(x, self.gate.T), None),
            o_proj=self.o_proj,
        )
        impl._v_up_proj_batch_major = lambda x: self.v_up(impl, x)
        impl.mla_preprocess_only_decode = lambda hidden, cache, meta: self.preprocess(impl, hidden, cache, meta)
        return impl

    def run(self, name):
        method = self.c8_forward if name == "c8" else self.bf16_forward
        cache = self.cache_c8 if name == "c8" else self.cache_bf16
        return method(self.impl[name], "synthetic_mla_layer", self.hidden, cache, self.meta[name], self.output[name])

    def zero_gate(self):
        hidden_first = self.hidden[0].clone()
        uk_first = self.uk[0].clone()
        self.hidden[0].zero_()
        self.uk[0].zero_()
        previous_latent = self.cache_c8[0].float().cpu()
        previous_rope = self.cache_c8[1].cpu()
        prolog, _ = self.impl["c8"].mla_preprocess_only_decode(self.hidden, self.cache_c8, self.meta["c8"])
        assert prolog.ql_nope.dtype == prolog.q_pe.dtype == torch.bfloat16
        assert prolog.dequant_scale_q_nope is None or prolog.dequant_scale_q_nope.numel() == 0
        assert torch.count_nonzero(prolog.ql_nope[0].cpu()) == 0
        assert torch.count_nonzero(prolog.q_pe[0].cpu()) == 0
        assert torch.count_nonzero(prolog.ql_nope[:, 0].cpu()) == 0
        assert torch.count_nonzero(prolog.q_pe[1:, 0].cpu()) > 0
        for name in ("bf16", "c8"):
            graph = capture(self, name, 1)
            assert torch.isfinite(self.output[name].cpu()).all()
            del graph
        torch.npu.synchronize()
        current = self.meta["c8"].flash.current_cache[:, 0].cpu()
        reciprocal = self.inverse_sk.cpu()
        for src, dst in zip(self.current_slots.cpu().tolist(), self.slots.cpu().tolist()):
            if dst >= 0:
                row = current[src // PAGE, src % PAGE]
                previous_latent[dst // PAGE, dst % PAGE, 0] = (
                    (row[:LATENT].float() * reciprocal).clamp(-448, 448).to(torch.float8_e4m3fn).float()
                )
                previous_rope[dst // PAGE, dst % PAGE, 0] = row[LATENT:]
        torch.testing.assert_close(self.cache_c8[0].float().cpu(), previous_latent, rtol=0, atol=0)
        torch.testing.assert_close(self.cache_c8[1].cpu(), previous_rope, rtol=0, atol=0)
        self.hidden[0].copy_(hidden_first)
        self.uk[0].copy_(uk_first)


def capture(pair, name, capture_forwards):
    for _ in range(3):
        pair.run(name)
    torch.npu.synchronize()
    eager = pair.output[name].clone()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(capture_forwards):
            pair.run(name)
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(pair.output[name].cpu(), eager.cpu(), rtol=0.01, atol=0.002)
    return graph


def time_graph(graph, repeats, capture_forwards):
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / (repeats * capture_forwards)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--capture-forwards", type=int, default=8)
    parser.add_argument("--warm-calls", type=int, default=3000)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == TP
    assert all(context > 0 and context % (TP * PAGE) == 0 for context in args.contexts)
    torch.npu.set_device(rank)
    torch.set_num_threads(4)
    init_device_properties_triton()
    init_distributed_environment(
        world_size=TP, rank=rank, local_rank=rank, distributed_init_method="env://", backend="hccl"
    )
    group = init_model_parallel_group(
        [list(range(TP))],
        local_rank=rank,
        backend="hccl",
        group_name="c8_full_mla_benchmark",
        use_device_communicator=False,
    )
    report = dict(rank=rank, shape_audit=shape_audit(), capture_forwards=args.capture_forwards, results=[])
    try:
        for context in args.contexts:
            pair = ForwardPair(context, group, rank)
            pair.zero_gate()
            graphs = {name: capture(pair, name, args.capture_forwards) for name in ("bf16", "c8")}
            baseline, actual = (pair.output[name].float().cpu() for name in ("bf16", "c8"))
            assert torch.isfinite(baseline).all() and torch.isfinite(actual).all()
            difference = actual - baseline
            numerical = dict(
                max_abs=float(difference.abs().max()),
                normalized_rmse=float((difference.square().mean() / baseline.square().mean().clamp_min(1e-12)).sqrt()),
                note="Includes C8 Q/P quantization; not model accuracy.",
            )
            assert numerical["normalized_rmse"] < 0.1, numerical
            for graph in graphs.values():
                for _ in range(math.ceil(args.warm_calls / args.capture_forwards)):
                    graph.replay()
                torch.npu.synchronize()
            trials = {"bf16": [], "c8": []}
            for round_index in range(args.rounds):
                order = ("bf16", "c8") if round_index % 2 == 0 else ("c8", "bf16")
                for name in order:
                    dist.barrier(group=group.device_group)
                    trials[name].append(time_graph(graphs[name], args.replays, args.capture_forwards))
            item = dict(
                context=context,
                active_zero_query_graph_gate="PASS",
                output_error=numerical,
                bf16=dict(median_us=statistics.median(trials["bf16"]), trials_us=trials["bf16"]),
                c8=dict(median_us=statistics.median(trials["c8"]), trials_us=trials["c8"]),
            )
            report["results"].append(item)
            print(json.dumps(dict(rank=rank, **item)), flush=True)
            dist.barrier(group=group.device_group)
            del graphs, pair
            torch.npu.empty_cache()
        rank_path = args.json.with_name(f"{args.json.stem}.rank{rank}{args.json.suffix}")
        rank_path.parent.mkdir(parents=True, exist_ok=True)
        rank_path.write_text(json.dumps(report, indent=2))
        dist.barrier(group=group.device_group)
    finally:
        group.destroy()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
