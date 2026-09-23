# SPDX-License-Identifier: Apache-2.0
"""NPU gate for production C8 attention orchestration without model weights."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa: F401
from vllm_ascend.ops.triton.flash_attention_output import flash_attention_output

from vllm_ascend.attention.context_parallel.common_cp import merge_flash_attention_output
from vllm_ascend.ops.triton.mla_query_rope import scale_mla_query_rope
from vllm_ascend.ops.triton.quantize_mla_kv import quantize_mla_kv
from vllm_ascend.worker.flash_kv_cache import split_flash_mla_c8_cache


def load_c8_forward():
    path = Path(__file__).resolve().parents[5] / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_flash_c8")
    scope = dict(
        torch=torch,
        torch_npu=torch_npu,
        wait_for_device_metadata=lambda *args: None,
        DeviceMetadataStage=SimpleNamespace(ATTENTION=0),
        MLAPO_MAX_SUPPORTED_TOKENS=128,
        wait_for_kv_layer_from_connector=lambda *args: None,
        quantize_mla_kv=quantize_mla_kv,
        scale_mla_query_rope=scale_mla_query_rope,
        notify_kv_cache_written=lambda *args: None,
        merge_flash_attention_output=merge_flash_attention_output,
        KimiOProjMMReduceScatterOp=type("ProjectionOp", (), {}),
        flash_attention_output=flash_attention_output,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method.name]


@pytest.mark.parametrize("history_lengths", [(0, 0), (127, 129), (2049, 4097)])
@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("query_case", ["random", "zero_latent"])
@torch.inference_mode()
def test_c8_history_and_bf16_current_module(history_lengths, is_prefill, query_case="random"):
    from flash_mla_c8_qualification import C8Case

    torch.manual_seed(891)
    qlen, heads, batch = (17 if is_prefill else 4), 12, len(history_lengths)
    case = C8Case([n + qlen for n in history_lengths], heads, qlen=qlen)
    # Exercise the same mixed-dtype page geometry allocated by both runners.
    # The 640-byte width is capacity; each page stores all latent rows first,
    # then all rope rows, so both inner layouts stay contiguous.
    storage = torch.zeros(case.k.shape[0], 3, 128, 640, dtype=torch.uint8, device="npu")
    latent_cache, rope_cache = split_flash_mla_c8_cache(storage[:, 1].view(torch.float8_e4m3fn))
    latent_cache.copy_(case.k)
    rope_cache.copy_(case.kr)
    case.k, case.kr = latent_cache, rope_cache
    total = batch * qlen
    query = (torch.randn(total, heads, 576, device="npu") * 0.25).to(torch.bfloat16)

    def set_query_boundary():
        if query_case == "zero_latent":
            query[0].zero_()
            query[1, :, :512].zero_()

    set_query_boundary()
    current = (torch.randn(total, 1, 576, device="npu") * 0.25).to(torch.bfloat16)
    slots = torch.empty(total, dtype=torch.int64)
    for request, length in enumerate(history_lengths):
        for i in range(qlen):
            at = length + i
            slots[request * qlen + i] = case.blocks_cpu[request, at // 128] * 128 + at % 128
    lengths = torch.tensor(
        history_lengths if is_prefill else [n + qlen for n in history_lengths], dtype=torch.int32, device="npu"
    )
    current_pages_per_request = (qlen + 127) // 128
    current_blocks = torch.arange(batch * current_pages_per_request, dtype=torch.int32).view(batch, -1).to("npu")
    current_slots = torch.tensor(
        [b * current_pages_per_request * 128 + i for b in range(batch) for i in range(qlen)],
        dtype=torch.int64,
        device="npu",
    )
    current_cache = torch.empty(batch * current_pages_per_request, 1, 128, 576, device="npu", dtype=torch.bfloat16)

    def metadata(seq_lens, c8, current=False):
        return torch.ops._C_ascend.flash_mla_with_kvcache_metadata(
            seq_lens,
            heads,
            1,
            cu_seqlens_q=case.cu,
            seqused_q=case.used,
            max_seqlen_q=qlen,
            max_seqlen_kv=qlen if current else max(case.lengths_cpu),
            head_dim_qk=576,
            head_dim_v=512,
            mask_mode=3 if current or not is_prefill else 0,
            layout_q="TND",
            is_c8=c8,
        )

    b = SimpleNamespace(
        query=torch.empty_like(query),
        split_kv=is_prefill,
        unabsorbed=False,
        dcp_size=1,
        is_prefill=is_prefill,
        causal=True,
        schedule=metadata(lengths, True),
        current_schedule=metadata(case.used, False, current=True),
        current_cache=current_cache,
        current_block_table=current_blocks,
        current_slots=current_slots,
        slots=slots.to("npu"),
        block_table=case.blocks,
        cache_lens=lengths,
        used_q=case.used,
        cu=case.cu,
        max_query_len=qlen,
        max_seq_len=max(case.lengths_cpu),
        token_live=torch.ones(total, dtype=torch.bool, device="npu"),
        attn_mask=case.mask,
    )
    impl = SimpleNamespace(
        enable_mlapo=False,
        flash_dcp_overlap=False,
        fused_qkv_a_proj=lambda x: (torch.cat((current.new_zeros(total, 1), current[:, 0]), -1), None),
        q_lora_rank=1,
        q_a_layernorm=lambda x: x,
        kv_a_layernorm=lambda x: x,
        _q_proj_and_k_up_proj=lambda x: (query[..., :512], query[..., 512:]),
        q_proj=SimpleNamespace(qrep_active=False),
        use_mla_rope=False,
        layerwise_kv_cache_hook=None,
        fak_descale_float=case.sk,
        fak_descale_reciprocal=case.sk.reciprocal(),
        scale=case.scale,
        _v_up_proj_batch_major=lambda x: x.flatten(1),
        _v_up_proj=lambda x: x.transpose(0, 1).reshape(total, -1),
        use_output_gate=False,
        o_proj=lambda x, **kwargs: (x, None),
    )
    output = torch.empty(total, heads * 512, dtype=torch.bfloat16, device="npu")
    hidden = torch.empty(total, 1, dtype=torch.bfloat16, device="npu")
    forward = load_c8_forward()

    def run():
        return forward(impl, "layer", hidden, (case.k, case.kr), SimpleNamespace(flash=b), output)

    def check():
        q8, sq = torch_npu.npu_dynamic_quant(query[..., :512].contiguous(), dst_type=torch.float8_e4m3fn)
        # Independent CPU reference chooses the mathematically equivalent
        # scale=1 for exactly-zero Q; it does not invoke the device scale op.
        sq = sq.float().cpu()
        sq[sq == 0] = 1
        sk = case.sk.float().cpu()
        qr = (query[..., 512:].float().cpu() / (sq[..., None] * sk)).to(torch.bfloat16)
        qh = torch.cat((q8.float().cpu() * sq[..., None], qr.float() * sq[..., None] * sk), -1)
        raw_q, new_kv = query.float().cpu(), current.float().cpu()
        key = torch.cat((case.k.float() * case.sk, case.kr.float()), -1).squeeze(2).cpu()
        expected = torch.empty(total, heads, 512)
        for request, history in enumerate(history_lengths):
            for i in range(qlen):
                row = request * qlen + i
                visible = history if is_prefill else history + i + 1
                positions = torch.arange(visible)
                physical = case.blocks_cpu[request, positions // 128]
                kv = key[physical, positions % 128]
                logits = qh[row] @ kv.T * case.scale
                values = kv[:, :512]
                if is_prefill:
                    current_kv = new_kv[request * qlen : request * qlen + i + 1, 0]
                    current_logits = raw_q[row] @ current_kv.T * case.scale
                    logits = torch.cat((logits, current_logits), -1)
                    values = torch.cat((values, current_kv[:, :512]), 0)
                expected[row] = torch.softmax(logits, -1) @ values
        actual = output.cpu().view(total, heads, 512).float()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.03)
        return float((actual - expected).abs().max())

    run()
    error = check()
    if not is_prefill:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        query.normal_(std=0.25)
        set_query_boundary()
        current.normal_(std=0.25)
        graph.replay()
        error = max(error, check())
    print(
        {"history_lengths": history_lengths, "prefill": is_prefill, "query_case": query_case, "max_abs": error},
        flush=True,
    )
