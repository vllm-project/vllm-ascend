import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@pytest.mark.parametrize("enable_rope", [True, False])
@pytest.mark.parametrize("token_num", [1, 6, 128])
@torch.inference_mode()
def test_mla_preprocess_kernel(enable_rope, token_num, monkeypatch):
    monkeypatch.setattr(torch.npu.config, "allow_internal_format", True)
    _exercise_no_quant_case(enable_rope, token_num)


def _exercise_no_quant_case(enable_rope, token_num):
    torch.manual_seed(0)
    # Retain the original no_quant test's dimensions. N=6 exercises several
    # destinations; N=128 also exercises per-core loops and local-buffer reuse.
    head_num, hidden_size, q_lora_rank = 2, 7168, 1536
    kv_lora_rank, rope_dim, qk_nope_head_dim = 512, 64, 128
    block_size, dtype = 128, torch.bfloat16
    mm1_out = q_lora_rank + kv_lora_rank + rope_dim
    mm2_out = head_num * (qk_nope_head_dim + rope_dim)
    hidden = torch.randn(token_num, hidden_size, dtype=dtype, device="npu") * 0.1
    wdqkv = torch.randn(1, hidden_size // 16, mm1_out, 16, dtype=dtype, device="npu") * 0.01
    wuq = torch.randn(1, q_lora_rank // 16, mm2_out, 16, dtype=dtype, device="npu") * 0.01
    wdqkv = torch_npu.npu_format_cast(wdqkv.contiguous(), 29)
    wuq = torch_npu.npu_format_cast(wuq.contiguous(), 29)
    cos = torch.rand(token_num, 1, 1, rope_dim, dtype=dtype, device="npu")
    sin = torch.rand_like(cos)
    inputs = (
        hidden,
        wdqkv,
        None,
        torch.ones(q_lora_rank, dtype=dtype, device="npu"),
        None,
        wuq,
        None,
        torch.ones(kv_lora_rank, dtype=dtype, device="npu"),
        cos.view(token_num, rope_dim) if enable_rope else None,
        sin.view(token_num, rope_dim) if enable_rope else None,
        # Original no_quant dispatch uses ND W_UK and ND krope_ctkv caches.
        torch.randn(head_num, qk_nope_head_dim, kv_lora_rank, dtype=dtype, device="npu") * 0.01,
    )
    # The schema accepts None for these tensors but has no default values.
    quant_args = {
        name: None
        for name in (
            "quant_scale0",
            "quant_offset0",
            "bias0",
            "quant_scale1",
            "quant_offset1",
            "bias1",
            "ctkv_scale",
            "q_nope_scale",
        )
    }
    caches = tuple(
        torch.full((1, block_size, width), float("nan"), dtype=dtype, device="npu")
        for width in (kv_lora_rank, rope_dim)
    )
    compact_slots = torch.arange(token_num, dtype=torch.int32, device="npu")
    q_outputs = tuple(
        torch.full(shape, float("nan"), dtype=dtype, device="npu")
        for shape in ((token_num, head_num, kv_lora_rank), (token_num, head_num, rope_dim), (token_num, q_lora_rank))
    )
    reference = (*q_outputs, caches[0].view(-1, kv_lora_rank)[:token_num], caches[1].view(-1, rope_dim)[:token_num])

    def original():
        torch.ops._C_ascend.mla_preprocess(
            *inputs,
            *caches,
            compact_slots,
            **quant_args,
            cache_mode="krope_ctkv",
            quant_mode="no_quant",
            enable_inner_out=False,
            q_out0=q_outputs[0],
            kv_cache_out0=caches[0],
            q_out1=q_outputs[1],
            kv_cache_out1=caches[1],
            inner_out=q_outputs[2],
        )

    # Original no_quant implements key 281 (no inner output), not key 793.
    original()
    torch.npu.synchronize()
    for index in (0, 1, 3, 4):
        assert torch.isfinite(reference[index]).all(), f"original no_quant output {index} was not fully written"
    assert torch.isnan(q_outputs[2]).all(), "disabled inner output must remain untouched"
    for cache in caches:
        assert torch.isnan(cache[:, token_num:]).all(), "unused cache rows must remain untouched"

    outputs = tuple(torch.full_like(value, float("nan")) for value in reference)
    # Direct currently requires quantized weights and inner output; no_quant
    # has no Direct key. Meta shape inference does not run this Host check.
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.mla_preprocess_with_kv_offload(
            *inputs,
            **quant_args,
            quant_mode="no_quant",
            q_out0=outputs[0],
            q_out1=outputs[1],
            inner_out=outputs[2],
            current_k=outputs[3],
            current_rope=outputs[4],
        )
