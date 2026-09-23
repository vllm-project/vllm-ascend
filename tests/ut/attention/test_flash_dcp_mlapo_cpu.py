# SPDX-License-Identifier: Apache-2.0
"""Exercise production FlashMLA orchestration with CPU cache/operator fakes."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_forward(scope):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_flash")
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method.name]


@pytest.mark.parametrize("dcp", [1, 8])
@pytest.mark.parametrize("quantized", [False, True])
def test_prolog_writes_replicated_current_slots_for_dcp(dcp, quantized):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAImpl")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "mla_preprocess_only_decode")
    tokens, heads = 4, 12 * dcp
    history = torch.empty(4, 2, 128, 576, dtype=torch.bfloat16)[:, 1]
    if quantized:
        history = (
            torch.empty(4, 2, 128, 1, 512, dtype=torch.float8_e4m3fn)[:, 1],
            torch.empty(4, 2, 128, 1, 64, dtype=torch.bfloat16)[:, 1],
        )
    current = torch.empty(3, 1, 128, 576, dtype=torch.bfloat16)
    owned_slots = torch.tensor([-1, 128, -1, -1])
    current_slots = torch.tensor([127, 128, 129, -1])
    metadata = SimpleNamespace(
        flash=SimpleNamespace(
            dcp_size=dcp,
            query=torch.empty(tokens, heads, 576),
            slots=owned_slots,
            current_slots=current_slots,
            current_cache=current,
        )
    )

    def prolog(**kwargs):
        c8_prolog = quantized and dcp == 1
        assert kwargs["kv_cache_quant_mode"] == kwargs["query_quant_mode"] == int(c8_prolog)
        assert (kwargs["quant_scale_ckv"] is not None) == c8_prolog
        if c8_prolog:
            assert kwargs["kv_cache"] is history[0]
            assert kwargs["kr_cache"] is history[1]
        else:
            cache = current.squeeze(1) if dcp > 1 else history
            assert kwargs["kv_cache"].dtype == torch.bfloat16
            assert kwargs["kv_cache"].data_ptr() == cache.data_ptr()
            assert kwargs["kr_cache"].data_ptr() == cache[..., 512:].data_ptr()
            assert kwargs["kv_cache"].stride() == cache[..., :512].unsqueeze(2).stride()
        torch.testing.assert_close(kwargs["cache_index"].reshape(-1), current_slots if dcp > 1 else owned_slots)
        scale = torch.ones(tokens, heads, 1) if c8_prolog else None
        return torch.empty(tokens, heads, 512), torch.empty(tokens, heads, 64), scale, None, None

    scope = dict(
        torch=torch,
        torch_npu=SimpleNamespace(
            npu_dynamic_mx_quant=lambda x, **kwargs: (
                x.to(torch.float8_e4m3fn),
                torch.empty(tokens, 1, 112, 2, dtype=torch.uint8),
            )
        ),
        get_dynamic_mx_quant_scale_alg=lambda _: 0,
        envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        _npu_mla_prolog_v3_no_rope=prolog,
        DecodeMLAPreprocessResult=lambda q, r, k, v, **kwargs: SimpleNamespace(ql_nope=q, q_pe=r),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    norm = SimpleNamespace(weight=SimpleNamespace(data=None), variance_epsilon=1e-6)
    impl = SimpleNamespace(
        kv_lora_rank=512,
        support_fp8_attention=True,
        mlapo_weight_quant_mode=3,
        flash_fused_dcp_prolog=True,
        vllm_config=None,
        dequant_scale_w_dq=torch.empty(1, dtype=torch.uint8),
        dequant_scale_w_uq_qr=torch.empty(1, dtype=torch.uint8),
        dequant_scale_w_dkv_kr=torch.empty(1, dtype=torch.uint8),
        fak_descale_reciprocal=torch.tensor([32.0]),
        use_mla_rope=False,
        _mlapo_empty_rope=torch.empty(0, 64),
        fa_quant_layer=quantized,
        weight_dq=None,
        weight_uq_qr=None,
        mlapo_W_UK_T=None,
        weight_dkv_kr=None,
        q_a_layernorm=norm,
        kv_a_layernorm=norm,
        mlapo_num_heads=heads,
        reorg_decode_q=lambda q, r: (q, r),
    )
    result, _ = scope[method.name](impl, torch.empty(tokens, 7168), history, metadata)
    assert result.ql_nope.shape == (tokens, heads, 512)


@pytest.mark.parametrize("tokens", [4, 16])
@pytest.mark.parametrize("rank", [0, 3, 7])
@pytest.mark.parametrize("mlapo", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_replicated_prolog_current_and_owner_cache(monkeypatch, tokens, rank, mlapo, overlap):
    events = []
    heads, dcp, width = 12, 8, 576
    torch.manual_seed(42)
    current = torch.randn(tokens, 1, width)
    queries = torch.randn(tokens, heads * dcp, width)
    history_storage = torch.full((4, 2, 128, width), 42.0)
    history = history_storage[:, 1]
    current_cache = torch.full((4, 1, 128, width), 19.0)
    src_slots = torch.arange(tokens) + 127
    src_slots[-1] = -1
    dst_slots = torch.arange(tokens) + 255
    dst_slots[torch.arange(tokens) % dcp != rank] = -1
    dst_slots[-1] = -1
    initial_history = history_storage.clone()

    class Projection:
        qrep_active = True

        def _local_view(self, value):
            return value[:, rank * heads : (rank + 1) * heads].contiguous()

    class Stream:
        def record_event(self):
            event = len(events)
            events.append(("record", event))
            return event

        def wait_event(self, event):
            events.append(("wait", event))

    main, side = Stream(), Stream()
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: main, stream=lambda _: nullcontext()), raising=False
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)

    def scatter(key, value, key_cache, value_cache, slot_mapping, **kwargs):
        target = "current_scatter" if key_cache.data_ptr() == current_cache.data_ptr() else "history_scatter"
        events.append((target,))
        for token, slot in enumerate(slot_mapping.tolist()):
            if slot >= 0:
                key_cache[slot // 128, slot % 128] = key[token]
                value_cache[slot // 128, slot % 128] = value[token]

    def preprocess(x, cache, metadata):
        assert cache is history
        events.append(("prolog",))
        for token, slot in enumerate(src_slots.tolist()):
            if slot >= 0:
                current_cache[slot // 128, 0, slot % 128] = current[token, 0]
        return SimpleNamespace(ql_nope=queries[..., :512], q_pe=queries[..., 512:]), None

    def copy(source, destination, source_slots, destination_slots):
        assert source.data_ptr() == current_cache.data_ptr()
        assert destination is history
        events.append(("copy_history",))
        for src, dst in zip(source_slots.tolist(), destination_slots.tolist()):
            if src >= 0 and dst >= 0:
                destination[dst // 128, dst % 128] = source[src // 128, src % 128]

    def flash(query, cache, **kwargs):
        is_current = cache.data_ptr() == current_cache.data_ptr()
        events.append(("current_flash" if is_current else "history_flash",))
        assert kwargs["mask_mode"] == (3 if is_current else 0)
        expected = initial_history.clone()
        for token, slot in enumerate(dst_slots.tolist()):
            if slot >= 0:
                expected[slot // 128, 1, slot % 128] = current[token, 0]
        torch.testing.assert_close(history_storage, expected)
        if is_current:
            for token, slot in enumerate(src_slots.tolist()):
                if slot >= 0:
                    torch.testing.assert_close(cache[slot // 128, 0, slot % 128], current[token, 0])
            torch.testing.assert_close(query, Projection()._local_view(queries))
        else:
            torch.testing.assert_close(query, queries)
        return torch.zeros(tokens, query.shape[1], 512), torch.zeros(query.shape[1], tokens)

    def exchange(*args, **kwargs):
        events.append(("pack",))
        return torch.empty(1)

    scope = dict(
        torch=torch,
        torch_npu=SimpleNamespace(npu_scatter_pa_kv_cache=scatter),
        wait_for_device_metadata=lambda *args: None,
        DeviceMetadataStage=SimpleNamespace(ATTENTION=0),
        MLAPO_MAX_SUPPORTED_TOKENS=128,
        FLASH_DCP_QUERY_PREP_MAX_TOKENS=128,
        _dcp_mtp_comm_stream=lambda: side,
        copy_mla_kv=copy,
        notify_kv_cache_written=lambda *args: events.append(("notify",)),
        exchange_flash_attention_output=exchange,
        combine_flash_attention_output=lambda *args, **kwargs: torch.ones(tokens, heads, 512),
        merge_flash_attention_output=lambda *args, **kwargs: torch.ones(tokens, heads, 512),
        KimiOProjMMReduceScatterOp=type("ProjectionOp", (), {}),
        flash_attention_output=lambda result, live, output: output.copy_(result),
    )
    monkeypatch.setattr(torch.ops._C_ascend, "flash_mla_with_kvcache", flash, raising=False)
    impl = SimpleNamespace(
        enable_mlapo=mlapo,
        fa_quant_layer=False,
        q_proj=Projection(),
        layerwise_kv_cache_hook=None,
        mla_preprocess_only_decode=preprocess,
        fused_qkv_a_proj=lambda x: (torch.cat((torch.zeros(tokens, 1), current[:, 0]), -1), None),
        q_lora_rank=1,
        q_a_layernorm=lambda x: x,
        kv_a_layernorm=lambda x: x,
        flash_fused_dcp_prolog=True,
        flash_dcp_overlap=overlap,
        flash_dcp_preprocess_overlap=True,
        use_mla_rope=False,
        _q_proj_and_k_up_proj=lambda x: (queries[..., :512], queries[..., 512:]),
        scale=0.1,
        dcp_group=SimpleNamespace(),
        _v_up_proj_batch_major=lambda x: x.flatten(1),
        use_output_gate=False,
        o_proj=lambda x, **kwargs: (x, None),
    )
    metadata = SimpleNamespace(
        causal=True,
        flash=SimpleNamespace(
            query=torch.empty_like(queries),
            is_prefill=False,
            dcp_size=dcp,
            split_kv=True,
            unabsorbed=False,
            current_cache=current_cache,
            current_slots=src_slots,
            slots=dst_slots,
            schedule=None,
            current_schedule=None,
            block_table=None,
            current_block_table=None,
            cache_lens=None,
            used_q=None,
            cu=None,
            attn_mask=None,
            max_query_len=tokens,
            max_seq_len=512,
            token_live=src_slots >= 0,
        ),
    )
    output = torch.empty(tokens, heads * 512)
    load_forward(scope)(impl, "attention", torch.empty(tokens, 3), history, metadata, output)
    names = [event[0] for event in events]
    if mlapo:
        assert names.count("prolog") == names.count("copy_history") == 1
        assert "current_scatter" not in names and "history_scatter" not in names
        assert names.index("copy_history") < names.index("notify") < names.index("history_flash")
        if overlap:
            assert names.index("history_flash") < names.index("pack") < names.index("current_flash")
    else:
        assert names.count("current_scatter") == names.count("history_scatter") == 1
        if overlap:
            assert names.index("history_flash") < names.index("pack")
            assert names.index("pack") < names.index("current_scatter") < names.index("current_flash")
    assert torch.all(output == 1)
