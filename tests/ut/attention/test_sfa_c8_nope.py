# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm_ascend.attention.context_parallel.sfa_cp as sfa_cp
import vllm_ascend.attention.sfa_v1 as sfa
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.device.device_op import BaseDeviceAdaptor
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


def _make_impl(device_type=AscendDeviceType.A2, enable_c8=True):
    config = SimpleNamespace(kv_transfer_config=None, model_config=SimpleNamespace(hf_config=SimpleNamespace()))
    ascend_config = SimpleNamespace(enable_sparse_sfa_c8=enable_c8, enable_mlapo=False)
    with (
        patch.object(sfa, "get_current_vllm_config", return_value=config),
        patch.object(sfa, "get_ascend_config", return_value=ascend_config),
        patch.object(sfa, "get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        patch.object(sfa, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(sfa, "enable_sp", return_value=False),
    ):
        return AscendSFAImpl(
            num_heads=16,
            head_size=512,
            scale=1 / 24,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            q_lora_rank=512,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=0,
            qk_head_dim=128,
            v_head_dim=128,
            q_b_proj=None,
            kv_b_proj=None,
            o_proj=None,
            indexer=None,
            skip_topk=True,
            topk_indices_buffer=torch.zeros(4, 128, dtype=torch.int32),
            kv_a_layernorm=SimpleNamespace(weight=torch.ones(512), variance_epsilon=1e-6),
        )


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3])
def test_c8_nope_init_selects_int8_cache(device_type):
    impl = _make_impl(device_type)
    assert impl.c8_k_cache_dtype == torch.int8
    assert impl.sfa_qsfa_packed_kv_head_dim == 528
    assert not impl.supports_dense_mha_prefill


@pytest.mark.parametrize("device_type", [AscendDeviceType.A5, AscendDeviceType._310P])
def test_c8_nope_init_rejects_other_hardware(device_type):
    with pytest.raises(NotImplementedError, match="only on A2/A3"):
        _make_impl(device_type)


def test_floating_nope_remains_available_on_a5():
    assert not _make_impl(AscendDeviceType.A5, enable_c8=False).supports_dense_mha_prefill


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_c8_nope_preprocess_skips_rotation_and_keeps_fp32_scale_bytes(dtype):
    impl = _make_impl()
    tokens = 4
    kv = torch.randn(tokens, 512, dtype=dtype)
    quantized = torch.arange(tokens * 512, dtype=torch.int32).remainder(255).sub(127).to(torch.int8)
    quantized = quantized.reshape(tokens, 1, 512)
    scales = torch.linspace(0.01, 0.16, tokens * 4, dtype=torch.float32).reshape(tokens, 1, 4)
    cache = torch.zeros(2, 2, 1, 528, dtype=torch.int8)
    with (
        patch.object(sfa.torch_npu, "npu_rms_norm", return_value=(kv.view(tokens, 1, 1, 512), None), create=True),
        patch.object(sfa.torch_npu, "npu_dynamic_block_quant", return_value=(quantized, scales), create=True) as quant,
        patch.object(sfa.torch_npu, "npu_interleave_rope", create=True) as rotate,
    ):
        k_pe, k_nope, scale_bytes = impl.exec_kv(kv, None, None, (cache,), torch.arange(tokens), None)

    rotate.assert_not_called()
    assert quant.call_args.kwargs == {"dst_type": torch.int8, "row_block_size": 1, "col_block_size": 128}
    assert quant.call_args.args[0].shape == (tokens, 1, 512)
    assert k_pe.shape == (tokens, 1, 1, 0)
    assert k_pe.dtype == torch.int8
    assert scale_bytes.shape == (tokens, 1, 1, 16)
    torch.testing.assert_close(k_nope.reshape(tokens, 512), quantized.reshape(tokens, 512))
    torch.testing.assert_close(scale_bytes.view(torch.float32).reshape(tokens, 4), scales.reshape(tokens, 4))
    assert not cache.any(), "Preprocessing must return the packed components before the cache-write stage."

    initial_cache = torch.randint(-128, 127, cache.shape, dtype=torch.int8)
    cache.copy_(initial_cache)
    slots = torch.tensor([0, 3, -1, 99], dtype=torch.int64)
    impl._store_parallel_kv(k_pe, k_nope, scale_bytes, None, [], (cache,), slots, None, False)
    expected = initial_cache.clone().view(-1, 528)
    expected[slots[:2], :512] = quantized[:2].reshape(2, 512)
    expected[slots[:2], 512:] = scales[:2].reshape(2, 4).contiguous().view(torch.int8)
    torch.testing.assert_close(cache.view(-1, 528), expected)


def test_c8_nope_dispatch_uses_custom_qsfa_and_masks_graph_padding():
    impl = _make_impl()
    query = torch.randn(3, 16, 512, dtype=torch.bfloat16)
    query_rope = query.new_empty(3, 16, 0)
    cache = torch.zeros(2, 256, 1, 528, dtype=torch.int8)
    indices = torch.zeros(3, 1, 128, dtype=torch.int32)
    query_ends = torch.tensor([2], dtype=torch.int32)
    kv_lengths = torch.tensor([256], dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    metadata = SimpleNamespace(block_table=block_table, block_size=128)
    expected = torch.randn_like(query)
    expected[2] = torch.nan
    with (
        patch.object(sfa, "DeviceOperator", BaseDeviceAdaptor),
        patch.object(sfa, "sparse_mla", side_effect=AssertionError("C8 must use the packed-KV operator")),
        patch.object(
            torch.ops._C_ascend,
            "npu_kv_quant_sparse_flash_attention",
            return_value=(expected, torch.empty(0), torch.empty(0)),
            create=True,
        ) as kernel,
    ):
        output = impl._execute_sparse_flash_attention_process(
            query, query_rope, (cache,), indices, metadata, query_ends, kv_lengths
        )
    kwargs = kernel.call_args.kwargs
    assert kwargs["rope_head_dim"] == 0
    assert kwargs["quant_scale_repo_mode"] == 1
    assert kwargs["key_quant_mode"] == kwargs["value_quant_mode"] == 2
    assert kwargs["query"].shape == (3, 16, 512)
    assert kwargs["key"].shape == (4, 128, 1, 528)
    assert kwargs["key"].data_ptr() == cache.data_ptr()
    assert kwargs["value"] is kwargs["key"]
    assert kwargs["block_table"] is block_table
    torch.testing.assert_close(output[:2], expected[:2])
    torch.testing.assert_close(output[2], torch.zeros_like(output[2]))


def test_c8_nope_lse_dispatch_preserves_three_outputs():
    impl = _make_impl()
    query = torch.randn(2, 16, 512, dtype=torch.float16)
    empty_rope = query.new_empty(2, 16, 0)
    cache = torch.zeros(1, 128, 1, 528, dtype=torch.int8)
    metadata = SimpleNamespace(block_table=torch.zeros(1, 1, dtype=torch.int32))
    lengths = torch.tensor([2], dtype=torch.int32)
    expected = (torch.randn_like(query), torch.randn(1, 2, 16), torch.rand(1, 2, 16))
    with patch.object(
        torch.ops._C_ascend, "npu_kv_quant_sparse_flash_attention", return_value=expected, create=True
    ) as kernel:
        actual = BaseDeviceAdaptor.execute_sparse_flash_attention_process(
            impl,
            query,
            empty_rope,
            (cache,),
            torch.zeros(2, 1, 128, dtype=torch.int32),
            metadata,
            lengths,
            lengths,
            sparse_mode=0,
            return_lse=True,
        )
    assert actual is expected
    assert kernel.call_args.kwargs["return_softmax_lse"] is True
    assert kernel.call_args.kwargs["rope_head_dim"] == 0
    assert kernel.call_args.kwargs["sparse_mode"] == 0


def test_c8_nope_dsa_cp_gather_accepts_empty_rope():
    impl = _make_impl()
    k_nope = torch.ones(2, 1, 1, 512, dtype=torch.int8)
    k_pe = torch.empty(2, 1, 1, 0, dtype=torch.int8)
    scale_bytes = torch.arange(32, dtype=torch.int8).reshape(2, 1, 1, 16)
    with (
        patch("vllm_ascend.attention.context_parallel.sfa_cp.get_tp_group", return_value=None),
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.all_gather_async", side_effect=lambda x, *a, **kw: (x, None)
        ),
    ):
        packed, handles = AscendSFADSACPImpl._prepare_kv_for_parallel(impl, k_pe, k_nope, scale_bytes, False)
    assert packed.shape == (2, 528)
    assert handles == []
    torch.testing.assert_close(packed[:, :512], k_nope.reshape(2, 512))
    torch.testing.assert_close(packed[:, 512:], scale_bytes.reshape(2, 16))


@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("num_decode_tokens", [0, 1, 2])
def test_pcp_cache_gather_accepts_nope_and_preserves_rope(rope_dim, num_decode_tokens):
    impl = sfa_cp.AscendSFAPCPImpl.__new__(sfa_cp.AscendSFAPCPImpl)
    impl.qk_rope_head_dim = rope_dim
    impl.enable_sparse_sfa_c8 = False
    kv = torch.arange(2 * (512 + rope_dim), dtype=torch.float32).reshape(2, -1)
    cos = torch.ones(2, 1, rope_dim) if rope_dim else None
    sin = torch.zeros(2, 1, rope_dim) if rope_dim else None
    slots = torch.tensor([10, 11, 20, 21], dtype=torch.int64)
    metadata = SimpleNamespace(num_decode_tokens=num_decode_tokens)
    cache = (torch.empty(1, 128, 1, 528 + 2 * rope_dim, dtype=torch.int8),)
    group = SimpleNamespace(world_size=2, all_gather=lambda tensor, dim: torch.cat((tensor, tensor + 100), dim=dim))
    real_gather = sfa_cp._gather_prefill_cache_inputs
    with (
        patch.dict(real_gather.__globals__, {"get_pcp_group": lambda: group}),
        patch.object(sfa_cp, "_gather_prefill_cache_inputs", wraps=real_gather) as gather,
        patch.object(AscendSFAImpl, "exec_kv", autospec=True, return_value="written") as base_exec,
    ):
        result = impl.exec_kv(kv, cos, sin, cache, slots, metadata)

    assert result == "written"
    gather_inputs = gather.call_args.args[0]
    assert len(gather_inputs) == (3 if rope_dim else 1)
    assert all(isinstance(tensor, torch.Tensor) for tensor in gather_inputs)
    expected_slots = (
        slots[:num_decode_tokens]
        if num_decode_tokens == 2
        else torch.cat((slots[:num_decode_tokens], slots.reshape(2, 2)[:, num_decode_tokens:].flatten()))
    )
    expected_kv = torch.cat((kv, kv[num_decode_tokens:] + 100))
    args = base_exec.call_args.args
    torch.testing.assert_close(args[1], expected_kv)
    torch.testing.assert_close(args[5], expected_slots)
    if rope_dim:
        torch.testing.assert_close(args[2], torch.cat((cos, cos[num_decode_tokens:] + 100)))
        torch.testing.assert_close(args[3], torch.cat((sin, sin[num_decode_tokens:] + 100)))
    else:
        assert args[2] is args[3] is None


@pytest.mark.parametrize("with_dcp", [False, True])
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("num_decode_tokens", [0, 1, 2])
def test_pcp_c8_forward_stores_gathered_rows_in_matching_slots(with_dcp, rope_dim, num_decode_tokens):
    impl_cls = sfa_cp.AscendSFAPCPDCPImpl if with_dcp else sfa_cp.AscendSFAPCPImpl
    impl = impl_cls.__new__(impl_cls)
    impl.__dict__.update(_make_impl().__dict__)
    impl.qk_rope_head_dim = rope_dim
    impl.sfa_qsfa_packed_kv_head_dim = 528 + 2 * rope_dim
    impl._o_proj_weight_switch_enabled = False
    impl.fused_qkv_a_proj = lambda hidden: (torch.cat((hidden.new_zeros(2, 512), hidden), dim=-1),)
    impl.q_a_layernorm = lambda query: query
    impl._q_proj_and_k_up_proj = lambda query: (query.new_zeros(2, 16, 512), query.new_zeros(2, 16, rope_dim))
    impl._record_query_gather_context = lambda *args: None
    impl._record_dcp_kv_gather_context = lambda *args: None
    impl._execute_sparse_flash_attention_process = lambda query, *args: query
    impl._v_up_proj = lambda values: values[:, 0, :]
    impl.o_proj = lambda values: (values,)

    hidden = torch.stack((torch.ones(512 + rope_dim), torch.full((512 + rope_dim,), 2.0))).to(torch.float16)
    cos = torch.ones(2, 1, 1, rope_dim) if rope_dim else None
    sin = torch.zeros(2, 1, 1, rope_dim) if rope_dim else None
    slots = torch.tensor([10, 11, 20, 21], dtype=torch.int64)
    slots[2 : 2 + num_decode_tokens] = slots[:num_decode_tokens]
    metadata = sfa_cp.AscendSFADCPMetadata.__new__(sfa_cp.AscendSFADCPMetadata)
    metadata.__dict__.update(
        num_input_tokens=2,
        num_actual_tokens=2,
        num_decode_tokens=num_decode_tokens,
        num_prefills=int(num_decode_tokens < 2),
        cos=cos,
        sin=sin,
        slot_mapping=slots[:2],
        pcp_slot_mapping=slots,
        cum_query_lens=torch.tensor([1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([1, 2], dtype=torch.int32),
        attn_state=sfa.AscendAttentionState.DecodeOnly,
        dcp_context=SimpleNamespace(slot_mapping=slots),
    )
    cache = torch.zeros(2, 16, 1, impl.sfa_qsfa_packed_kv_head_dim, dtype=torch.int8)
    group = SimpleNamespace(world_size=2, all_gather=lambda tensor, dim: torch.cat((tensor, tensor + 100), dim=dim))

    def scatter(cache, indices, updates):
        assert indices.shape[0] == updates.shape[0]
        cache[indices.flatten()] = updates
        return cache

    backend = SimpleNamespace(
        npu_rms_norm=lambda values, gamma, epsilon: (values, None),
        npu_dynamic_block_quant=lambda values, **kwargs: (
            values.to(torch.int8),
            torch.ones((*values.shape[:-1], 4), dtype=torch.float32),
        ),
        npu_interleave_rope=lambda rope, cos, sin: rope,
        npu_scatter_nd_update_=scatter,
    )
    real_gather = sfa_cp._gather_prefill_cache_inputs
    with (
        patch.dict(real_gather.__globals__, {"get_pcp_group": lambda: group}),
        patch.object(sfa, "torch_npu", backend),
        patch.object(
            sfa, "get_forward_context", return_value=SimpleNamespace(cudagraph_runtime_mode=sfa.CUDAGraphMode.NONE)
        ),
        patch.object(sfa, "record_attention_compute_start"),
        patch.object(sfa, "wait_for_kv_layer_from_connector"),
        patch.object(sfa, "notify_kv_cache_written"),
        patch.object(sfa, "maybe_save_kv_layer_to_connector"),
    ):
        impl.forward("test.pcp", hidden, (cache,), metadata, output=hidden.new_empty(2, 512))

    expected_slots = torch.cat((slots[:2], slots[2 + num_decode_tokens :]))
    expected_values = torch.cat((hidden, hidden[num_decode_tokens:] + 100))
    expected_packed = torch.cat(
        (
            expected_values[:, :512].to(torch.int8),
            expected_values[:, 512:].contiguous().view(torch.int8),
            torch.ones(expected_values.shape[0], 4, dtype=torch.float32).view(torch.int8),
        ),
        dim=-1,
    )
    expected_cache = torch.zeros_like(cache).view(-1, impl.sfa_qsfa_packed_kv_head_dim)
    expected_cache[expected_slots] = expected_packed
    torch.testing.assert_close(cache.view_as(expected_cache), expected_cache)
