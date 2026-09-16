# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm_ascend.attention.sfa_v1 as sfa
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.device.device_op import BaseDeviceAdaptor
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


def _make_impl(device_type=AscendDeviceType.A2, enable_c8=True, pcp=1, dcp=1, dsa_cp=False):
    config = SimpleNamespace(kv_transfer_config=None, model_config=SimpleNamespace(hf_config=SimpleNamespace()))
    config.parallel_config = SimpleNamespace(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp)
    ascend_config = SimpleNamespace(enable_sparse_sfa_c8=enable_c8, enable_mlapo=False)
    with (
        patch.object(sfa, "get_current_vllm_config", return_value=config),
        patch.object(sfa, "get_ascend_config", return_value=ascend_config),
        patch.object(sfa, "get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        patch.object(sfa, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(sfa, "enable_sp", return_value=False),
        patch.object(sfa, "enable_dsa_cp", return_value=dsa_cp),
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
def test_c8_nope_init_rejects_unintegrated_model_adaptors(device_type):
    with pytest.raises(NotImplementedError, match="model integration is not enabled for this device adaptor"):
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


@pytest.mark.parametrize("pcp,dcp,dsa_cp", [(2, 1, False), (1, 2, False), (1, 1, True)])
def test_c8_nope_keeps_context_parallelism_out_of_scope(pcp, dcp, dsa_cp):
    with pytest.raises(NotImplementedError, match="does not yet support context parallelism"):
        _make_impl(pcp=pcp, dcp=dcp, dsa_cp=dsa_cp)
