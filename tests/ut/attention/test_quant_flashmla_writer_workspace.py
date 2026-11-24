# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.ops  # noqa: F401
from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAImpl

# These are input configurations, not production alignment constants.
MANAGER_BLOCK_SIZES = (16, 32, 128, 6144)


def _impl(manager_block_size, capacity, quant=True):
    return SimpleNamespace(
        use_quant_flashmla=quant,
        _quant_flashmla_writer_scales=None,
        _flashmla_writer_buffers=None,
        fak_descale_reciprocal=torch.tensor([4.0]),
        kv_lora_rank=mla_v1.FLASHMLA_V_DIM,
        qk_rope_head_dim=mla_v1.FLASHMLA_QK_DIM - mla_v1.FLASHMLA_V_DIM,
        num_kv_heads=1,
        dtype=torch.float8_e4m3fn if quant else torch.bfloat16,
        W_UV=torch.empty(1),
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(mla_v1.FLASHMLA_V_DIM), variance_epsilon=1e-5),
        _validate_mla_cache_layout=MagicMock(),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=capacity),
            cache_config=SimpleNamespace(block_size=manager_block_size),
        ),
    )


@pytest.mark.parametrize("manager_block_size", MANAGER_BLOCK_SIZES)
def test_quant_staging_uses_backend_kernel_pages(manager_block_size):
    kernel_page = min(AscendMLABackend.get_supported_kernel_block_sizes())
    capacity = kernel_page + 1
    impl = _impl(manager_block_size, capacity)
    AscendMLAImpl._prepare_flashmla_writer_buffers(impl)
    nope, rope, indices = impl._flashmla_writer_buffers
    pages = mla_v1.cdiv(capacity, kernel_page)
    assert nope.shape == (pages, kernel_page, impl.num_kv_heads, impl.kv_lora_rank)
    assert rope.shape == (pages, kernel_page, impl.num_kv_heads, impl.qk_rope_head_dim)
    assert indices.numel() == capacity
    assert impl.vllm_config.cache_config.block_size == manager_block_size


@pytest.mark.parametrize("manager_block_size", MANAGER_BLOCK_SIZES)
def test_nonquant_staging_keeps_existing_prolog_limit(manager_block_size):
    impl = _impl(manager_block_size, manager_block_size + 1, quant=False)
    AscendMLAImpl._prepare_flashmla_writer_buffers(impl)
    nope, rope, _ = impl._flashmla_writer_buffers
    assert nope.shape[1] == rope.shape[1] == min(manager_block_size, mla_v1.PROLOG_V3_MAX_BLOCK_SIZE)
    assert nope.dtype == rope.dtype == torch.bfloat16
    assert impl._quant_flashmla_writer_scales is None


def test_quant_staging_reads_backend_contract_not_a_literal(monkeypatch):
    kernel_page = min(AscendMLABackend.get_supported_kernel_block_sizes())
    alternative_page = kernel_page * 2
    getter = MagicMock(return_value=[alternative_page])
    monkeypatch.setattr(AscendMLABackend, "get_supported_kernel_block_sizes", getter)
    impl = _impl(MANAGER_BLOCK_SIZES[0], alternative_page + 1)
    AscendMLAImpl._prepare_flashmla_writer_buffers(impl)
    assert impl._flashmla_writer_buffers[0].shape[1] == alternative_page
    getter.assert_called_once_with()


def test_quant_staging_survives_manager_replanning_and_reload():
    kernel_page = min(AscendMLABackend.get_supported_kernel_block_sizes())
    impl = _impl(MANAGER_BLOCK_SIZES[0], kernel_page + 1)
    AscendMLAImpl._prepare_flashmla_writer_buffers(impl)
    buffers = impl._flashmla_writer_buffers
    scales = impl._quant_flashmla_writer_scales
    addresses = [x.data_ptr() for x in (*buffers, *scales)]
    torch.testing.assert_close(scales[0], torch.ones(impl.qk_rope_head_dim, dtype=torch.float32))
    torch.testing.assert_close(scales[1], torch.full((impl.kv_lora_rank,), 4.0, dtype=torch.float32))
    impl.vllm_config.cache_config.block_size = MANAGER_BLOCK_SIZES[-1]
    impl.fak_descale_reciprocal = torch.tensor([8.0])
    AscendMLAImpl._prepare_flashmla_writer_buffers(impl)
    assert impl._flashmla_writer_buffers is buffers
    assert addresses == [x.data_ptr() for x in (*buffers, *impl._quant_flashmla_writer_scales)]
    assert buffers[0].shape[1] == kernel_page
    torch.testing.assert_close(scales[1], torch.full((impl.kv_lora_rank,), 8.0, dtype=torch.float32))


@pytest.mark.parametrize(
    "edge,output_kv",
    [
        ("empty", "bf16"),
        ("one", "bf16"),
        ("before_page", "bf16"),
        ("page", "bf16"),
        ("after_page", "bf16"),
        ("capacity", "bf16"),
        ("after_page", "fp8"),
        ("one", "decode"),
        ("one", "context"),
    ],
)
def test_quant_writer_scatter_independent_page_geometry(monkeypatch, edge, output_kv):
    kernel_page = min(AscendMLABackend.get_supported_kernel_block_sizes())
    capacity = kernel_page * 2 + 1
    tokens = {
        "empty": 0,
        "one": 1,
        "before_page": kernel_page - 1,
        "page": kernel_page,
        "after_page": kernel_page + 1,
        "capacity": capacity,
    }[edge]
    impl = _impl(MANAGER_BLOCK_SIZES[0], capacity)
    impl.use_pcp = False
    impl.use_mla_rope = True
    impl._uses_strided_kv_adapter = lambda: False
    impl._exec_quant_flashmla_kv_writer = MethodType(AscendMLAImpl._exec_quant_flashmla_kv_writer, impl)
    AscendMLAImpl._prepare_flashmla_writer_buffers(impl)
    rope_scale, latent_scale = impl._quant_flashmla_writer_scales
    # Persist into differently sized fused pages with reversed token slots.
    persistent_page = kernel_page * 2
    pages = mla_v1.cdiv(capacity, persistent_page)
    fused = torch.zeros(pages, persistent_page, 1, mla_v1.FLASHMLA_QK_DIM).to(impl.dtype)
    cache = (fused[..., : impl.kv_lora_rank], fused[..., impl.kv_lora_rank :])
    slots = torch.arange(tokens - 1, -1, -1, dtype=torch.int64)
    expected_nope = (torch.arange(tokens) % 8).float().view(tokens, 1, 1).expand(tokens, 1, impl.kv_lora_rank)
    expected_rope = expected_nope[..., : impl.qk_rope_head_dim] + 1
    current = (
        torch.full((tokens, 1, impl.qk_rope_head_dim), -1.0, dtype=torch.bfloat16),
        torch.full((tokens, 1, impl.kv_lora_rank), -2.0, dtype=torch.bfloat16),
    )

    def writer(*args, **kwargs):
        assert kwargs["k_rope_scale"] is rope_scale
        assert kwargs["c_kv_scale"] is latent_scale
        assert kwargs["is_output_kv"] is (output_kv == "bf16")
        rope, nope = args[5], args[6]
        assert rope.shape[1] == nope.shape[1] == kernel_page
        assert rope.shape[0] == mla_v1.cdiv(tokens, kernel_page)
        torch.testing.assert_close(args[4], torch.arange(tokens))
        nope.flatten(0, 1)[:tokens].copy_(expected_nope)
        rope.flatten(0, 1)[:tokens].copy_(expected_rope)
        return rope, nope, *(current if output_kv == "bf16" else (None, None))

    def scatter(**kwargs):
        assert kwargs["key"].dtype == kwargs["value"].dtype == torch.float8_e4m3fn
        assert kwargs["slot_mapping"] is slots
        assert kwargs["key_cache"] is cache[0] and kwargs["value_cache"] is cache[1]
        for i, slot in enumerate(slots.tolist()):
            cache[0][slot // persistent_page, slot % persistent_page].copy_(kwargs["key"][i])
            cache[1][slot // persistent_page, slot % persistent_page].copy_(kwargs["value"][i])

    writer_mock = MagicMock(side_effect=writer)
    scatter_mock = MagicMock(side_effect=scatter)
    monkeypatch.setattr(mla_v1.torch_npu, "npu_kv_rmsnorm_rope_cache", writer_mock)
    monkeypatch.setattr(mla_v1.DeviceOperator, "reshape_and_cache", scatter_mock)
    inputs = (
        torch.zeros(tokens, mla_v1.FLASHMLA_QK_DIM, dtype=torch.bfloat16),
        torch.ones(tokens, impl.qk_rope_head_dim),
        torch.zeros(tokens, impl.qk_rope_head_dim),
        cache,
        slots,
    )
    if output_kv in ("bf16", "context"):
        returned = AscendMLAImpl.exec_kv_prefill(impl, *inputs, return_bf16_kv=output_kv == "bf16")
    else:
        cached_rope, cached_nope, *returned = AscendMLAImpl.exec_kv_decode(
            impl, *inputs, return_current_kv=output_kv == "fp8"
        )
        assert cached_rope is cache[1] and cached_nope is cache[0]
    if not tokens:
        writer_mock.assert_not_called()
        scatter_mock.assert_not_called()
        assert returned[0].shape[0] == returned[1].shape[0] == 0
        assert returned[0].dtype == returned[1].dtype == torch.bfloat16
        return
    writer_mock.assert_called_once()
    scatter_mock.assert_called_once()
    if output_kv == "bf16":
        assert returned[0] is current[0] and returned[1] is current[1]
    elif output_kv == "fp8":
        assert returned[0].dtype == returned[1].dtype == torch.float8_e4m3fn
        torch.testing.assert_close(returned[0].float(), expected_rope)
        torch.testing.assert_close(returned[1].float(), expected_nope)
    elif output_kv == "context":
        assert returned == (None, None)
    else:
        assert not returned
    flat = fused.flatten(0, 1).float()
    torch.testing.assert_close(flat[slots, :, : impl.kv_lora_rank], expected_nope)
    torch.testing.assert_close(flat[slots, :, impl.kv_lora_rank :], expected_rope)
    assert torch.count_nonzero(flat[tokens:]) == 0


def test_quant_writer_rejects_capacity_overflow_before_operator(monkeypatch):
    capacity = min(AscendMLABackend.get_supported_kernel_block_sizes())
    impl = _impl(MANAGER_BLOCK_SIZES[0], capacity)
    AscendMLAImpl._prepare_flashmla_writer_buffers(impl)
    op = MagicMock()
    monkeypatch.setattr(mla_v1.torch_npu, "npu_kv_rmsnorm_rope_cache", op)
    with pytest.raises(RuntimeError, match="profiled writer capacity"):
        AscendMLAImpl._exec_quant_flashmla_kv_writer(
            impl,
            torch.empty(capacity + 1, mla_v1.FLASHMLA_QK_DIM),
            None,
            None,
            None,
            None,
        )
    op.assert_not_called()
