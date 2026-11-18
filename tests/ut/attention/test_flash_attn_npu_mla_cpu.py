# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU boundary tests; these do not validate the installed Ascend kernel.

Run with --confcutdir=tests/ut/attention when vLLM/torch_npu are unavailable.
"""

import ast
import runpy
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from .test_flashmla_dcp import ROOT, common, load_method
from .test_flashmla_dcp import runtime as _dcp_runtime

dcp_runtime = _dcp_runtime


@pytest.fixture
def a3(dcp_runtime, monkeypatch):
    def schedule(**kwargs):
        lengths, cu = kwargs["cache_seqlens"], kwargs["cu_seqlens_q"]
        # Preserve Meta dispatch and refresh visibility without emulating tiling.
        return torch.cat((lengths, cu)).view(torch.uint8)

    metadata = Mock(side_effect=schedule)

    def attention(q, rope, nope, **kwargs):
        tokens, heads = q.shape[:2]
        out = torch.zeros(tokens, heads, 512, dtype=q.dtype)
        lse = torch.full((heads, tokens), torch.inf, dtype=torch.float32)
        for row, length in enumerate(kwargs["cache_seqlens"].tolist()):
            start, end = kwargs["cu_seqlens_q"][row : row + 2].tolist()
            if not length:
                continue
            idx = torch.arange(length)
            pages = kwargs["page_table"][row, idx // 128].long()
            key_rope = rope[pages, idx % 128, 0].double()
            key_nope = nope[pages, idx % 128, 0].double()
            for token in range(start, end):
                visible = length - (end - token) + 1 if kwargs["causal"] else length
                scores = (
                    q[token].double() @ key_rope[:visible].T + kwargs["qv"][token].double() @ key_nope[:visible].T
                ) * kwargs["softmax_scale"]
                out[token] = (scores.softmax(-1) @ key_nope[:visible]).to(q.dtype)
                lse[:, token] = scores.logsumexp(-1).float()
        return (out, lse, None, None) if kwargs["return_softmax_lse"] else out

    op = Mock(side_effect=attention)
    utils = sys.modules["vllm_ascend.attention.utils"]
    monkeypatch.setattr(utils, "get_flash_attn_npu_ops", lambda: (op, metadata), raising=False)
    api = SimpleNamespace(**runpy.run_path(str(ROOT / "vllm_ascend/attention/flash_attn_npu.py")))

    def builder(size=1, speculative=False, rank=0):
        impl = SimpleNamespace(
            num_heads=8,
            scale=0.125,
            dtype=torch.float16,
            use_mla_rope=False,
            dcp_size=size,
            dcp_rank=rank,
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
                speculative_config=object() if speculative else None,
            ),
        )
        return api.FlashAttnNPUMetadataBuilder(impl, torch.device("cpu"), 8, torch.ones(2048, 2048, dtype=torch.int8))

    return SimpleNamespace(api=api, op=op, metadata=metadata, builder=builder, dcp=dcp_runtime)


def paged_components(pages=5):
    # Nonzero offset and padded physical pages, as in hybrid Kimi caches.
    def component(dim):
        stride = 128 * dim + 32
        backing = torch.randn(pages * stride + 16, dtype=torch.float16) * 0.1
        return backing.as_strided((pages, 128, 1, dim), (stride, dim, dim, 1), 16)

    return component(512), component(64)


@pytest.mark.parametrize("causal", [False, True])
def test_a3_paged_absorbed_mapping_matches_dense_and_preserves_cache(a3, causal):
    torch.manual_seed(5)
    inputs = common([132, 2], [3, 1], tokens=6, causal=causal)
    inputs.block_table_tensor.copy_(torch.tensor([[3, 1, 2], [4, 0, 2]]))
    flash = a3.builder().build(inputs, 2, 4, False)
    flash.query.normal_(std=0.1)
    caches = paged_components()
    snapshots = tuple(cache.clone() for cache in caches)
    output, _ = flash.adapter.attention(
        flash.query,
        caches,
        block_table=flash.block_table,
        cache_seqlens=flash.cache_lens,
        cu_seqlens_q=flash.cu,
        seqused_q=flash.used_q,
        metadata=flash.schedule,
    )
    expected = torch.zeros_like(output)
    for row, length in enumerate([132, 2]):
        idx = torch.arange(length)
        pages = flash.block_table[row, idx // 128].long()
        keys = torch.cat([cache[pages, idx % 128, 0] for cache in caches], -1).double()
        start, end = flash.cu[row : row + 2].tolist()
        for token in range(start, end):
            visible = length - (end - token) + 1 if causal else length
            scores = flash.query[token].double() @ keys[:visible].T * 0.125
            expected[:, token] = (scores.softmax(-1) @ keys[:visible, :512]).to(output.dtype)
    torch.testing.assert_close(output, expected)
    for cache, snapshot in zip(caches, snapshots):
        torch.testing.assert_close(cache, snapshot)
    args, kwargs = a3.op.call_args
    assert args[1] is caches[1] and args[2] is caches[0]
    assert args[0].untyped_storage().data_ptr() == flash.query.untyped_storage().data_ptr()
    assert kwargs["qv"].untyped_storage().data_ptr() == flash.query.untyped_storage().data_ptr()
    assert kwargs["scheduler_metadata"] is flash.schedule
    assert kwargs["max_seqlen_q"] == 6
    assert kwargs["causal"] is causal
    meta = a3.metadata.call_args.kwargs
    assert (meta["headdim"], meta["headdim_v"], meta["page_size"], meta["max_seqlen_k"]) == (64, 512, 128, 384)
    assert meta["qkv_dtype"] == torch.float16 and meta["causal"] is causal


@pytest.mark.parametrize("size,speculative", [(1, False), (8, False), (8, True)])
def test_a3_graph_capacity_refresh_and_split_current_cache(a3, size, speculative):
    builder = a3.builder(size=size, speculative=speculative)
    inputs = common([132, 4], [3, 1], tokens=8)
    first = builder.build(inputs, 2, 4, False, retain_for_graph=True)
    schedule = first.schedule.clone()
    addresses = {name: value.data_ptr() for name, value in vars(first).items() if isinstance(value, torch.Tensor)}
    inputs.seq_lens.add_(size)
    again = builder.build(inputs, 2, 4, False)
    assert again is first
    assert addresses == {
        name: value.data_ptr() for name, value in vars(again).items() if isinstance(value, torch.Tensor)
    }
    assert not torch.equal(schedule, again.schedule)
    assert again.schedule.dtype == torch.uint8
    if size > 1 and speculative:
        assert isinstance(again.current.cache, tuple)
        assert all(cache.stride(1) == cache.shape[-1] for cache in again.current.cache)
        assert again.current.adapter.max_seqlen_k == again.current.block_table.shape[1] * 128
    else:
        assert again.current is None


def test_a3_rejects_fused_history_before_package_could_copy_it(a3):
    flash = a3.builder().build(common([2], [1]), 1, 1, False)
    raw = torch.empty(2, 128, 1, 576, dtype=torch.float16)
    for cache in (raw, (raw[..., :512], raw[..., 512:])):
        with pytest.raises(ValueError, match="separate|packed component"):
            flash.adapter.attention(
                flash.query,
                cache,
                block_table=flash.block_table,
                cache_seqlens=flash.cache_lens,
                cu_seqlens_q=flash.cu,
                metadata=flash.schedule,
            )
    a3.op.assert_not_called()


def test_a3_single_rank_forward_preserves_batch_major_projection(a3):
    flash = a3.builder().build(common([132], [3]), 1, 3, False)
    query = torch.randn_like(flash.query) * 0.1
    preprocessed = SimpleNamespace(ql_nope=query[..., :512], q_pe=query[..., 512:])
    project = Mock(side_effect=lambda latent: latent)
    impl = SimpleNamespace(use_flash_attn_npu=True, kv_lora_rank=512, _v_up_proj_batch_major=project)
    scope = dict(torch=torch, record_attention_compute_start=Mock())
    forward = load_method("vllm_ascend/attention/mla_v1.py", "AscendMLAImpl", "_forward_external_flashmla", scope)
    result = forward(impl, preprocessed, paged_components(), SimpleNamespace(external_flashmla=flash))
    torch.testing.assert_close(flash.query, query)
    project.assert_called_once()
    assert result.shape == (3, 8, 512) and result.is_contiguous()


@pytest.mark.parametrize("speculative", [False, True])
def test_a3_dcp_production_orchestration_matches_dense(a3, monkeypatch, speculative):
    torch.manual_seed(19)
    size, heads = 2, 8
    qlen, length = (2, 131) if speculative else (1, 130)
    tokens = 3
    keys = torch.randn(length, 576, dtype=torch.float16) * 0.1
    query = torch.randn(tokens, size * heads, 576, dtype=torch.float16) * 0.1
    flashes, caches, partials = [], [], []
    for rank in range(size):
        inputs = common([length], [qlen], tokens=tokens)
        flash = a3.builder(size=size, speculative=speculative, rank=rank).build(inputs, 1, qlen, False)
        flash.query.copy_(query)
        cache = paged_components()
        local = keys[torch.arange(length) % size == rank]
        for row, vector in enumerate(local):
            cache[0][row // 128, row % 128, 0].copy_(vector[:512])
            cache[1][row // 128, row % 128, 0].copy_(vector[512:])
        partials.append(
            flash.adapter.attention(
                flash.query,
                cache,
                block_table=flash.block_table,
                cache_seqlens=flash.cache_lens,
                cu_seqlens_q=flash.cu,
                seqused_q=flash.used_q,
                metadata=flash.schedule,
            )
        )
        flashes.append(flash)
        caches.append(cache)
    expected = torch.zeros(tokens, size * heads, 512, dtype=torch.float16)
    for token in range(qlen):
        visible = length - qlen + token + 1
        scores = query[token].double() @ keys[:visible].double().T * 0.125
        expected[token] = (scores.softmax(-1) @ keys[:visible, :512].double()).half()
    for rank, (flash, cache) in enumerate(zip(flashes, caches)):
        owned = slice(rank * heads, (rank + 1) * heads)

        def exchange(*args, owned=owned, **kwargs):
            return (
                [out[owned].transpose(0, 1) for out, _ in partials],
                [lse[owned].transpose(0, 1).unsqueeze(-1) for _, lse in partials],
            )

        monkeypatch.setattr(torch.ops.vllm, "dcp_a2a_fused", exchange, raising=False)
        impl = a3.dcp.bind(
            SimpleNamespace(
                use_flash_attn_npu=True,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                num_kv_heads=1,
                num_heads=heads,
                dcp_size=size,
                dcp_rank=rank,
                dcp_group=SimpleNamespace(unique_name="cpu-dcp"),
                _v_up_proj_batch_major=lambda latent: latent,
            )
        )
        current = torch.zeros(tokens, 1, 576, dtype=torch.float16)
        current[:qlen, 0].copy_(keys[-qlen:])
        preprocessed = SimpleNamespace(
            ql_nope=query[..., :512],
            q_pe=query[..., 512:],
            current_k_nope=current[..., :512] if speculative else None,
            current_k_pe=current[..., 512:] if speculative else None,
        )
        result = a3.dcp.forward(
            impl,
            preprocessed,
            cache,
            SimpleNamespace(
                external_flashmla=flash,
                decode=SimpleNamespace(cp_history_seq_len=None, actual_seq_lengths_q=None),
            ),
        )
        torch.testing.assert_close(result, expected[:, owned], atol=2e-4, rtol=2e-3)
        impl._run_dcp_mtp_split_attention_op.assert_not_called()


@pytest.mark.parametrize("device", ["A2", "A3", "A5"])
def test_a3_selection_does_not_change_a5_or_a2(device):
    class Capability(Enum):
        MLA_FLASH = 1
        MLA_FLASH_ATTN_NPU = 2

    support = {"A2": set(), "A3": {Capability.MLA_FLASH_ATTN_NPU}, "A5": {Capability.MLA_FLASH}}
    scope = dict(
        HardwareCapability=Capability,
        MLA_FLASH_SUPPORTED_Q_HEADS={8, 12, 64, 96},
        FLASHMLA_QK_DIM=576,
        FLASHMLA_V_DIM=512,
        get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda cap: cap in support[device]),
        supports_component_major_mla_pd=lambda config: True,
        get_flashmla_ops=lambda: (object(), object()),
        get_flash_attn_npu_ops=lambda: (object(), object()),
        torch=torch,
    )
    select = load_method("vllm_ascend/attention/mla_v1.py", "AscendMLAImpl", "_can_use_flashmla", scope)
    impl = SimpleNamespace(
        num_heads=8,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fa_quant_layer=False,
        dtype=torch.bfloat16,
        enable_kv_nz=False,
        use_pcp=False,
        vllm_config=SimpleNamespace(
            use_v2_model_runner=True,
            speculative_config=None,
            parallel_config=SimpleNamespace(decode_context_parallel_size=8),
        ),
    )
    assert select(impl) is (device == "A5")
    assert select(impl, flash_attn_npu=True) is (device == "A3")
    scope["get_flash_attn_npu_ops"] = lambda: None
    assert not select(impl, flash_attn_npu=True)
    scope["get_flash_attn_npu_ops"] = lambda: (object(), object())
    for field, value in (("use_pcp", True), ("enable_kv_nz", True), ("fa_quant_layer", True), ("kv_lora_rank", 256)):
        original = getattr(impl, field)
        setattr(impl, field, value)
        assert not select(impl, flash_attn_npu=True)
        setattr(impl, field, original)


def test_a3_loader_only_treats_missing_public_apis_as_unavailable():
    scope = dict(find_spec=lambda name: object(), Callable=object)
    scope["import_module"] = lambda name: SimpleNamespace(flash_attn_with_kvcache=Mock())
    tree = ast.parse((ROOT / "vllm_ascend/attention/utils.py").read_text(encoding="utf-8"))
    node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_flash_attn_npu_ops"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), "<loader>", "exec"), scope
    )
    loader = scope["get_flash_attn_npu_ops"]
    assert loader() is None
    scope["find_spec"] = lambda name: None
    assert loader() is None
    scope["find_spec"] = lambda name: object()

    def broken(name):
        raise ModuleNotFoundError("missing binary dependency", name="flash_attn_npu_3.flash_attn_npu_3")

    scope["import_module"] = broken
    with pytest.raises(ModuleNotFoundError, match="binary dependency"):
        loader()


def test_actual_hardware_profiles_keep_operator_families_separate(monkeypatch):
    hardware = ModuleType("vllm_ascend.device.hardware")
    hardware.__dict__.update(runpy.run_path(str(ROOT / "vllm_ascend/device/hardware.py")))
    monkeypatch.setitem(sys.modules, "vllm_ascend.device.hardware", hardware)
    config = ModuleType("vllm_ascend.device.device_config")
    config.get_device_config = lambda: SimpleNamespace(_device_type=hardware.AscendDeviceType.A3)
    monkeypatch.setitem(sys.modules, "vllm_ascend.device.device_config", config)
    profile = runpy.run_path(str(ROOT / "vllm_ascend/device/hardware_profile.py"))
    capability = profile["HardwareCapability"]
    for family in hardware.AscendDeviceType:
        selected = profile["get_hardware_profile"](family)
        assert selected.supports(capability.MLA_FLASH_ATTN_NPU) is (family == hardware.AscendDeviceType.A3)
        assert selected.supports(capability.MLA_FLASH) is (family == hardware.AscendDeviceType.A5)
