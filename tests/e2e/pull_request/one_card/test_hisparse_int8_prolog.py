# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare HiSparse's INT8 Prolog writes with the ordinary SFA contract."""

from types import SimpleNamespace as NS

import pytest
import torch
import torch_npu

pytest.importorskip("vllm.v1.hisparse.backend")
from vllm.forward_context import override_forward_context  # noqa: E402
from vllm.v1.hisparse.runtime import PagedCacheView  # noqa: E402
from vllm.v1.worker.workspace import WorkspaceManager  # noqa: E402

from vllm_ascend.attention import sfa_hisparse  # noqa: E402
from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl  # noqa: E402
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl  # noqa: E402
from vllm_ascend.hisparse.cache_ops import gather_resident_rows  # noqa: E402
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod  # noqa: E402
from vllm_ascend.utils import AscendDeviceType, bootstrap_custom_op_env, get_ascend_device_type  # noqa: E402


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize(("hisparse", "pcp"), [(False, False), (True, False), (True, True)])
def test_int8_prolog_writes_match_sfa(capture, hisparse, pcp, monkeypatch):
    """Check INT8 Prolog outputs and KV writes against SFA in eager and graph modes."""
    bootstrap_custom_op_env()
    if not torch.npu.is_available() or get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3):
        pytest.skip("requires A2/A3")
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(42)
    graph = None
    try:
        impl = AscendHiSparseImpl.__new__(AscendHiSparseImpl)
        impl.vllm_config = NS(attention_config=NS(hisparse_config=NS() if hisparse else None))
        impl._quant_type = AscendW8A8DynamicLinearMethod
        impl.use_rope = True
        impl.enable_sparse_sfa_c8 = True
        impl.c8_cache_dtype = torch.int8
        impl.sfa_qsfa_tile_size = 128
        impl.sfa_qsfa_k_nope_clip_alpha = 1.0
        impl.sfa_qsfa_kr_cache_dummy = torch.empty(0, device="npu", dtype=torch.bfloat16)
        impl.is_kv_consumer = False
        impl.has_indexer = True
        impl.pcp_enabled = pcp
        impl.is_pcp_decode_sharded = False
        impl.local_num_heads = 4
        impl.q_lora_rank = 2048
        impl.kv_lora_rank = 512
        impl.qk_rope_head_dim = 64
        impl._kv_staging_capacity = 128
        impl.q_a_layernorm = NS(weight=torch.ones(2048, device="npu", dtype=torch.bfloat16), variance_epsilon=1e-6)
        impl.kv_a_layernorm = NS(weight=torch.ones(512, device="npu", dtype=torch.bfloat16), variance_epsilon=1e-6)
        impl.fused_qkv_a_proj = NS(
            weight=torch_npu.npu_format_cast(torch.randint(-16, 16, (6144, 2624), device="npu", dtype=torch.int8), 29),
            weight_scale=torch.full((2624,), 0.002, device="npu"),
        )
        impl.q_proj = NS(
            weight=torch_npu.npu_format_cast(torch.randint(-16, 16, (2048, 1024), device="npu", dtype=torch.int8), 29),
            weight_scale=torch.full((1024,), 0.002, device="npu"),
        )
        impl.W_UK_T = (torch.randn(4, 192, 512, device="npu", dtype=torch.bfloat16) * 0.01).contiguous()
        expected_weight = torch_npu.npu_format_cast(impl.fused_qkv_a_proj.weight, 2).cpu()
        AscendSFAImpl._process_weights_for_fused_prolog_v3(impl)
        torch.testing.assert_close(torch_npu.npu_format_cast(impl.weight_dq, 2).cpu(), expected_weight[:, :2048])
        torch.testing.assert_close(torch_npu.npu_format_cast(impl.weight_dkv_kr, 2).cpu(), expected_weight[:, 2048:])
        local_slots = torch.arange(128, device="npu", dtype=torch.int64)
        raw = torch.zeros(8 * 6 * 128 * 656, device="npu", dtype=torch.int8)
        cache = PagedCacheView.bind(
            raw,
            dtype=torch.int8,
            row_width=656,
            byte_offset=128 * 656,
            block_stride=6 * 128 * 656,
            num_blocks=8,
            block_size=128,
        ).cache
        rows = 32
        valid = 32
        slots = torch.arange(128, 128 + rows, device="npu", dtype=torch.int32)
        metadata = NS(
            query_start_loc=torch.tensor([0, valid], dtype=torch.int32, device="npu"),
            hisparse_for_capture=True,
            uses_pcp_sharded_layout=pcp,
            num_prefills=int(pcp),
        )
        impl.hisparse_cache = NS(write_target=lambda n, s: (cache, slots, rows))
        x = torch.randn(rows, 6144, device="npu", dtype=torch.bfloat16)
        cos = torch.randn(rows, 1, 1, 64, device="npu", dtype=torch.bfloat16)
        sin = torch.randn_like(cos)
        refcache = (torch.empty((1, 128, 1, 656), device="npu", dtype=torch.int8),)
        if pcp:
            workspace = WorkspaceManager(slots.device)
            monkeypatch.setattr(sfa_hisparse, "current_workspace_manager", lambda: workspace)
            impl.index_group = NS(logical_topk_indices=slots)
            scratch, _ = impl._pcp_prolog_workspace()
            workspace.lock()

        def forward():
            with override_forward_context(NS()):
                return impl._sfa_preprocess_prolog_v3(x, None, cos, sin, slots, attn_metadata=metadata)

        if capture:
            forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = forward()
        for valid, start in ((31, 128), (1, 256)):
            x.neg_()
            slots.copy_(torch.arange(start, start + rows, device="npu", dtype=torch.int32))
            metadata.query_start_loc[-1] = valid
            slots[valid:] = -1
            cache.fill_(-7)
            ref = AscendSFAImpl._sfa_preprocess_prolog_v3(
                impl, x, refcache, cos, sin, local_slots[:rows], attn_metadata=metadata
            )
            if capture:
                assert graph is not None
                graph.replay()
                actual = captured
            else:
                actual = forward()
            for a, b in zip(actual[:3], ref[:3]):
                torch.testing.assert_close(a.cpu(), b.cpu(), atol=0, rtol=0)
            for a, b in zip(actual[3], ref[3]):
                torch.testing.assert_close(a.cpu(), b.cpu(), atol=0, rtol=0)
            expected = refcache[0].view(-1, 656)
            if pcp:
                torch.testing.assert_close(scratch.view(-1, 656)[:rows].cpu(), expected[:rows].cpu(), atol=0, rtol=0)
            else:
                selected = torch.empty((rows, 656), device="npu", dtype=torch.int8)
                gather_resident_rows(
                    cache, selected, torch.arange(start, start + rows, device="npu", dtype=torch.int32)
                )
                torch.testing.assert_close(selected[:valid].cpu(), expected[:valid].cpu(), atol=0, rtol=0)
                assert torch.all(selected[valid:].cpu() == -7)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
        torch.npu.config.allow_internal_format = previous == b"enable"
