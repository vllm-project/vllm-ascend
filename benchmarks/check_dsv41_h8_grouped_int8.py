# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify H8 attention in the layer-packed pool against per-layer G64 on one A2."""

import math
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs

from tests.deepseek_v41_utils import pack_test_caches
from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl, H8GroupedInt8KV
from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.models.deepseek_v41.long_kv import pack_long_kv_int8
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

PAGE, HEADS, DIM, WINDOW, TOPK = 128, 8, 512, 128, 512
SOURCES = (2, 8, 14, 20)
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
CASES = ((1, 1, 32768), (1, 64, 32768), (8, 1, 8192))


def page_view(data, stride_bytes):
    n, rows, width = data.shape
    stride = stride_bytes // data.element_size()
    backing = torch.empty(n, stride, dtype=data.dtype)
    backing[:, : rows * width].copy_(data.reshape(n, -1))
    return backing.npu().as_strided((n, rows, 1, width), (stride, width, width, 1))


def idx(x):
    return x.to(torch.int32).npu()


def sample_indices(batch, q, length, ratio):
    selected = []
    for _ in range(batch):
        for j in range(q):
            visible = (length - q + j + 1) // ratio
            # Always include the most recent visible row. Other rows are distinct.
            prev = torch.randperm(visible - 1, dtype=torch.int32)[: TOPK - 1].sort().values
            selected.append(torch.cat((prev, torch.tensor([visible - 1], dtype=torch.int32))))
    return torch.stack(selected).unsqueeze(1).npu()


def packed_rows(data):
    chunks = data.float().reshape(-1, DIM // 64, 64)
    scales = chunks.abs().amax(-1) / 127
    codes = (chunks / torch.where(scales == 0, 1, scales).unsqueeze(-1)).round().to(torch.int8)
    return torch.cat((codes.flatten(1), scales.view(torch.int8)), dim=-1).view(*data.shape[:-1], 544)


def make_chain(batch, q, length):
    torch.manual_seed(93)
    op = torch.ops._C_ascend
    tokens = batch * q
    npages = math.ceil(length / PAGE)
    # Only the final window plus the current chunk needs live SWA storage.
    ring = math.ceil((WINDOW + q - 1) / PAGE) + 1
    swa_table = idx(torch.arange(npages)[None, :] % ring + torch.arange(batch)[:, None] * ring)
    long_table = torch.randperm(batch * npages, dtype=torch.int32).view(batch, npages)
    device_table = idx(long_table)
    sources, layers = {}, {}
    for ratio in (1, 2):
        cmp_len = length // ratio
        kwargs = dict(
            cu_seqlens_q=idx(torch.arange(batch + 1) * q),
            seqused_ori_kv=idx(torch.full((batch,), length)),
            seqused_cmp_kv=idx(torch.full((batch,), cmp_len)),
            cmp_residual_kv=idx(torch.full((batch,), length % 2)) if ratio == 2 else None,
            cmp_ratio=ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=WINDOW - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
        )
        meta = op.npu_sparse_flash_mla_metadata(
            HEADS,
            1,
            DIM,
            batch_size=batch,
            max_seqlen_q=q,
            max_seqlen_ori_kv=length,
            max_seqlen_cmp_kv=cmp_len,
            cmp_topk=TOPK,
            has_ori_kv=True,
            has_cmp_kv=True,
            **kwargs,
        )
        sources[ratio] = dict(
            kwargs, metadata=meta, ori_block_table=swa_table, cmp_block_table=device_table, softmax_scale=DIM**-0.5
        )

    caches = {}
    for layer in range(2, 40):
        ratio = 2 if layer < 20 else 1
        if layer in SOURCES:
            owner = layer
            physical_rows = PAGE // ratio
            prior = torch.randn(batch * npages, physical_rows, DIM, dtype=torch.bfloat16)
            positions = torch.arange(length - q, length).repeat(batch)
            req = torch.arange(batch).repeat_interleave(q)
            rows = positions // ratio
            coords = torch.stack((long_table[req, rows // physical_rows], rows % physical_rows), dim=1)
            if ratio == 2:
                coords[positions % 2 == 0] = -1
            slots = idx(coords)
            latent = torch.randn(tokens, DIM, dtype=torch.bfloat16).npu()
            caches[owner] = (page_view(packed_rows(prior), 131072), slots, latent)
        if layer in INDEX_SOURCES:
            indices = sample_indices(batch, q, length, ratio)
        query = torch.randn(tokens, HEADS, DIM, dtype=torch.bfloat16).npu()
        sink = torch.randn(HEADS).npu()
        swa = torch.randn(batch * ring, PAGE, DIM, dtype=torch.bfloat16)
        layers[layer] = (owner, query, sink, page_view(swa, 131072), dict(sources[ratio], cmp_sparse_indices=indices))

    packed_sources, packed_long_table, packed_swa = pack_test_caches(
        caches, layers, device_table, swa_table, batch * ring
    )

    workspace = H8GroupedInt8KV(indices)
    payload = workspace.buffer[2 * sources[1]["metadata"].numel() :].view(torch.int8).view(tokens, TOPK, 544)

    def forward(shared):
        outputs = []
        for layer, (owner, query, sink, swa, kwargs) in layers.items():
            cache, slots, latent = (packed_sources if shared else caches)[owner]
            if owner == layer:
                update = pack_long_kv_int8(latent)
                op.npu_scatter_nd_update_sk(cache.squeeze(2), slots, update)
            head = layer in INDEX_SOURCES
            ratio = kwargs["cmp_ratio"]
            if shared:
                swa, packed_swa_table = packed_swa[layer]
                if head:
                    payload.fill_(-87)
                # Use the serving call site, including its metadata handoff and active-token view.
                impl = SimpleNamespace(
                    role=SimpleNamespace(compress_ratio=ratio, has_long_context=True),
                    topology=SimpleNamespace(index_topk=TOPK),
                    grouped_int8_kv=workspace,
                    topk_value_mode=workspace.WRITE if head else workspace.READ,
                )
                attn = SimpleNamespace(
                    dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=swa)),
                    attn_sink=sink,
                    softmax_scale=kwargs["softmax_scale"],
                )
                meta = SimpleNamespace(
                    swa=SimpleNamespace(
                        num_reqs=batch,
                        query_start_loc=kwargs["cu_seqlens_q"],
                        seq_lens=kwargs["seqused_ori_kv"],
                        block_table=packed_swa_table,
                        replay_block_table=None,
                        ori_sparse_indices=None,
                        ori_topk_length=None,
                        ori_mask_mode=4,
                        ori_win_left=WINDOW - 1,
                        ori_win_right=0,
                    ),
                    attention=SimpleNamespace(
                        block_table=packed_long_table,
                        cache_seq_lens=kwargs["seqused_cmp_kv"],
                        cmp_residual=kwargs["cmp_residual_kv"],
                        smla_metadata=kwargs["metadata"],
                    ),
                )
                result = AscendDSAV41Impl._forward_attention(
                    impl, attn, query, meta, kwargs["cmp_sparse_indices"].squeeze(1), source_cache=cache
                )
            else:
                result = op.npu_sparse_flash_mla(query, ori_kv=swa, cmp_kv=cache, sinks=sink, **kwargs)[0]
            outputs.append(result)
            if shared and head:
                candidates = kwargs["cmp_sparse_indices"].cpu().squeeze(1).long()
                request = torch.arange(tokens)[:, None] // q
                page = long_table[request, candidates // (PAGE // ratio)].long()
                reference = caches[owner][0].cpu()[page, candidates % (PAGE // ratio), 0, :]
                assert torch.equal(payload.cpu(), reference), (layer, "INT8 payload or FP32 scale bytes differ")
        return outputs

    return forward


if __name__ == "__main__":
    torch.set_num_threads(4)
    bootstrap_custom_op_env()
    torch.npu.set_device(0)
    if not enable_custom_op():
        raise RuntimeError("Build and enable the vLLM Ascend custom operators before running this check.")
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    for batch, query_length, context in CASES:
        run = make_chain(batch, query_length, context)
        reference = run(shared=False)
        grouped = run(shared=True)
        for expected, actual in zip(reference, grouped, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(f"B={batch}, Q={query_length}, context={context}: 8 payloads and 38 layer outputs verified")
