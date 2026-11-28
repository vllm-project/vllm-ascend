# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from . import compile_mtp_decode
from .mtp import INDEX_HEAD_PANEL_SHAPE, PAGE_SIZE, mtp_projection_rows, score_context
from .weights import DecodeWeights
from .workspace import scratch_allocator


class MtpQueryAttention:
    """Whole NoPE attention for a TP1 six-query graph bucket on Ascend 950."""

    def __init__(
        self,
        weights: DecodeWeights,
        requests: int,
        index_pages: int,
        main_pages: int,
        ring_pages: int,
        *,
        ring_table_stride=1,
        index_width=2176,
        tokens=None,
        model_context=139264,
        table_columns=None,
        workspace=None,
        head_panel=None,
    ):
        self.weights = weights
        self.requests = requests
        self.tokens = 6 * requests if tokens is None else tokens
        required = (model_context + PAGE_SIZE - 1) // PAGE_SIZE
        self.table_columns = table_columns or (required, required)
        context = score_context(model_context)
        self.main_pages = main_pages
        self.index_pages = index_pages
        self.ring_pages = ring_pages
        self.ring_table_stride = ring_table_stride
        self.index_width = index_width
        device = weights.qkv_a.device
        if device.type != "npu":
            raise ValueError("Expected NPU weights")
        self.kernel = compile_mtp_decode(
            requests,
            index_pages,
            main_pages,
            ring_pages,
            ring_table_stride=ring_table_stride,
            index_width=index_width,
            tokens=self.tokens,
            model_context=model_context,
            table_columns=self.table_columns,
            native_qkv_a=weights.qkv_a.shape == (4096, 2048),
            native_q_b=weights.q_b.shape == (1536, 16384),
            native_output=weights.out.shape == (16384, 4096),
            native_head_weights=weights.w_uv.shape == (64, 512, 256),
        )
        rows = mtp_projection_rows(self.tokens)
        padded_heads = (self.tokens + 5) * 32
        fp8 = torch.float8_e4m3fn
        f32, i16 = torch.float32, torch.int16

        allocate = scratch_allocator(device, workspace)

        self.q = allocate((padded_heads, 128), fp8)
        self.q_scale = allocate((padded_heads,), f32)
        # The score DMA reads all six queries even on a partial request.
        self.head_weight = allocate((max(rows * 32, padded_heads),))[: rows * 32]
        self.runs = allocate((requests, context // 1024), torch.int32)
        self.score = allocate((self.tokens, context // 4))
        self.raw = allocate((rows, 4096))
        self.query = allocate((rows, 64, 512))
        self.latent = allocate(self.query.shape)
        self.value = allocate((rows, 16384))
        self.value_codes = allocate((rows, 16384), fp8)
        self.value_scales = allocate((256, rows), i16)
        self.hidden_codes = allocate((rows, 4096), fp8)
        self.hidden_scales = allocate((64, rows), i16)
        self.qkv = allocate((rows, 2048))
        self.norm = allocate((rows, 1536))
        self.norm_codes = allocate((rows, 1536), fp8)
        self.norm_scales = allocate((24, rows), i16)
        self.query_b = allocate((rows, 16384))
        self.parts = allocate((rows // 32, 32, 32, 32), f32)
        self.key_gate = allocate((rows, 256))
        self.projected = allocate((self.tokens, 256), f32)
        self.head_panel = self.pack_head_panel(weights) if head_panel is None else head_panel

        w = weights
        self.input_projection = (
            self.hidden_codes,
            self.hidden_scales,
            w.qkv_a,
            w.qkv_a_scales,
            self.qkv,
            w.q_norm,
            w.kv_norm,
        )
        self.query_projection = (
            self.norm,
            self.norm_codes,
            self.norm_scales,
            w.q_b,
            w.q_b_scales,
            self.query_b,
            w.w_uk,
            w.index_query,
            w.index_heads,
            self.head_panel,
            self.parts,
            w.index_key_gate.view(1, 256, 4096),
            self.key_gate,
            w.index_norm,
            w.index_bias,
            self.projected,
            w.index_ape,
        )

    @staticmethod
    def pack_head_panel(weights):
        heads = weights.index_heads
        panel = heads.new_zeros(INDEX_HEAD_PANEL_SHAPE, dtype=torch.float32)
        for part, chunk in enumerate(heads.split(147, dim=1)):
            offset = part * 147 % 16
            panel[part, :, offset : offset + chunk.size(1)].copy_(chunk)
        return panel

    def __call__(
        self,
        hidden,
        main_cache,
        main_table,
        main_slots,
        lengths,
        query_ends,
        positions,
        query_starts,
        index_table,
        index_storage,
        ring,
        ring_pages,
        index_slots,
        write_indices,
        selected_indices,
        output,
    ):
        """Write BF16 hidden states, expanded token IDs and both paged caches on the current stream.

        Use one instance per concurrent stream. Active rows occupy a prefix described by
        query_starts. main_slots, ring_pages and index_slots are precomputed scheduler tensors.
        selected_indices contains 2051 logical token IDs per row, optionally padded to 2176.
        """
        if hidden.shape != (self.tokens, 4096) or output.shape != hidden.shape:
            raise ValueError("Input and output must match the compiled decode bucket")
        if (
            index_storage.dtype != torch.uint8
            or index_storage.shape != (self.index_pages, 4352)
            or not index_storage.is_contiguous()
        ):
            raise ValueError("Expected contiguous uint8 index storage with 4352 bytes per page")
        keys = index_storage.view(torch.float8_e4m3fn).view(self.index_pages, 34, 128)
        scales = index_storage.view(torch.float32).view(self.index_pages, 1088)
        if ring.shape != (self.ring_pages, 9, 256) or ring.stride() != (289 * 256, 256, 1):
            raise ValueError("Expected the nine-row KeyPool ring view with a 289-row physical stride")
        if ring_pages.shape != (self.requests,) or ring_pages.stride() != (self.ring_table_stride,):
            raise ValueError("Ring page IDs must match the compiled request count and row stride")
        if (
            selected_indices.shape != (self.tokens, self.index_width)
            or selected_indices.dtype != torch.int32
            or not selected_indices.is_contiguous()
        ):
            raise ValueError("Expected a contiguous int32 token index buffer with the compiled width")
        main_cols, index_cols = self.table_columns
        if (
            main_table.shape != (self.requests, main_cols)
            or index_table.shape != (self.requests, index_cols)
            or not main_table.is_contiguous()
            or not index_table.is_contiguous()
        ):
            raise ValueError("Page tables must match their compiled row widths")
        w = self.weights
        self.kernel(
            self.q,
            self.q_scale,
            keys,
            scales[:, 1024:1056],
            self.head_weight,
            index_table,
            lengths,
            query_ends,
            self.runs,
            self.score,
            self.raw,
            positions,
            selected_indices,
            self.query,
            main_cache.view(torch.int32).view(self.main_pages, 128, 132),
            main_table,
            query_starts,
            self.latent,
            w.w_uv,
            self.value,
            self.value_codes,
            self.value_scales,
            w.out,
            w.out_scales,
            output,
            hidden,
            *self.input_projection,
            main_slots.view(self.tokens, 1),
            *self.query_projection,
            ring_pages,
            ring,
            index_slots,
            write_indices,
        )
        return output
