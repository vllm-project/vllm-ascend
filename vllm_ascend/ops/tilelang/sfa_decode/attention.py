# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, fields

import torch

from . import compile_single_decode
from .single import (
    MAX_RESIDENT_REQUESTS,
    WORKSPACE_PADDING_BYTES,
    head_panel_layout,
    latent_workspace_elements,
    projection_rows,
    query_workspace_rows,
    resident_workspace_bytes,
)
from .workspace import scratch_allocator


@dataclass(frozen=True)
class DecodeWeights:
    """TP1 GLM-5.3-Flash weights. Prepared MX codes are [out, in] and the two E8M0
    scales per 64 input channels are packed into int16 [in // 64, out].
    Q6 can reuse a native FRACTAL_NZ [in, out] output matrix and input-major W_UK and W_UV.
    Small Q1 workers can store output weights and scales as [K panel, out, inner K].
    """

    qkv_a: torch.Tensor
    qkv_a_scales: torch.Tensor
    q_b: torch.Tensor
    q_b_scales: torch.Tensor
    w_uk: torch.Tensor
    w_uv: torch.Tensor
    out: torch.Tensor
    out_scales: torch.Tensor
    q_norm: torch.Tensor
    kv_norm: torch.Tensor
    index_key_gate: torch.Tensor
    index_norm: torch.Tensor
    index_bias: torch.Tensor
    index_ape: torch.Tensor
    index_query: torch.Tensor
    index_heads: torch.Tensor


class SingleQueryAttention:
    """One active query per request in a fixed Ascend 950 C8 NoPE graph bucket.

    Dense projections pad to Cube tiles inside the kernel. Tables have 128-token pages and
    cover the compiled raw-token context. Pass (main, index) physical widths in table_columns
    for wider scheduler tables and index_pages for a separate compressed cache.
    tail_layout specifies (page count, FP32 page stride, page-ID stride).
    Set index_norm_eps to the indexer's LayerNorm epsilon.
    With mask_padding, a negative main slot marks an inactive graph row.
    Use a separate instance per concurrent stream.
    """

    def __init__(
        self,
        weights: DecodeWeights,
        batch: int,
        pages: int,
        context: int,
        *,
        index_width=512,
        index_pages=None,
        table_columns=None,
        tail_layout=None,
        index_norm_eps=1e-5,
        mask_padding=False,
        workspace=None,
        head_panel=None,
    ):
        self.weights = tuple(getattr(weights, field.name) for field in fields(weights))
        device = weights.qkv_a.device
        if device.type != "npu" or any(w.device != device for w in self.weights):
            raise ValueError("All attention weights must be on the same NPU")
        self.batch = batch
        self.pages = pages
        self.index_pages = pages if index_pages is None else index_pages
        self.index_width = index_width
        self.tail_layout = (batch, 4 * 256, 1) if tail_layout is None else tuple(tail_layout)
        resident_workspace = 1 < batch <= MAX_RESIDENT_REQUESTS and (
            workspace is None or workspace.numel() >= resident_workspace_bytes(batch, context) + WORKSPACE_PADDING_BYTES
        )
        self.kernel = compile_single_decode(
            batch,
            pages,
            context,
            index_width=index_width,
            prepacked_heads=True,
            packed_output=weights.out.ndim == 3,
            index_pages=self.index_pages,
            table_columns=table_columns,
            tail_layout=self.tail_layout,
            index_norm_eps=index_norm_eps,
            mask_padding=mask_padding,
            resident_workspace=resident_workspace,
        )
        if weights.index_heads.shape != (32, 4096) or weights.index_heads.dtype != torch.bfloat16:
            raise ValueError("Expected 32 BF16 index head vectors of width 4096")
        if head_panel is None:
            head_panel = self.pack_head_panel(weights, batch)
        self.weights = (*self.weights[:-1], head_panel)
        fp8 = torch.float8_e4m3fn

        allocate = scratch_allocator(device, workspace)

        def prefix(buffer, rows, width, dtype):
            return buffer.view(dtype).flatten()[: rows * width].view(rows, width)

        rows = projection_rows(batch)
        hidden_codes = allocate((rows, 4096), fp8)
        hidden_scales = allocate((66 if mask_padding else 64, rows), torch.int16)
        query = allocate((query_workspace_rows(batch, rows, resident_workspace), 32768))
        query_b = allocate((rows, 16384))
        latent_storage = allocate((latent_workspace_elements(batch, rows, context),))
        latent = prefix(latent_storage, rows, 32768, torch.bfloat16)
        # Each alias spans disjoint phases of the same graph replay. UK reads query_b
        # while selection reads scores; query and latent must remain distinct.
        self.scratch = (
            hidden_codes,
            hidden_scales,
            prefix(query, rows, 1536, fp8),
            allocate((24, rows), torch.int16),
            allocate((rows, 16384), fp8),
            allocate((256, rows), torch.int16),
            prefix(query_b, rows, 2048, torch.bfloat16),
            allocate((rows, 1536)),
            query_b,
            query,
            latent,
            allocate((rows, 16384)),
            allocate((rows, 4096)),
            hidden_codes.view(rows, 32, 128),
            hidden_scales[:64].view(torch.float32).view(rows, 32),
            allocate((rows, 32)),
            prefix(latent_storage, batch, context // 4, torch.bfloat16),
        )

    @staticmethod
    def pack_head_panel(weights, batch):
        parts, width, aligned = head_panel_layout(batch)
        if parts == 1:
            return weights.index_heads.float().contiguous()
        panel = weights.index_heads.new_zeros((parts, 32, aligned), dtype=torch.float32)
        for part, chunk in enumerate(weights.index_heads.split(width, dim=1)):
            panel[part, :, : chunk.size(1)].copy_(chunk)
        return panel.view(parts * 32, aligned)

    def __call__(
        self,
        hidden,
        main_cache,
        main_table,
        slots,
        lengths,
        tail,
        tail_pages,
        index_table,
        index_storage,
        selected_indices,
        output,
    ):
        """Write the output and indices, and update the main, compressed and tail caches.

        index_width=512 returns ordered pool IDs. Widths 2051 and 2176 return token IDs.
        index_storage contains 4352 bytes per page. Bytes 0..4095 store FP8 keys,
        bytes 4096..4223 store their FP32 scales. Each row of tail holds four FP32 key/gate vectors.
        Each active request owns its writable cache slots for this invocation.
        """
        tail_count, row_stride, id_stride = self.tail_layout
        if (
            tail.shape != (tail_count, 4, 256)
            or tail.stride() != (row_stride, 256, 1)
            or tail_pages.shape != (self.batch,)
            or tail_pages.stride() != (id_stride,)
        ):
            raise ValueError("Tail storage and page IDs must match the compiled layout")
        if (
            index_storage.dtype != torch.uint8
            or index_storage.shape != (self.index_pages, 4352)
            or not index_storage.is_contiguous()
        ):
            raise ValueError("Expected contiguous uint8 index storage with 4352 bytes per page")
        if hidden.shape != (self.batch, 4096) or output.shape != hidden.shape:
            raise ValueError("Input and output must match the compiled decode bucket")
        if selected_indices.shape != (self.batch, self.index_width) or selected_indices.dtype != torch.int32:
            raise ValueError("Index output must be int32 and match the compiled index width")
        code = index_storage.view(torch.float8_e4m3fn).view(self.index_pages, 34, 128)
        scale = index_storage.view(torch.float32).view(self.index_pages, 1088)
        self.kernel(
            hidden,
            *self.weights,
            main_cache,
            main_table,
            slots,
            lengths,
            tail,
            tail_pages,
            index_table,
            code,
            scale,
            *self.scratch,
            selected_indices,
            output,
        )
        return output
