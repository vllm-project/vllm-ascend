# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from functools import lru_cache

from tilelang import tvm  # type: ignore[import-not-found]

from ..shared.compiler import compile_fused, cube, mixed, phase, vector
from ..shared.index_query_quant_exact_div import program as quant_program
from ..shared.index_score import score_program
from ..shared.keypool import pool_norm
from ..shared.mx_projection import mx_projection, output_projections, query_projection
from ..shared.qkv_norm import qkv_norm
from ..shared.selection import CHUNK_POOLS, CHUNKED_CONTEXT, pack_dense_indices, selection_phases
from ..shared.sparse_attention import DIM, paged_attention
from ..shared.tile_head_bmm_geometry import head_gemm, wuv_program
from ..shared.tile_keypool_partial_row import partial_projection
from ..shared.tile_mx_clamped_scale import quant_program as output_quant
from . import (
    DENSE_PREFIX_TOKENS,
    PAGE_SIZE,
    Q6_MAX_CONTEXT,
    Q6_MAX_REQUESTS,
    TOKENS_PER_REQUEST,
    mtp_projection_rows,
    mtp_projection_tile,
    score_context,
)
from .dense_cache import prepare_dense_cache
from .exact_head import dot, pack_x
from .exact_head import finish as native_finish
from .live_cache_abi import abi_program
from .live_stride_ring import update_program
from .pad_query_rows import prepare as input_edges


@lru_cache(maxsize=128)
def compile_mtp_decode(
    requests,
    index_pages,
    main_pages,
    ring_pages,
    *,
    ring_table_stride=1,
    index_width=2176,
    tokens=None,
    model_context=139264,
    table_columns=None,
    native_output=False,
    native_head_weights=False,
    native_qkv_a=False,
    native_q_b=False,
):
    """Compile the six-query C8 decode graph. Use the tensor layout in mtp.live_cache_abi.

    The main and indexer page tables retain their own physical row strides.
    The caller owns the workspace, output and writable cache buffers.
    """

    if not 0 < model_context <= Q6_MAX_CONTEXT:
        raise ValueError("Expected an MTP5 context between 1 and 1048576 tokens")
    required = (model_context + PAGE_SIZE - 1) // PAGE_SIZE
    main_cols, index_cols = table_columns or (required, required)
    if min(main_cols, index_cols) < required:
        raise ValueError("Page tables cannot cover the configured context")
    r, b, p, ctx = requests, requests * 6 if tokens is None else tokens, index_pages, score_context(model_context)

    cols = ctx // PAGE_SIZE
    rows = mtp_projection_rows(b)
    if not 1 <= r <= Q6_MAX_REQUESTS or not 1 <= b <= r * TOKENS_PER_REQUEST:
        raise ValueError("Invalid request or token count for MTP5 decode")
    if min(p, main_pages, ring_pages) < 1:
        raise ValueError("The index, main and ring page arenas must be nonempty")
    if ring_table_stride < 1:
        raise ValueError("Ring page IDs must have a positive row stride")
    bm = mtp_projection_tile(rows)
    bulk = rows >= 128
    if index_width not in (2051, 2176):
        raise ValueError("Expected 2051 logical or 2176 padded token IDs per query")
    abi = abi_program(
        r,
        b,
        p,
        cols,
        main_pages,
        main_cols,
        ring_pages,
        ring_table_stride,
        index_width,
        index_cols,
        native_output=native_output,
        native_head_weights=native_head_weights,
        native_qkv_a=native_qkv_a,
        native_q_b=native_q_b,
    )

    score_rows = ctx // 4 // DIM if bulk and (ctx // 4 // DIM) * TOKENS_PER_REQUEST >= PAGE_SIZE else 0
    needs_score = tvm.tirx.Var("score_needed", "bool") if bulk else None
    dense = tvm.tirx.Not(needs_score) if score_rows else None

    multiple = tvm.tirx.Var("multiple_chunks", "bool") if ctx >= CHUNKED_CONTEXT else None

    def controls(params):
        lengths = tvm.tirx.decl_buffer((r,), "int32", data=params["Length"])
        bindings = []
        for flag, threshold in ((needs_score, DENSE_PREFIX_TOKENS), (multiple, (CHUNK_POOLS + 1) * 4)):
            if flag is not None:
                bindings.append(tvm.tirx.Bind(flag, tvm.tirx.any(*[lengths[i] >= threshold for i in range(r)])))
        return bindings

    chunk_guards: tuple = ()
    if multiple is not None:
        single = tvm.tirx.Not(multiple)
        chunk_guards = (tvm.tirx.And(needs_score, single) if score_rows else single, multiple)

    def project(program, **bindings):
        return cube(program, independent=bulk, **bindings)

    cache = []
    output = [
        project(out, C3="OC", S3="XS", Y="Final", nz=native_output)
        for out in output_projections(rows, b, bm, bulk=bulk, native=native_output)
    ]
    if score_rows:
        cache.append(
            vector(
                prepare_dense_cache(r, main_pages, main_cols, b, score_rows), Cache="MK", Table="MT", PreparedKV="Score"
            )
        )
        output.append(vector(pack_dense_indices(b, r, index_width), Y="Packed", guard=dense))

    # The 950DT native 4096 x 2048 MXFP8 projection uses three K partials in this token range.
    split_qkv = not native_qkv_a or 23 <= b <= 160
    mx_tile_rows = 128 if bulk and rows % 128 == 0 else bm
    qa = mx_projection(
        rows,
        4096,
        2048,
        mx_tile_rows if bulk else 32,
        split_width=1536 if split_qkv else None,
        balance_columns=not split_qkv,
        bn=128 if bulk else 64,
        stages=2 if bulk else 1,
        native_weight=native_qkv_a,
    )
    uk = head_gemm(
        rows,
        64,
        256,
        512,
        bm,
        256 if bulk else 128,
        128,
        2,
        l1_k=256 if bulk else 128,
        input_major=native_head_weights,
        row_tiles=rows // bm if bulk else 1,
    )
    rank = selection_phases(
        b,
        r,
        ctx,
        index_width,
        inputs={},
        output="Packed",
        chunk_guards=chunk_guards,
        # TopK can reuse OC after rollback normalization. The concurrent projection still reads QB.
        scratch=dict(Scores="Raw", LocalIds="OC"),
        concurrent=(project(uk, Latent3="QB", WV="UK", Value="MQ"),),
        guard=needs_score if score_rows else None,
    )
    sparse = paged_attention(
        b, r, main_pages, main_cols, query_count=6, index_width=index_width, score_rows_per_query=score_rows
    )
    value_tile = 2 * bm if rows >= 4 * bm and rows % (2 * bm) == 0 else bm
    phases = [
        phase(vector(input_edges(b, rows), Input="MQ", UnusedLatent="MY")),
        phase(
            project(
                partial_projection(rows, 7, 32, 128, rollback=True), p="OC", Value="OC", KGate="GateWeight", X="MQ"
            ),
            vector(output_quant(rows, 4096, 8, 256), Value="MQ", C3="AC", S3="AX"),
        ),
        phase(
            project(qa, C3="AC", S3="AX", WO="AW", OS="AWX", Y="V", nz=native_qkv_a),
            vector(pack_x(rows), HPX="QB", X="MQ", guard=needs_score),
        ),
        phase(
            project(dot(rows), HP="Panel", HPX="QB", guard=needs_score),
            vector(pool_norm(b, 7, rollback=True), PoolParts="OC", Current="Projected"),
            vector(qkv_norm(b, main_pages, split=split_qkv), CacheRows="MK", QKV="V"),
            project(partial_projection(rows, 8, 32, 128), KGate="GateWeight", X="MQ", p="MY", P="MY", Value="MY"),
        ),
        phase(
            project(
                head_gemm(rows, 1, 1536, 4096, bm, 128, 128, 2),
                Latent3="Norm",
                WV="IQuery",
                Value="Raw",
                guard=needs_score,
            ),
            vector(output_quant(rows, 1536, 8, 256), Value="Norm", C3="BC", S3="BX"),
            vector(pool_norm(b, 8), PoolParts="MY", Current="QKV"),
        ),
        phase(
            project(
                query_projection(rows, mx_tile_rows, bulk=bulk, native=native_q_b),
                WO="BW",
                OS="BWX",
                C3="BC",
                S3="BX",
                Y="QB",
                nz=native_q_b,
            ),
            vector(
                update_program(r, p, ring_pages, b, ring_table_stride=ring_table_stride), Starts="MS", Current="QKV"
            ),
            vector(native_finish(rows), IW="W", guard=needs_score),
            vector(quant_program(b), IQRaw="Raw", IQ="Q", IQS="QS", guard=needs_score),
            *cache,
        ),
        phase(
            mixed(score_program(r, b, p, cols, 6, 256, 4, 2, index_cols=index_cols), guard=needs_score),
            barrier=needs_score if score_rows else None,
        ),
        *rank,
        phase(
            mixed(
                sparse,
                Q="MQ",
                Cache="MK",
                Table="MT",
                Indices="Packed",
                Positions="Pos",
                Starts="MS",
                Y="MY",
                PreparedKV="Score",
            )
        ),
        phase(mixed(wuv_program(rows, value_tile, input_major=native_head_weights), Latent3="MY", C3="OC", S3="XS")),
        phase(*output),
    ]
    return compile_fused(abi, phases, name="glm53_sfa_mtp5_decode_kernel", controls=controls)
