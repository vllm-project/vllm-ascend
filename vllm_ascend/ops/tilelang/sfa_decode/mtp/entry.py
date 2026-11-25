# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Any

from tilelang import tvm
from tilelang.engine.lower import (
    device_codegen,
    device_codegen_without_compile,
    extrac_params,
    host_codegen,
    lower_to_host_device_ir,
)
from tilelang.engine.param import CompiledArtifact
from tilelang.jit.kernel import JITKernel
from tvm import tirx

from ..shared.index_query_quant_exact_div import program as quant_program
from ..shared.joint_phase_tir import emit_stage, grid_barrier, rewrite_host
from ..shared.tile_head_bmm_geometry import head_gemm
from ..shared.tile_keypool_partial_row import partial_projection
from ..shared.tile_mx_clamped_scale import quant_program as output_quant
from ..single.output_pipelined import banked_mx
from ..single.select_long_context import CHUNK_POOLS
from ..single.tile_banked_dense_projection import projection
from ..single.value_projection import project_and_quantize
from . import (
    DENSE_PREFIX_TOKENS,
    MIN_BULK_QUERY_TOKENS,
    PAGE_SIZE,
    Q6_MAX_CONTEXT,
    Q6_MAX_REQUESTS,
    TOKENS_PER_REQUEST,
    mtp_projection_rows,
)
from .dense_cache import prepare_dense_cache
from .exact_head import dot, pack_x
from .index_pool_norm import pool_norm
from .index_score import score_program
from .index_select import pack_dense_indices, pack_program, rank_long_chunks
from .live_cache_abi import abi_program
from .live_row_wuv import wuv_program
from .live_stride_ring import update_program
from .mtp_projection_slices import kg_slices, qa_slices
from .native_output import lower_native_weight_copy
from .pad_query_rows import prepare as input_edges
from .scaled_head import finish as native_finish
from .sparse_attention import DIM, large_main_program
from .split_main_norm import program as rms_scatter
from .split_state import state_norm
from .tile_o_l1_panel import panel_mx


def build(
    r,
    b,
    p,
    ctx,
    main_pages,
    main_cols,
    ring_pages,
    ring_table_stride=1,
    index_width=2176,
    index_cols=None,
    native_output=False,
    native_head_weights=False,
):
    cols = ctx // PAGE_SIZE
    index_cols = cols if index_cols is None else index_cols
    workers = 28
    nr = mtp_projection_rows(b)
    rows = nr
    if not 1 <= r <= Q6_MAX_REQUESTS or not 1 <= b <= r * TOKENS_PER_REQUEST:
        raise ValueError("Invalid request or token count for MTP5 decode")
    if not 0 < ctx <= Q6_MAX_CONTEXT or min(main_cols, index_cols) < 1:
        raise ValueError("Invalid context or page table width for MTP5 decode")
    if min(p, main_pages, ring_pages) < 1:
        raise ValueError("The index, main and ring page arenas must be nonempty")
    if ring_table_stride < 1:
        raise ValueError("Ring page IDs must have a positive row stride")
    bm = next(size for size in (96, 80, 64, 48, 32) if rows % size == 0)
    bulk = b >= MIN_BULK_QUERY_TOKENS
    split_topk = ctx == Q6_MAX_CONTEXT and (b <= 4 * TOKENS_PER_REQUEST or bulk)
    if index_width not in (2051, 2176):
        raise ValueError("Expected 2051 logical or 2176 padded token IDs per query")
    abi = abi_program(
        r,
        b,
        p,
        cols,
        6,
        256,
        4,
        2,
        main_pages,
        main_cols,
        ring_pages,
        ring_table_stride,
        index_width,
        index_cols,
        native_output=native_output,
        native_head_weights=native_head_weights,
    )

    class Fused(JITKernel):
        def _compile_artifact(self, func, pass_configs, compile_metadata):
            with (
                self.target,
                tvm.transform.PassContext(opt_level=3, config=pass_configs),
            ):
                host, carrier_mod, *_ = lower_to_host_device_ir(func, self.backend_context)
                carrier = next(iter(carrier_mod.functions.values()))
                params = {p.name: p for p in carrier.params}
                # Short queries leave Score unused. Its storage can hold one BF16 KV prefix per request.
                share_dense_cache = bulk and b * (ctx // 4) >= r * DENSE_PREFIX_TOKENS * DIM
                prepare_kv = []
                dense_attention = []
                attention_buffers = {
                    "Q": "MQ",
                    "Cache": "MK",
                    "Table": "MT",
                    "Indices": "Packed",
                    "Positions": "Pos",
                    "Starts": "MS",
                    "Y": "MY",
                    "PreparedKV": "Score",
                }
                if share_dense_cache:
                    prepare_kv.append(
                        (
                            "prepare_dense_cache",
                            prepare_dense_cache(r, main_pages, main_cols),
                            "VECTOR",
                            {"Cache": "MK", "Table": "MT", "PreparedKV": "Score"},
                        )
                    )
                    dense_attention.append(
                        (
                            "dense_qk_pv",
                            large_main_program(
                                main_cols * PAGE_SIZE,
                                6,
                                64,
                                r,
                                main_pages,
                                index_width=index_width,
                                tokens=b,
                                prepared_kv=True,
                            ),
                            "mixed",
                            attention_buffers,
                        )
                    )
                # The indexer skips short owners. Their score rows can hold BF16 KV while Cube projects Q.
                score_rows_per_query = ctx // 4 // DIM if share_dense_cache and ctx == Q6_MAX_CONTEXT else 0
                mixed_attention = []
                if score_rows_per_query:
                    prepare_kv.append(
                        (
                            "prepare_short_cache",
                            prepare_dense_cache(
                                r, main_pages, main_cols, tokens=b, score_rows_per_query=score_rows_per_query
                            ),
                            "VECTOR",
                            {"Cache": "MK", "Table": "MT", "PreparedKV": "Score"},
                        )
                    )
                    mixed_attention.append(
                        (
                            "mixed_qk_pv",
                            large_main_program(
                                main_cols * 128,
                                6,
                                64,
                                r,
                                main_pages,
                                index_width=index_width,
                                tokens=b,
                                score_rows_per_query=score_rows_per_query,
                            ),
                            "mixed",
                            attention_buffers,
                        )
                    )
                index_select = [
                    ("select", pack_program(b, r, ctx, index_width), "VECTOR", {"X": "Score", "Y": "Packed"})
                ]
                merge_select = []
                if split_topk:
                    # Quantization and W_UK have consumed Raw and QB before Top-K needs the shortlists.
                    aliases = {"Scores": "Raw", "LocalIds": "QB"}
                    index_select.append(
                        ("chunk_topk", rank_long_chunks(b, r, ctx // 4), "VECTOR", {"X": "Score", **aliases})
                    )
                    merge_select.append(
                        [
                            (
                                "merge_topk",
                                pack_program(b, r, ctx, index_width, merge_shortlists=True),
                                "VECTOR",
                                {"Y": "Packed", **aliases},
                            )
                        ]
                    )
                phases: list[list[tuple[str, Any, str, dict[str, str]]]] = [
                    [
                        (
                            "pad_hidden",
                            input_edges(b, nr),
                            "VECTOR",
                            {"Input": "MQ", "UnusedLatent": "MY"},
                        ),
                    ],
                    [
                        (
                            "key_gate_partial",
                            kg_slices(nr, rows),
                            "CUBE",
                            {"P": "OC", "X": "MQ"},
                        ),
                        (
                            "quant_hidden",
                            output_quant(nr, 4096, 8, 256, 5),
                            "VECTOR",
                            {"Value": "MQ", "C3": "AC", "S3": "AX"},
                        ),
                    ],
                    [
                        (
                            "qkv_a",
                            qa_slices(nr, rows, bm, 128, stages=2) if bulk else qa_slices(nr, rows),
                            "CUBE",
                            {"P": "V"},
                        ),
                        (
                            "pack_head_input",
                            pack_x(nr),
                            "VECTOR",
                            {"HPX": "QB", "X": "MQ"},
                        ),
                    ],
                    [
                        ("head_partial", dot(nr), "CUBE", {"HP": "Panel", "HPX": "QB"}),
                        (
                            "norm_and_cache",
                            rms_scatter(b, main_pages, 128, ctx, True, "perfnr", nr, rows),
                            "VECTOR",
                            {"CacheRows": "MK", "P": "V"},
                        ),
                        (
                            "key_pool_partial",
                            partial_projection(nr, 8, 32, 128),
                            "CUBE",
                            {
                                "KGate": "GateWeight",
                                "X": "MQ",
                                "p": "MY",
                                "P": "MY",
                                "Value": "MY",
                            },
                        ),
                    ],
                    [
                        (
                            "index_query",
                            head_gemm(nr, 1, 1536, 4096, bm, 128, 128, 2),
                            "CUBE",
                            {"Latent3": "Norm", "WV": "IQuery", "Value": "Raw"},
                        ),
                        (
                            "quant_q_lora",
                            output_quant(nr, 1536, 8, 256, 5),
                            "VECTOR",
                            {"Value": "Norm", "C3": "BC", "S3": "BX"},
                        ),
                        (
                            "normalize_pool",
                            pool_norm(b),
                            "VECTOR",
                            {"PoolParts": "MY", "Current": "QKV"},
                        ),
                    ],
                    [
                        (
                            "q_b",
                            projection(nr, 1536, 16384, bm, 256, 512, 128, True)
                            if bulk
                            else panel_mx(nr, 1536, 16384, bm, 256, 2, 4096, 512),
                            "CUBE",
                            {
                                "WO": "BW",
                                "OS": "BWX",
                                "C3": "BC",
                                "S3": "BX",
                                "Y": "QB",
                            },
                        ),
                        (
                            "normalize_key_gate",
                            state_norm(b, nr, rows),
                            "VECTOR",
                            {"P": "OC"},
                        ),
                        *prepare_kv,
                    ],
                    [
                        (
                            "w_uk",
                            head_gemm(
                                rows,
                                64,
                                256,
                                512,
                                bm,
                                256,
                                128,
                                2,
                                l1_k=256,
                                input_major=native_head_weights,
                                row_tiles=2 if rows % (2 * bm) == 0 else 1,
                            )
                            if bulk
                            else head_gemm(rows, 64, 256, 512, bm, 128, 128, 2, input_major=native_head_weights),
                            "CUBE",
                            {"Latent3": "QB", "WV": "UK", "Value": "MQ"},
                        ),
                        (
                            "update_keypool",
                            update_program(r, p, ring_pages, b, ring_table_stride=ring_table_stride),
                            "VECTOR",
                            {"Starts": "MS", "Current": "QKV"},
                        ),
                        ("head_reduce", native_finish(nr), "VECTOR", {"IW": "W"}),
                        (
                            "quant_index_query",
                            quant_program(b, p, 128, ctx, False),
                            "VECTOR",
                            {"IQRaw": "Raw", "IQ": "Q", "IQS": "QS"},
                        ),
                    ],
                    [
                        (
                            "score",
                            score_program(r, b, p, cols, 6, 256, 4, 2, index_cols=index_cols),
                            "mixed",
                            {},
                        ),
                    ],
                    index_select,
                    *merge_select,
                    [
                        (
                            "sparse_qk_pv",
                            large_main_program(
                                main_cols * 128,
                                6,
                                64,
                                r,
                                main_pages,
                                index_width=index_width,
                                tokens=b,
                            ),
                            "mixed",
                            attention_buffers,
                        ),
                        *mixed_attention,
                        *dense_attention,
                    ],
                    [
                        (
                            "w_uv_mx",
                            project_and_quantize(rows, bm, input_major=native_head_weights)
                            if bulk
                            else wuv_program(rows, bm, 256, input_major=native_head_weights),
                            "mixed",
                            {"Latent3": "MY", "C3": "OC", "S3": "XS"},
                        ),
                    ],
                ]
                output_map = {"C3": "OC", "S3": "XS", "Y": "Final"}
                output_region = []
                full_groups, last_rows = divmod(b, bm)
                for count, edge in ((full_groups, False), (int(last_rows > 0), True)):
                    if count:
                        output_region.append(
                            (
                                "out_tail" if edge else "out",
                                banked_mx(
                                    rows, 16384, 4096, bm, 256, 512, 128, "nested", b, native_output=native_output
                                )
                                if bulk and not edge
                                else panel_mx(
                                    rows, 16384, 4096, bm, 256, 2, 4096, 512, b, edge, native_output=native_output
                                ),
                                "CUBE",
                                output_map,
                            )
                        )
                if share_dense_cache:
                    # Dense attention reads contiguous KV. Later layers consume the indices after output projection.
                    output_region.append(
                        ("dense_indices", pack_dense_indices(b, r, index_width), "VECTOR", {"Y": "Packed"})
                    )
                phases.append(output_region)
                block = tirx.Var("worker", "int32")
                thread = tirx.IterVar(
                    tvm.ir.Range.from_min_extent(0, workers),
                    block,
                    tirx.IterVar.ThreadIndex,
                    "blockIdx.x",
                )
                pieces = []
                ub = 0
                score_needed = tirx.Var("score_needed", "bool")
                parallel_topk = tirx.Var("parallel_topk", "bool")
                for phase_id, regions in enumerate(phases):
                    for stage_name, program, engine, aliases in regions:
                        _, mod, *_ = lower_to_host_device_ir(program, self.backend_context)
                        stage = next(iter(mod.functions.values()))
                        if native_output and stage_name in ("out", "out_tail"):
                            stage = lower_native_weight_copy(stage)
                        work = int(stage.attrs["thread_extent"]["blockIdx.x"])
                        bound = {
                            **params,
                            **{name: params[target] for name, target in aliases.items()},
                        }
                        step = emit_stage(stage, bound, block, workers, work, engine, independent_tiles=bulk)
                        if bulk and stage_name in (
                            "pack_head_input",
                            "head_partial",
                            "index_query",
                            "head_reduce",
                            "quant_index_query",
                            "score",
                        ):
                            step = tirx.IfThenElse(score_needed, step, None)
                        if share_dense_cache:
                            if stage_name in ("prepare_dense_cache", "dense_qk_pv", "dense_indices"):
                                step = tirx.IfThenElse(tirx.Not(score_needed), step, None)
                            elif score_rows_per_query and stage_name in ("prepare_short_cache", "mixed_qk_pv"):
                                step = tirx.IfThenElse(parallel_topk, step, None)
                            elif score_rows_per_query and stage_name == "sparse_qk_pv":
                                step = tirx.IfThenElse(tirx.And(score_needed, tirx.Not(parallel_topk)), step, None)
                            elif stage_name in ("sparse_qk_pv", "select"):
                                step = tirx.IfThenElse(score_needed, step, None)
                        if split_topk:
                            if stage_name in ("chunk_topk", "merge_topk"):
                                step = tirx.IfThenElse(parallel_topk, step, None)
                            elif stage_name == "select":
                                step = tirx.IfThenElse(tirx.Not(parallel_topk), step, None)
                        pieces.append(step)
                        ub = max(ub, int(stage.attrs.get("dyn_shared_memory_buf", 0)))
                    if phase_id + 1 < len(phases):
                        barrier = grid_barrier()
                        if share_dense_cache and regions[0][0] in ("score", "select"):
                            barrier = tirx.IfThenElse(score_needed, barrier, None)
                        if split_topk and regions is index_select:
                            barrier = tirx.IfThenElse(parallel_topk, barrier, None)
                        pieces.append(barrier)
                body = tirx.SeqStmt(pieces)
                if bulk:
                    lengths = tirx.decl_buffer((r,), "int32", data=params["Length"])
                    # Up to 512 complete pools fit in the selection. The indexer cache still advances.
                    needs_topk = tirx.any(*[lengths[i] >= DENSE_PREFIX_TOKENS for i in range(r)])
                    flags = [tirx.Bind(score_needed, needs_topk)]
                    if split_topk:
                        num_sparse = sum(tirx.Cast("int32", lengths[i] >= DENSE_PREFIX_TOKENS) for i in range(r))
                        has_long = tirx.any(*[lengths[i] > 4 * CHUNK_POOLS for i in range(r)])
                        flags.append(
                            tirx.Bind(parallel_topk, tirx.And(has_long, num_sparse * TOKENS_PER_REQUEST <= 2 * workers))
                        )
                    body = tirx.SeqStmt([*flags, *pieces])
                elif split_topk:
                    lengths = tirx.decl_buffer((r,), "int32", data=params["Length"])
                    long_prefix = tirx.any(*[lengths[i] > 4 * CHUNK_POOLS for i in range(r)])
                    body = tirx.SeqStmt([tirx.Bind(parallel_topk, long_prefix), *pieces])
                fn = carrier.with_body(tirx.AttrStmt(thread, "thread_extent", workers, body))
                fn = fn.with_attr("dyn_shared_memory_buf", tirx.IntImm("int64", ub))
                kernel_name = "glm53_sfa_mtp5_decode_kernel"
                dev = tvm.IRModule({kernel_name: fn.with_attr("global_symbol", kernel_name)})
                host = rewrite_host(
                    host,
                    workers,
                    workers,
                    int(carrier.attrs["dyn_shared_memory_buf"]),
                    ub,
                    kernel_name=kernel_name,
                )
                code = device_codegen_without_compile(dev, self.backend_context).inspect_source()
                assert code.count('extern "C" __global__') == 1
                binary = device_codegen(dev, self.backend_context)
                ready = host_codegen(host, self.backend_context)
                ready.import_module(binary)
                return CompiledArtifact(
                    ready,
                    dev,
                    extrac_params(func),
                    code,
                    rt_mod=ready,
                    target=self.target,
                    target_host=self.target_host,
                )

    return Fused(func=abi, target="ascend", execution_backend="tvm_ffi")
