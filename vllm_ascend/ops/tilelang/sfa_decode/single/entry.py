# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite

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

from ..mtp.mtp_projection_slices import qa_slices
from ..mtp.pad_query_rows import prepare as prepare_rows
from ..mtp.split_main_norm import program as reduce_qkv_and_norm
from ..mtp.tile_o_l1_panel import panel_mx
from ..shared.index_query_quant_exact_div import program as quant_index
from ..shared.joint_phase_tir import emit_stage, grid_barrier, rewrite_host
from ..shared.tile_head_bmm_geometry import head_gemm
from ..shared.tile_keypool_partial_row import partial_projection
from ..shared.tile_mx_clamped_scale import quant_program as quant
from . import (
    DECODE_CONTEXT_BUCKETS,
    INDEX_COVERAGE_SCALE,
    INDEX_VOTE_STRIDE,
    MAX_PACKED_OUTPUT_REQUESTS,
    MAX_RESIDENT_REQUESTS,
    SHARED_INDEX_WEIGHT,
    head_panel_layout,
    projection_rows,
    use_four_request_index_scan,
    use_segmented_index_scan,
)
from .expand_selected import expand_selected, mask_graph_lengths
from .index_score import reset_index_count, score_program, select_index_schedule
from .keypool_cache_update import state_update
from .main_kv_norm import program as norm
from .native_split_head_tile import stages as split_head_stages
from .output_interleaved import banked_mx as interleaved_output
from .output_pipelined import banked_mx as nested_output
from .parallel_projection import mx_quant
from .resident_sparse import merge_value_tiles
from .select_long_context import CHUNK_POOLS, select_long_context
from .select_pool_radix_fused import select_program
from .select_short_context import select_program as select_short
from .single_abi import full_decode_program
from .sparse_attention import main_from_pool_ids as attend_short_tail
from .tile_banked_dense_projection import projection
from .tile_head_chunk_pack import pack
from .tile_index_weights_local import local_head
from .tile_mx_interleaved import quant_program as quant_output
from .value_projection import project_and_quantize

CUBE = "CUBE"

INDEPENDENT_CUBE = "independent_cube_tiles"

VECTOR = "VECTOR"

MIXED = "mixed"

CORES = 28

RESIDENT_MIN_LENGTH = 768
RESIDENT_BATCH_MIN_LENGTH = 2048


@dataclass(frozen=True)
class _IndexScoreDispatch:
    offset: int
    minimum: int
    regular: tuple
    ordinary: tuple
    shared: tuple
    fragmented: tuple
    paired: tuple | None
    unpacked_fragmented: tuple | None
    partial_rows: tuple | None


@dataclass(frozen=True)
class _ScoreAndQueryProjection:
    score: tuple
    projection: tuple
    selection: tuple


@dataclass(frozen=True)
class _ResidentAttention:
    serial: tuple
    resident: tuple
    merge: tuple


def _phases(
    batch,
    pages,
    context,
    index_width,
    prepacked_heads,
    index_pages,
    table_columns,
    tail_layout=None,
    index_norm_eps=1e-5,
    mask_padding=False,
    resident_workspace=False,
    packed_output=False,
):
    native_rows = (batch + 31) // 32 * 32
    b = projection_rows(batch, prepacked_heads)
    shape = (batch, pages, 128, context, True)
    hidden = "X" if batch == b else "IQRaw"
    small = native_rows <= 512
    m = next(tile for tile in (128, 96, 64, 32) if b % tile == 0)
    aq = quant(b, 4096, 16, 1024, 5) if small else quant(b, 4096, 32, 512, 5)
    bq = mx_quant(b, 1536) if small else quant(b, 1536, 16, 512, 5)
    length_offset = 32 * b if mask_padding else 0
    length_buffer = "S1" if mask_padding else "Length"
    # Each int32 can address 2**26 pages with 32 pools per page.
    translate_pools = not small and context == 16384 and pages <= 1 << 26
    if context // 4 > CHUNK_POOLS:
        selector = select_long_context(batch, context, length_offset)
    elif small and context == 16384:
        selector = select_short(*shape, length_offset=length_offset)
    else:
        selector = select_program(
            *shape,
            True,
            length_offset=length_offset,
            physical_table_columns=table_columns[0] if translate_pools else None,
        )
    output = interleaved_output if small else nested_output
    head_rows = 32 if small else 16
    bf16_rows = 16 if batch <= 16 else b
    bf16_tile = min(m, bf16_rows)
    full_k_panel = 1 < batch <= 16
    tail_rows = bf16_tile
    if batch > 256:
        used = (batch - 1) % bf16_tile + 1
        rounded = (used + 15) // 16 * 16
        if bf16_tile - rounded >= 32:
            tail_rows = rounded
    uk_row_tiles = 1
    if batch > 128 and tail_rows == bf16_tile:
        uk_row_tiles = next(span for span in (4, 2, 1) if bf16_rows % (span * bf16_tile) == 0)
    uk_panel = 256 if full_k_panel or uk_row_tiles > 1 else 128
    aqu = (aq, VECTOR, dict(Value=hidden, C3="C1", S3="S1"))
    # The 950 BF16-output MX matmul partitions K by the unpadded M dimension.
    if 23 <= batch <= 160:
        # Keep its 1536/1536/1024 partials in the query arena while Vector updates KeyPool.
        a = (qa_slices(b, b), CUBE, dict(AC="C1", AX="S1", AW="A", AWX="AS", P="QA", V="QA"))
        rms = (reduce_qkv_and_norm(*shape, "perfnr", b, b), VECTOR, dict(P="QA", V="QA"))
    else:
        a = (
            projection(b, 4096, 2048, m, 256, 512, 128, False, column_tiles=CORES if batch < 23 else None),
            INDEPENDENT_CUBE,
            dict(C3="C1", S3="S1", WO="A", OS="AS", Y="QKV"),
        )
        rms = (norm(*shape, "perfnr"), VECTOR, {})
    bqu = (bq, VECTOR, dict(Value="Norm", C3="C2", S3="S2"))
    qb = (
        projection(b, 1536, 16384, m, 256, 512, 128, not small, column_tiles=3 * CORES if b == 32 else None),
        INDEPENDENT_CUBE,
        dict(C3="C2", S3="S2", WO="B", OS="BS", Y="QB"),
    )
    uk = (
        head_gemm(
            bf16_rows, 64, 256, 512, bf16_tile, 256, 64, 2, l1_k=uk_panel, tail_rows=tail_rows, row_tiles=uk_row_tiles
        ),
        INDEPENDENT_CUBE,
        dict(Latent3="QB", WV="WK", Value="Q3"),
    )
    key_parts = 8 if native_rows <= 256 else 28
    key = (partial_projection(bf16_rows, key_parts, bf16_tile, 128), CUBE, dict(p="Value", X=hidden))
    split_descriptors = use_segmented_index_scan(batch, context) and batch < 512
    cache_stages = state_update(
        batch,
        index_pages,
        128,
        context,
        True,
        key_parts,
        table_columns[1],
        tail_layout=tail_layout,
        index_norm_eps=index_norm_eps,
        length_offset=length_offset,
        split_descriptors=split_descriptors,
    )
    # The index query projection only needs Cube while Vector inspects the page table.
    parallel_descriptors = ()
    if split_descriptors:
        cache_stages, descriptors = cache_stages
        parallel_descriptors = ((descriptors, VECTOR, dict(ITable="IT", Len=length_buffer, Runs="C3")),)
    state = (
        cache_stages,
        VECTOR,
        dict(
            partial="Value",
            Weight="IG",
            Bias="IB",
            Len=length_buffer,
            TailTable="TailPage",
            ITable="IT",
            K="IK",
            KS="IKS",
            Runs="C3",
        ),
    )
    iq = (
        head_gemm(bf16_rows, 1, 1536, 4096, bf16_tile, 128, 128, 2, l1_k=256, tail_rows=tail_rows),
        INDEPENDENT_CUBE,
        dict(Latent3="Norm", WV="IQuery", Value="IQRaw"),
    )
    weight_buffer = "IWeight" if prepacked_heads else "Value"
    head_pack = (pack(b, prepacked_heads), VECTOR, dict(Panel="Value", X=hidden))
    head_weights = (
        local_head(b, head_rows, 256 if 704 <= native_rows <= 896 else 128, prepacked_heads=prepacked_heads),
        MIXED,
        dict(Panel="Value", HeadPanel=weight_buffer),
    )
    head_reduce = ()
    if native_rows <= 448:
        parts, head_width, aligned = head_panel_layout(batch)
        rows = b if prepacked_heads else b + 32
        assert parts * (rows * aligned + b * 32) * 4 <= b * 32768 * 2
        panel, cube_head, reduce_head = split_head_stages(b, head_width, prepacked_heads=prepacked_heads)
        head_pack = (panel, VECTOR, dict(Panel="Latent", X=hidden))
        head_weights = (
            cube_head,
            CUBE,
            dict(Panel="Latent", Parts="Latent", HeadPanel="IWeight" if prepacked_heads else "Latent"),
        )
        head_reduce = ((reduce_head, VECTOR, dict(Parts="Latent")),)
    iq_quant = (quant_index(*shape), VECTOR, {})
    score_inputs = dict(
        Q="IQ",
        QS="IQS",
        K="IK",
        KS="IKS",
        Table="IT",
        W="IW",
        Runs="C3",
        bulk="IK",
        halves="IK",
        quart="IK",
        Length=length_buffer,
    )
    score_modes = (False, True) if use_segmented_index_scan(batch, context) else (False,)
    score_phase = tuple(
        (
            score_program(
                batch,
                index_pages,
                context,
                table_columns=table_columns[1],
                length_offset=length_offset,
                fragmented_only=fragmented_only,
            ),
            MIXED,
            score_inputs,
        )
        for fragmented_only in score_modes
    )
    reset_count = ()
    head_phase = (head_weights,)
    if use_segmented_index_scan(batch, context):
        # The count uses the existing grid barrier between cache updates and index scoring.
        count_offset = batch * (context // 512 + context // 16384)
        if split_descriptors:
            reset_count = ((reset_index_count(batch, context), VECTOR, dict(Mode="C3")),)
            minimum = (batch * (context // 16384) + 3) // 4
        else:
            count_offset += batch
            minimum = 1
            summary = select_index_schedule(batch, context)
            head_phase = (head_weights, (summary, VECTOR, dict(Mode="C3")))
        regular = score_program(
            batch,
            index_pages,
            context,
            table_columns=table_columns[1],
            length_offset=length_offset,
            allow_specialized=False,
        )
        four_requests = use_four_request_index_scan(batch, context)
        shared = score_program(
            batch,
            index_pages,
            context,
            table_columns=table_columns[1],
            length_offset=length_offset,
            shared_only=True,
            quartets=four_requests,
        )
        paired = None
        if four_requests:
            pair_program = score_program(
                batch,
                index_pages,
                context,
                table_columns=table_columns[1],
                length_offset=length_offset,
                shared_only=True,
            )
            paired = (pair_program, MIXED, score_inputs)
        ordinary, fragmented = score_phase
        partial_rows = None
        if four_requests:
            partial_rows = (
                score_program(
                    batch,
                    index_pages,
                    context,
                    table_columns=table_columns[1],
                    length_offset=length_offset,
                    partial_rows_only=True,
                ),
                MIXED,
                score_inputs,
            )
        unpacked_fragmented = None
        if batch >= 512 and batch * (context // 16384) % CORES:
            unpacked_fragmented = (
                score_program(
                    batch,
                    index_pages,
                    context,
                    table_columns=table_columns[1],
                    length_offset=length_offset,
                    fragmented_only=True,
                    allow_paired_loads=False,
                ),
                MIXED,
                score_inputs,
            )
        score_phase = _IndexScoreDispatch(
            count_offset,
            minimum * INDEX_VOTE_STRIDE,
            (regular, MIXED, score_inputs),
            ordinary,
            (shared, MIXED, score_inputs),
            fragmented,
            paired,
            unpacked_fragmented,
            partial_rows,
        )
    compact_ids = "Selected" if index_width == 512 else "IQRaw"
    select_args = dict(X="Score", L=length_buffer, Y=compact_ids)
    if translate_pools:
        # Index scoring has consumed the quantized hidden-state buffer.
        select_args["Physical"] = "C1"
    select = (selector, VECTOR, select_args)
    layout = dict(table_columns=table_columns[0], length_offset=length_offset)
    attn_inputs = dict(Q="Q3", Indices="C1" if translate_pools else compact_ids, Length=length_buffer)
    attn = (
        attend_short_tail(*shape, **layout, physical_pool_ids=translate_pools),
        MIXED,
        dict(**attn_inputs, Y="Latent3"),
    )
    if batch == 1 or resident_workspace and batch <= MAX_RESIDENT_REQUESTS:
        # All QK tiles finish before the resident stage reuses the padded query arena for FP32 PV partials.
        intermediates = dict(Partial="QA", TailValue="Value", Stats="C1")
        resident = attend_short_tail(*shape, **layout, resident_cores=CORES)
        attention_phase = _ResidentAttention(
            attn,
            (resident, MIXED, dict(**attn_inputs, **intermediates, Maxima="C3")),
            (merge_value_tiles(batch, length_offset), VECTOR, dict(**intermediates, Length=length_buffer, Y="Latent3")),
        )
    else:
        attention_phase = (attn,)
    if batch <= 16:
        value_stages = [((project_and_quantize(b, b, balanced=True), MIXED, {}),)]
    elif b >= 2 * m and tail_rows == m:
        value_stages = [((project_and_quantize(b, m), MIXED, {}),)]
    else:
        uv = (
            head_gemm(
                bf16_rows, 64, 512, 256, bf16_tile, 256, 64, 2, l1_k=512 if full_k_panel else 64, tail_rows=tail_rows
            ),
            INDEPENDENT_CUBE,
            {},
        )
        oq = quant_output(b, 16384, 32, 512, 6) if small else quant_output(b, 16384, 16, 1024, 6)
        value_stages = [(uv,), ((oq, VECTOR, {}),)]
    epilogue = ()
    if batch >= m:
        if batch == 32:
            o = panel_mx(b, 16384, 4096, m, 128, 2, 4096, 1024, uniform_n=True)
        else:
            o = output(b, 16384, 4096, m, 256, 512, 128, "both" if small else "nested", batch)
        epilogue = ((o, INDEPENDENT_CUBE, {}),)
    prologue = []
    if batch != b:
        # IQRaw becomes the index-query output only after all hidden-state readers finish.
        prologue.append(((prepare_rows(batch, b), VECTOR, dict(Input="IQRaw", UnusedLatent="Latent")),))
        tail_panel = 1024 if b == 32 else 512
        tail_output = panel_mx(
            b, 16384, 4096, m, 256, 2, 4096, tail_panel, batch, True, uniform_n=b == 32, packed_output=packed_output
        )
        epilogue += ((tail_output, INDEPENDENT_CUBE, {}),)
    if index_width != 512:
        # Sparse attention consumes compact pool IDs. Publish the expanded index cache
        # from the other engine while Cube computes the final projection.
        epilogue += (
            (
                expand_selected(batch, index_width, length_offset=length_offset),
                VECTOR,
                dict(Pools=compact_ids, Packed="Selected", Length=length_buffer),
            ),
        )
    if batch <= 4 or (batch <= 16 and context == 16384):
        (score,) = score_phase
        index_and_query = (_ScoreAndQueryProjection(score, uk, select),)
    else:
        index_and_query = (score_phase, (uk, select))
    # state consumes the KeyPool partials before head_pack reuses the same buffer.
    length_mask = ((mask_graph_lengths(batch, length_offset), VECTOR, dict(Effective="S1")),) if mask_padding else ()
    return [
        *prologue,
        (key, aqu, *length_mask, *reset_count),
        (state, a),
        (rms, head_pack),
        (iq, bqu, *parallel_descriptors),
        head_phase,
        (qb, iq_quant, *head_reduce),
        *index_and_query,
        attention_phase,
        *value_stages,
        epilogue,
    ]


@lru_cache(maxsize=16)
def compile_decode(
    batch,
    pages,
    context,
    *,
    index_width=512,
    prepacked_heads=False,
    index_pages=None,
    table_columns=None,
    tail_layout=None,
    index_norm_eps=1e-5,
    mask_padding=False,
    resident_workspace=False,
    packed_output=False,
):
    """Return a graph-capturable JIT kernel. The last ABI argument is the caller's output buffer."""
    index_pages = pages if index_pages is None else index_pages
    table_columns = (context // 128, context // 128) if table_columns is None else table_columns
    if not 1 <= batch <= 1024 or context not in DECODE_CONTEXT_BUCKETS or min(pages, index_pages) < 1:
        raise ValueError(f"Expected Q1 batch 1 to 1024, context in {DECODE_CONTEXT_BUCKETS} and a nonempty page arena")
    if packed_output and batch > MAX_PACKED_OUTPUT_REQUESTS:
        raise ValueError("K-panel output weights require a small one-query decode bucket")
    if index_width not in (512, 2051, 2176):
        raise ValueError("Index output must contain 512 pool IDs, 2051 token IDs, or 2176 padded token IDs")
    if len(table_columns) != 2 or min(table_columns) < context // 128:
        raise ValueError("Main and index block tables must each cover the compiled context")
    tail_layout = (batch, 4 * 256, 1) if tail_layout is None else tail_layout
    if len(tail_layout) != 3 or tail_layout[0] < 1 or tail_layout[1] < 4 * 256 or tail_layout[2] < 1:
        raise ValueError("Expected tail page count, FP32 page stride and page-ID stride")
    if not isfinite(index_norm_eps) or index_norm_eps <= 0:
        raise ValueError("Expected a finite positive KeyPool normalization epsilon")
    carrier = full_decode_program(
        batch,
        pages,
        128,
        context,
        True,
        index_width=index_width,
        prepacked_heads=prepacked_heads,
        index_pages=index_pages,
        table_columns=table_columns,
        tail_layout=tail_layout,
        mask_padding=mask_padding,
        resident_workspace=resident_workspace,
        packed_output=packed_output,
    )
    phases = _phases(
        batch,
        pages,
        context,
        index_width,
        prepacked_heads,
        index_pages,
        table_columns,
        tail_layout,
        index_norm_eps,
        mask_padding,
        resident_workspace,
        packed_output,
    )

    class DecodeKernel(JITKernel):
        def _compile_artifact(self, func, pass_configs, compile_metadata):
            with self.target, tvm.transform.PassContext(opt_level=3, config=pass_configs):
                host, device, *_ = lower_to_host_device_ir(func, self.backend_context)
                abi = next(iter(device.functions.values()))
                pointers = {p.name: p for p in abi.params}
                for base, view in (("Cache", "CacheRows"), ("QA", "Q3"), ("Latent", "Latent3")):
                    ptr = pointers[base] if base in pointers else pointers[view]
                    pointers[base] = pointers[view] = ptr
                block = tirx.Var("group", "int32")
                grid = tirx.IterVar(
                    tvm.ir.Range.from_min_extent(0, CORES), block, tirx.IterVar.ThreadIndex, "blockIdx.x"
                )
                pieces = []
                ub = 0

                def lower_region(spec):
                    nonlocal ub
                    program, _, names = spec
                    _, region, *_ = lower_to_host_device_ir(program, self.backend_context)
                    stage = next(iter(region.functions.values()))
                    bound = {**pointers, **{a: pointers[b] for a, b in names.items()}}
                    ub = max(ub, int(stage.attrs.get("dyn_shared_memory_buf", 0)))
                    return stage, bound

                def lower_stage(spec):
                    stage, bound = lower_region(spec)
                    kind = spec[1]
                    work = int(stage.attrs["thread_extent"]["blockIdx.x"])
                    independent = kind == INDEPENDENT_CUBE
                    return emit_stage(
                        stage,
                        bound,
                        block,
                        CORES,
                        work,
                        CUBE if independent else kind,
                        independent_tiles=independent and batch >= 128,
                    )

                for index, phase in enumerate(phases):
                    if isinstance(phase, _ScoreAndQueryProjection):
                        score, score_args = lower_region(phase.score)
                        query, query_args = lower_region(phase.projection)
                        score_cores = int(score.attrs["thread_extent"]["blockIdx.x"])
                        query_tiles = int(query.attrs["thread_extent"]["blockIdx.x"])
                        tail = query_tiles % CORES if score_cores < CORES else 0
                        # Keep the main wave intact when the spare Cubes would need over four waves.
                        if batch > 4 and tail > 4 * (CORES - score_cores):
                            tail = 0
                        pieces.append(emit_stage(score, score_args, block, CORES, score_cores, MIXED))
                        if tail:
                            # The score grid uses only the first core pairs. Its idle Cubes
                            # can finish the last partial wave of the independent query projection.
                            # Each emitted TIR region needs its own buffer definitions.
                            early_query, early_args = lower_region(phase.projection)
                            early = emit_stage(
                                early_query,
                                early_args,
                                block - score_cores,
                                CORES - score_cores,
                                tail,
                                CUBE,
                                first_tile=query_tiles - tail,
                                independent_tiles=True,
                            )
                            pieces.append(tirx.IfThenElse(block >= score_cores, early, None))
                        pieces.append(grid_barrier(pair_at_entry=False))
                        pieces.append(
                            emit_stage(
                                query, query_args, block, CORES, query_tiles - tail, CUBE, independent_tiles=True
                            )
                        )
                        pieces.append(lower_stage(phase.selection))
                    elif isinstance(phase, _IndexScoreDispatch):
                        mode = tirx.decl_buffer(
                            (phase.offset + 1 + int(phase.paired is not None) + int(phase.partial_rows is not None),),
                            "int32",
                            data=pointers["C3"],
                        )
                        count = mode[phase.offset]
                        # The three scans write disjoint score segments and only rendezvous before selection.
                        shared_body = lower_stage(phase.shared)
                        if phase.paired is not None:
                            shared_body = tirx.IfThenElse(
                                mode[phase.offset + 1] != 0, shared_body, lower_stage(phase.paired)
                            )
                        shared = tirx.IfThenElse(
                            (count & (INDEX_VOTE_STRIDE - 1)) != 0,
                            shared_body,
                            None,
                        )
                        ordinary = lower_stage(phase.ordinary)
                        # With every full descriptor covered, one task per row can visit the remaining scores.
                        if phase.partial_rows is not None:
                            ordinary = tirx.IfThenElse(
                                mode[phase.offset + 2] == 0, lower_stage(phase.partial_rows), ordinary
                            )
                        fragmented = lower_stage(phase.fragmented)
                        if phase.unpacked_fragmented is not None:
                            shared_count = count & (INDEX_VOTE_STRIDE - 1)
                            # The high bits contain coverage-weighted votes for both private and shared pages.
                            private_votes = (
                                count // INDEX_VOTE_STRIDE - INDEX_COVERAGE_SCALE * SHARED_INDEX_WEIGHT * shared_count
                            )
                            fragmented = tirx.IfThenElse(
                                private_votes >= batch * (context // 16384),
                                fragmented,
                                lower_stage(phase.unpacked_fragmented),
                            )
                        pieces.append(
                            tirx.IfThenElse(
                                count >= phase.minimum,
                                tirx.SeqStmt([shared, ordinary, fragmented]),
                                lower_stage(phase.regular),
                            )
                        )
                    elif isinstance(phase, _ResidentAttention):
                        lengths = tirx.decl_buffer((batch,), "int32", data=pointers["Length"])
                        length = lengths[0]
                        for row in range(1, batch):
                            length = tirx.Min(length, lengths[row])
                        minimum = RESIDENT_MIN_LENGTH if batch == 1 else RESIDENT_BATCH_MIN_LENGTH
                        resident = tirx.SeqStmt(
                            [lower_stage(phase.resident), grid_barrier(pair_at_entry=False), lower_stage(phase.merge)]
                        )
                        pieces.append(tirx.IfThenElse(length >= minimum, resident, lower_stage(phase.serial)))
                    else:
                        pieces.extend(lower_stage(spec) for spec in phase)
                    if index + 1 < len(phases):
                        # Each engine drains before its own rendezvous. Pair them once on exit.
                        pieces.append(grid_barrier(pair_at_entry=False))
                body = tirx.AttrStmt(grid, "thread_extent", CORES, tirx.SeqStmt(pieces))
                kernel = (
                    abi.with_body(body)
                    .with_attr("thread_extent", {"blockIdx.x": tirx.IntImm("int32", CORES)})
                    .with_attr("dyn_shared_memory_buf", tirx.IntImm("int64", ub))
                )
                dev = tvm.IRModule({"main_kernel": kernel})
                host = rewrite_host(
                    host,
                    projection_rows(batch, prepacked_heads) // 32,
                    CORES,
                    int(abi.attrs["dyn_shared_memory_buf"]),
                    ub,
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

    return DecodeKernel(func=carrier, target="ascend", execution_backend="tvm_ffi")
