# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate the A3 Cube projection and Vector recurrence used by model Verify."""

from hashlib import sha256

import tilelang
from tilelang import language as T

from ._model_epilogues import (
    NORM_BINDINGS,
    NORM_BUFFERS,
    NORM_FP32_BINDINGS,
    PACK_BINDINGS,
    PACK_BUFFERS,
    PACK_JOB,
    VECTOR_RSTD_BUFFERS,
    VECTOR_RSTD_INIT,
    norm_apply,
)


def build_pipeline(
    rows,
    pages,
    state_stride,
    replay_stride,
    index_stride,
    value_stride,
    *,
    depth,
    variant="original",
    gate_stride=48,
    token_count=None,
    pack_inside=False,
    prefetch_state=False,
    directional_fix_sync=None,
    directional_wave_end=False,
    q_stride=1024,
    k_stride=1024,
    z_stride=3072,
    norm_eps=None,
    norm_activation="silu",
    norm_weight_fp32=False,
    skip_inactive=False,
    shared_pages=False,
    projection_rows=16,
    shared_replay_keys=False,
    directional_input_sync=False,
    split_checkpoint=False,
    dense_qk=False,
    dense_qk_once=False,
    grouped_replay=False,
    vector_rstd=False,
    steady_aic=False,
    address_mode="base",
):
    if variant not in ("original", "balanced", "axpy", "inline", "batched", "model"):
        raise ValueError(f"unsupported AIV variant {variant!r}")
    inline = variant == "inline"
    model = variant == "model"
    fused_norm = norm_eps is not None
    norm_weight_dtype = "float32" if norm_weight_fp32 else "bfloat16"
    if (pack_inside or prefetch_state or fused_norm) and not model:
        raise ValueError("The V3 options require the model ABI")
    if shared_pages and not (skip_inactive and model and not pack_inside):
        raise ValueError("Shared pages require the external model Pack and inactive-row skipping")
    if projection_rows not in (8, 16):
        raise ValueError("Projection Fixpipe supports 8 or 16 output rows")
    if shared_replay_keys and not model:
        raise ValueError("Shared replay K requires the A3 model layout")
    if grouped_replay and not shared_replay_keys:
        raise ValueError("Grouped FP32 replay requires a single K owner")
    if dense_qk and (pack_inside or not model):
        raise ValueError("Dense Q/K requires the external model Pack")
    if dense_qk_once and not dense_qk:
        raise ValueError("Once-per-core Q/K initialization requires dense Pack")
    if split_checkpoint and not (model and prefetch_state):
        raise ValueError("Split checkpoint requires model state prefetch")
    if vector_rstd and not fused_norm:
        raise ValueError("Vector rstd requires fused norm")
    if steady_aic and (rows != 32 or depth != 2 or not prefetch_state or not directional_input_sync):
        raise ValueError("Steady AIC phases require the B32 depth-2 directional prefetch schedule")
    packed_rows = 8 if dense_qk else 16
    first_v = 256 if split_checkpoint else 384
    key_ub = 136 if grouped_replay else 128
    decay_ub = 128 if grouped_replay else 256
    key_load = f"G2U(keys,kg[job*{packed_rows * 128}],1,(end-begin)*128,(end-begin)*128,bfloat16_t);"
    if shared_replay_keys:
        key_load = f"if (first == 0) {{ {key_load} }}"
    use_address_table = address_mode in ("pack", "cached")
    hoist_address = address_mode == "hoist"
    shift_address = address_mode == "shift"
    if address_mode not in ("base", "shift", "hoist", "pack", "cached", "replay_i32"):
        raise ValueError(f"Unknown Verify address mode {address_mode!r}")
    if address_mode != "base" and not (model and prefetch_state and not shared_pages and not pack_inside):
        raise ValueError("Address experiments require external Pack and checkpoint prefetch")
    page_index_stride = 2 if use_address_table else 1 if shared_pages else index_stride
    query_norm_base = 40 if pack_inside else 36
    if directional_fix_sync is None:
        directional_fix_sync = prefetch_state
    state_buffers = 2 if prefetch_state else 1
    axpy = variant in ("axpy", "inline", "batched", "model")
    jobs = rows * 8
    tokens = rows * 4 if token_count is None else token_count
    gate_width = 48 if model else 32
    page_rows = state_stride // 128
    if shift_address and page_rows != 3584:
        raise ValueError("Shift/add requires a physical page of 3584 checkpoint rows")
    if address_mode == "replay_i32" and pages * replay_stride > 2**31:
        raise ValueError("Replay element offsets exceed int32")
    slots_size = rows * 2 if use_address_table else (rows - 1) * page_index_stride + 1
    waves = (jobs + 23) // 24
    arrays = [
        ("proj", "float", 8 * 256),
        ("gram", "float", 16 * 16),
        ("gates", "float", gate_width),
        ("keys", "bfloat16_t", 4 * 128),
        ("values_bf16", "bfloat16_t", 4 * 256),
        ("values", "float", 4 * 256),
        ("outputs", "bfloat16_t", 4 * 256),
        ("records", "float", 2 * 4 * 264),
        ("pred", "float", 128),
        ("result", "float", 128),
    ]
    if not axpy:
        arrays.append(("tmp", "float", 128))
    if inline:
        arrays.extend(
            [
                ("a_bf", "bfloat16_t", 4 * 32),
                ("b_bf", "bfloat16_t", 4 * 32),
                ("bias_bf", "bfloat16_t", 32),
                ("bias_ub", "float", 32),
                ("alog_ub", "float", 32),
                ("a_ub", "float", 4 * 32),
                ("b_ub", "float", 4 * 32),
                ("work", "float", 16),
                ("round_ub", "bfloat16_t", 16),
            ]
        )
    if pack_inside:
        arrays.extend(PACK_BUFFERS)
    if fused_norm:
        arrays.extend(NORM_BUFFERS)
        if vector_rstd:
            arrays.extend(VECTOR_RSTD_BUFFERS)
        if not pack_inside:
            arrays.append(("reduce_tmp", "float", 128))
    offset, buffers = 0, []
    for name, dtype, elements in arrays:
        buffers.append(f"auto {name} = ascend_ub.GetWithOffset<{dtype}>({elements},{offset});")
        offset += elements * (2 if dtype == "bfloat16_t" else 4)
    assert offset < 196352
    source = _EPILOGUE
    for k, v in {
        "BUFFERS": "\n".join(buffers),
        "JOBS": jobs,
        "PAGES": pages,
        "TOKENS": tokens,
        "INDEX": page_index_stride,
        "PROJECTION_SLOT": projection_rows * 384,
        "VALUE": value_stride,
        "REPLAY": replay_stride,
        "DEPTH": depth,
        "PART": "sid" if variant == "original" else "sid ^ (wave & 1)",
        "PACK_DECL": PACK_JOB if pack_inside else "",
        "PACK_INIT": PACK_BINDINGS + "\nPACK_JOB(group);" if pack_inside else "",
        "PACK_LOOKAHEAD": "if (job+24<@JOBS@) { PACK_JOB(job+24); }" if pack_inside else "",
        "MODEL_INPUTS_BEFORE_READY": "" if pack_inside else "@GATE_LOAD@\n" + key_load,
        "MODEL_INPUTS_AFTER_READY": "@GATE_LOAD@\n" + key_load if pack_inside else "",
        "NORM_BINDINGS": (NORM_FP32_BINDINGS if norm_weight_fp32 else NORM_BINDINGS) if fused_norm else "",
        "NORM_INIT": VECTOR_RSTD_INIT if vector_rstd else "",
        "NORM_APPLY": norm_apply(norm_activation, vector_rstd=vector_rstd) + "\nSYNC(V_MTE3);" if fused_norm else "",
        "NORM_EPS": norm_eps.hex() + "f" if fused_norm else "0.0f",
        "Z_STRIDE": z_stride,
        "Q_STRIDE": q_stride,
        "K_STRIDE": k_stride,
        "ACCUM": ("Axpy(dst,src,scale,128);" if axpy else "Muls(tmp,src,scale,128); BV(); Add(dst,dst,tmp,128); BV();"),
        "AFTER_ACCUM": "BV();" if axpy else "",
        "GATE_INIT": _GATE_INIT if inline else "",
        "GATE_LOAD": (
            "G2U(a_bf,ar[begin*@GATE@],end-begin,24,@GATE@,bfloat16_t);\n"
            "G2U(b_bf,br[begin*@GATE@],end-begin,24,@GATE@,bfloat16_t);"
            if inline
            else f"G2U(gates,ag[job*{gate_width}],1,{gate_width},{gate_width},float);"
        ),
        "GATE_COMPUTE": _GATE_COMPUTE if inline else "",
        "GATE_COEFF": (
            "decay[t]=dec_lane[t*2+h]; beta[t]=bet_lane[t*2+h];"
            if inline
            else "decay[t]=gates.GetValue(t*4+lane); beta[t]=gates.GetValue(16+t*4+lane);"
        ),
        "GATE": gate_stride,
        "MODEL_NORMS": (
            "float kn[4], qn[4]; for (int t=0; t<count; ++t) {"
            f"kn[t]=gates.GetValue(32+t); qn[t]=gates.GetValue({query_norm_base}+t); }}"
            if model
            else ""
        ),
        "KEY_FACTOR": " * kn[t]" if model else "",
        "QUERY_FACTOR": " * qn[t]" if model else "",
        "PREV_KEY_FACTOR": " * kn[j] * kn[t]" if model else "",
        "PREV_QUERY_FACTOR": " * kn[j] * qn[t]" if model else "",
        "CURR_QUERY_FACTOR": " * kn[t] * qn[t]" if model else "",
        "RECORD_KEY_UB": key_ub,
        "RECORD_DECAY_UB": decay_ub,
        "RECORD_KEY_NORM": f"BV(); Muls(c[{key_ub}],c[{key_ub}],kn[t],128);" if model else "",
        "RECORD_KEY_BEGIN": "if (lane == 0) { " if shared_replay_keys else "",
        "RECORD_KEY_END": " }" if shared_replay_keys else "",
        "MODEL_FIXED_BEGIN": (
            "if (count == 4) {\n"
            + _model_full_window_source(query_norm_base, shared_replay_keys, key_ub, decay_ub)
            + "\n} else {"
            if model
            else ""
        ),
        "MODEL_FIXED_END": "}" if model else "",
        "MODEL_REPLAY_BEGIN": (
            f"if (count == 4) {{ U2G(rg[int64_t(page)*{replay_stride}LL+head*1028],"
            "records,owned*4,257,257,float); } else {"
            if model and not shared_replay_keys
            else ""
        ),
        "MODEL_REPLAY_END": "}" if model and not shared_replay_keys else "",
        "GROUPED_REPLAY_BEGIN": (
            "if (count == 4) { U2G_GROUPED_RECORDS(rg[int64_t(page)*@REPLAY@LL+head*516],records,owned*4); }\n"
            "        if (first == 0) { U2G_GROUPED_K("
            "rg[int64_t(page)*@REPLAY@LL+12384+(head/3)*512],records[136],count); }"
            if grouped_replay
            else ""
        ),
        "REPLAY_STORE": (
            "if (count != 4) { U2G_GROUPED_RECORDS(rg[int64_t(page)*@REPLAY@LL+(head+h)*516],records[h*1056],count); }"
            if grouped_replay
            else "auto dst = rg[int64_t(page)*@REPLAY@LL+(head+h)*1028];\n"
            "            auto src = records[h*1056];\n"
            "            if (first+h == 0) { U2G(dst,src,count,257,257,float); }\n"
            "            else { U2G_REPLAY_FIELDS(dst,src,count); }"
            if shared_replay_keys
            else "U2G(rg[int64_t(page)*@REPLAY@LL+(head+h)*1028],records[h*1056],count,257,257,float);"
        ),
    }.items():
        source = source.replace(f"@{k}@", str(v))
    if address_mode == "replay_i32":
        source = source.replace("int64_t(page)*@REPLAY@LL", "int64_t(replay_base)")
        active_declaration = f"const bool active=valid && page>0 && page<{pages};"
        source = source.replace(
            active_declaration, active_declaration + "\n    const int replay_base=active ? page*@REPLAY@ : 0;"
        )
    source = (
        source.replace("@JOBS@", str(jobs)).replace("@TOKENS@", str(tokens)).replace("@REPLAY@", str(replay_stride))
    )
    bias_type = "float32" if model else "bfloat16"

    @T.prim_func
    def gdn_cube_pipe(
        state: T.Tensor((pages * page_rows, 128), "bfloat16"),
        slots: T.Tensor((slots_size,), "int32"),
        packed: T.Tensor((jobs, packed_rows, 128), "bfloat16"),
        gate: T.Tensor((jobs, gate_width), "float32"),
        yws: T.Tensor((24 * depth, projection_rows, 384), "float32"),
        gws: T.Tensor((24 * depth, 16, 16), "float32"),
        v: T.Tensor(((tokens - 1) * value_stride + 3072,), "bfloat16"),
        replay: T.Tensor((pages * replay_stride,), "float32"),
        starts: T.Tensor((rows + 1,), "int32"),
        output: T.Tensor((tokens * 3072,), "bfloat16"),
        a_raw: T.Tensor(((tokens - 1) * gate_stride + 24,), "bfloat16"),
        b_raw: T.Tensor(((tokens - 1) * gate_stride + 24,), "bfloat16"),
        log_a: T.Tensor((24,), "float32"),
        bias: T.Tensor((24,), bias_type),
        q_raw: T.Tensor(((tokens - 1) * q_stride + 1024,), "bfloat16"),
        k_raw: T.Tensor(((tokens - 1) * k_stride + 1024,), "bfloat16"),
        norm_z_raw: T.Tensor(((tokens - 1) * z_stride + 3072,), "bfloat16"),
        norm_w_raw: T.Tensor((128,), norm_weight_dtype),
    ):
        with T.Kernel(24, is_npu=True) as (core, sub):
            with T.Scope("C"):
                state_l1 = T.alloc_L1((state_buffers, first_v, 128), "bfloat16")
                state_tail_l1 = T.alloc_L1((128, 128), "bfloat16")
                qk_l1 = T.alloc_L1((16, 128), "bfloat16")
                projection_l0 = T.alloc_L0C((16, first_v), "float32")
                projection_tail_l0 = T.alloc_L0C((16, 128), "float32")
                gram_l0 = T.alloc_L0C((16, 16), "float32")
                if dense_qk:
                    # Clear the physical NZ tile once. Per-wave ND->NZ copy only writes the first 8 rows.
                    T._src_code("AscendC::InitConstValue(qk_l1[0], {1, 128, 0, bfloat16_t(0)});")
                    if dense_qk_once:
                        T._src_code("AscendC::PipeBarrier<PIPE_MTE2>();")
                    else:
                        T.barrier_all()
                # Only cores with a first-wave job may read its request metadata.
                core_row = core // 8
                core_head = core % 8
                head_start = (((core_head << 1) + core_head) << 7) if shift_address else core_head * 384
                if prefetch_state and (rows >= 3 or core < jobs):
                    raw = slots[core_row * page_index_stride]
                    page = T.min(T.max(raw, 0), pages - 1)
                    base = (
                        slots[core_row * 2 + 1]
                        if use_address_table
                        else ((((page << 3) - page) << 9) if shift_address else page * page_rows)
                    )
                    start = base + head_start
                    if not skip_inactive or (
                        raw > 0
                        and raw < pages
                        and (
                            shared_pages
                            or (
                                starts[core // 8] >= 0
                                and starts[core // 8 + 1] <= tokens
                                and starts[core // 8 + 1] > starts[core // 8]
                                and starts[core // 8 + 1] - starts[core // 8] <= 4
                            )
                        )
                    ):
                        T.copy(state[start : start + first_v, 0:128], state_l1[0, 0:first_v, 0:128])
                # B32 runs two initial waves, seven steady waves, then two tail waves.
                for phase in T.unroll(4 if steady_aic else 1):
                    wave_start = (
                        T.if_then_else(phase == 0, 0, T.if_then_else(phase == 1, 2, phase + 7)) if steady_aic else 0
                    )
                    wave_end = (
                        T.if_then_else(phase == 0, 2, T.if_then_else(phase == 1, 9, phase + 8)) if steady_aic else waves
                    )
                    for wave in T.serial(wave_start, wave_end):
                        job = wave * 24 + core
                        if (steady_aic and phase < 3) or job < jobs:
                            row = wave * 3 + core_row if hoist_address else job // 8
                            raw = slots[row * page_index_stride]
                            begin = starts[row]
                            end = starts[row + 1]
                            active = not skip_inactive or (
                                raw > 0
                                and raw < pages
                                and (
                                    shared_pages or (begin >= 0 and end <= tokens and end > begin and end - begin <= 4)
                                )
                            )
                            if pack_inside:
                                T.wait_cross_flag(7)
                            if active:
                                if not prefetch_state:
                                    page = T.min(T.max(raw, 0), pages - 1)
                                    start = page * page_rows + (job % 8) * 384
                                    T.copy(state[start : start + first_v, 0:128], state_l1[0, 0:first_v, 0:128])
                                if dense_qk_once:
                                    # The generic partial copy repeats full-tile clear on each wave.
                                    T._src_code(
                                        "tl::ascend::copy_gm_to_l1<bfloat16_t, 16, 128>"
                                        "(qk_l1[0], packed[(wave * 24 + core) * 1024], 128, 8, 128, false);"
                                    )
                                else:
                                    T.copy(packed[job, 0:packed_rows, 0:128], qk_l1[0:packed_rows, 0:128])
                            if directional_input_sync:
                                T.set_flag("MTE2", "MTE1", 1)
                                T.wait_flag("MTE2", "MTE1", 1)
                            else:
                                T.barrier_all()
                            if split_checkpoint and active:
                                tail_page = T.min(T.max(raw, 0), pages - 1)
                                tail_start = tail_page * page_rows + (job % 8) * 384 + first_v
                                T.copy(state[tail_start : tail_start + 128, 0:128], state_tail_l1)
                                T.set_flag("MTE2", "MTE1", 3)
                            if prefetch_state and (
                                (steady_aic and phase < 2) or ((not steady_aic or phase == 2) and job + 24 < jobs)
                            ):
                                next_row = row + 3
                                next_raw = slots[next_row * page_index_stride]
                                next_page = T.min(T.max(next_raw, 0), pages - 1)
                                next_base = (
                                    slots[next_row * 2 + 1]
                                    if use_address_table
                                    else (
                                        (((next_page << 3) - next_page) << 9)
                                        if shift_address
                                        else next_page * page_rows
                                    )
                                )
                                next_start = next_base + (
                                    head_start if hoist_address or shift_address else (job % 8) * 384
                                )
                                next_begin = starts[next_row]
                                next_end = starts[next_row + 1]
                                if not skip_inactive or (
                                    next_raw > 0
                                    and next_raw < pages
                                    and (
                                        shared_pages
                                        or (
                                            next_begin >= 0
                                            and next_end <= tokens
                                            and next_end > next_begin
                                            and next_end - next_begin <= 4
                                        )
                                    )
                                ):
                                    T.copy(
                                        state[next_start : next_start + first_v, 0:128],
                                        state_l1[(wave + 1) % state_buffers, 0:first_v, 0:128],
                                    )
                            if active:
                                T.gemm_v0(
                                    qk_l1,
                                    state_l1[wave % state_buffers, 0:first_v, 0:128],
                                    projection_l0,
                                    transpose_B=True,
                                    init=True,
                                    kL0Size=32,
                                )
                                if directional_fix_sync:
                                    T.set_flag("M", "FIX", 2)
                                    T.wait_flag("M", "FIX", 2)
                                else:
                                    T.barrier_all()
                            if (steady_aic and phase > 0) or (not steady_aic and wave >= depth):
                                T.wait_cross_flag(6)
                            if active:
                                workspace = core * depth + wave % depth
                                T.copy(
                                    projection_l0[0:projection_rows, 0:first_v],
                                    yws[workspace, 0:projection_rows, 0:first_v],
                                )
                                if split_checkpoint:
                                    T.wait_flag("MTE2", "MTE1", 3)
                                    T.gemm_v0(
                                        qk_l1,
                                        state_tail_l1,
                                        projection_tail_l0,
                                        transpose_B=True,
                                        init=True,
                                        kL0Size=32,
                                    )
                                    T.set_flag("M", "FIX", 2)
                                    T.wait_flag("M", "FIX", 2)
                                    T.copy(
                                        projection_tail_l0[0:projection_rows, 0:128],
                                        yws[workspace, 0:projection_rows, first_v:384],
                                    )
                                T.gemm_v0(
                                    qk_l1,
                                    qk_l1,
                                    gram_l0,
                                    transpose_B=True,
                                    init=True,
                                    kL0Size=128,
                                )
                                if directional_fix_sync:
                                    T.set_flag("M", "FIX", 2)
                                    T.wait_flag("M", "FIX", 2)
                                else:
                                    T.barrier_all()
                                T.copy(gram_l0, gws[workspace, 0:16, 0:16])
                            # Empty jobs still complete the same ready/free handshake.
                            T.set_cross_flag("FIX", 5)
                            if directional_wave_end and job + 24 < jobs:
                                # The next Mmad must not overwrite L0C while Fixpipe reads it.
                                # The next wave's input barrier still waits for its L1 loads.
                                T.set_flag("FIX", "M", 4)
                                T.wait_flag("FIX", "M", 4)
                            else:
                                T.barrier_all()
                for tail in T.serial(depth):
                    if core + 24 * tail < jobs:
                        T.wait_cross_flag(6)
            with T.Scope("V"):
                T._src_code(source)

    source_id = sha256(
        f"{prefetch_state=}\n{directional_fix_sync=}\n{directional_wave_end=}\n"
        f"{skip_inactive=}\n{shared_pages=}\n{directional_input_sync=}\n"
        f"{split_checkpoint=}\n{dense_qk=}\n{dense_qk_once=}\n"
        f"{steady_aic=}\n{address_mode=}\n{source}".encode()
    ).hexdigest()[:20]
    return tilelang.compile(
        gdn_cube_pipe.with_attr("global_symbol", f"gdn_cube_pipe_{source_id}"),
        target="ascendc",
        platform="A3",
    )


def _model_full_window_source(query_norm_base, shared_replay_keys=False, key_ub=128, decay_ub=256):
    """Unroll the four-step model path and share raw Gram loads across heads."""
    lines = []
    for t in range(4):
        lines.extend(
            (
                f"const float kn{t}=gates.GetValue({32 + t}), qn{t}=gates.GetValue({query_norm_base + t});",
                f"const float self_qk{t}=gram.GetValue({t * 16 + t + 4});",
            )
        )
        for j in range(t):
            lines.append(f"const float kk{j}{t}=gram.GetValue({j * 16 + t});")
            lines.append(f"const float qk{j}{t}=gram.GetValue({j * 16 + t + 4});")
    lines.extend(("for (int h=0; h<owned; ++h) {", "const int lane=first+h;"))
    for t in range(4):
        lines.extend(
            (
                f"const float d{t}=gates.GetValue({t * 4}+lane), beta{t}=gates.GetValue({16 + t * 4}+lane);",
                f"records.SetValue(h*1056+{t * 264 + decay_ub},d{t});",
            )
        )
    lines.extend(
        (
            "SYNC(S_V);",
            "const float d12=d1*d2, d23=d2*d3, d123=d12*d3;",
            "float prefix=1.0f;",
        )
    )
    decay_products = {(0, 2): "d12", (0, 3): "d123", (1, 3): "d23"}
    for t in range(4):
        lines.extend(
            (
                f"prefix*=d{t};",
                f"Muls(pred,proj[{t}*width+h*128],prefix*kn{t},128);",
                f"Muls(result,proj[{t + 4}*width+h*128],prefix*qn{t},128);",
                "BV();",
            )
        )
        for j in range(t):
            weight = decay_products.get((j, t), f"d{t}")
            lines.extend(
                (
                    f"ACCUM(pred,c{j},{weight}*kk{j}{t}*kn{j}*kn{t});",
                    f"ACCUM(result,c{j},{weight}*qk{j}{t}*kn{j}*qn{t}); BV();",
                )
            )
        lines.extend(
            (
                f"auto c{t}=records[h*1056+{t * 264}];",
                f"Sub(c{t},values[{t}*width+h*128],pred,128); BV();",
                f"Muls(c{t},c{t},beta{t},128); BV();",
                f"ACCUM(result,c{t},self_qk{t}*kn{t}*qn{t}); BV();",
                "Muls(result,result,0x1.6a09e6p-4f,128); BV();",
                f"Cast(outputs[{t}*width+h*128],result,RoundMode::CAST_RINT,128);",
            )
        )
        if shared_replay_keys:
            lines.append("if (lane == 0) {")
        lines.extend(
            (
                f"Cast(c{t}[{key_ub}],keys[{t * 128}],RoundMode::CAST_NONE,128);",
                f"BV(); Muls(c{t}[{key_ub}],c{t}[{key_ub}],kn{t},128); BV();",
            )
        )
        if shared_replay_keys:
            lines.append("} else { BV(); }")
    lines.append("}")
    return "\n".join(lines)


_EPILOGUE = r"""
{
using namespace AscendC;
@BUFFERS@
GlobalTensor<float> pg, gg, ag, rg;
GlobalTensor<bfloat16_t> kg, vg, og;
GlobalTensor<int32_t> sg, lg;
pg.SetGlobalBuffer((__gm__ float*)yws_handle);
gg.SetGlobalBuffer((__gm__ float*)gws_handle);
ag.SetGlobalBuffer((__gm__ float*)gate_handle);
rg.SetGlobalBuffer((__gm__ float*)replay_handle);
kg.SetGlobalBuffer((__gm__ bfloat16_t*)packed_handle);
vg.SetGlobalBuffer((__gm__ bfloat16_t*)v_handle);
og.SetGlobalBuffer((__gm__ bfloat16_t*)output_handle);
sg.SetGlobalBuffer((__gm__ int32_t*)slots_handle);
lg.SetGlobalBuffer((__gm__ int32_t*)starts_handle);
#define G2U(dst,src,n,width,stride,Tp) do { \
  DataCopyExtParams p={uint16_t(n),uint32_t((width)*sizeof(Tp)),uint32_t(((stride)-(width))*sizeof(Tp)),0,0}; \
  DataCopyPadExtParams<Tp> pad={false,0,0,0}; DataCopyPad((dst),(src),p,pad); } while(0)
#define U2G(dst,buf,n,width,stride,Tp) do { \
  DataCopyExtParams p={uint16_t(n),uint32_t((width)*sizeof(Tp)),0,uint32_t(((stride)-(width))*sizeof(Tp)),0}; \
  DataCopyPad((dst),(buf),p); } while(0)
// The existing UB record uses 264 floats, with decay at offset 256.
// Copy correction and decay as two strided transfers, leaving the duplicated K slot untouched.
#define U2G_REPLAY_FIELDS(dst,buf,n) do { \
  DataCopyExtParams correction={uint16_t(n),128*sizeof(float),(264-128)/8,(257-128)*sizeof(float),0}; \
  DataCopyExtParams decay={uint16_t(n),sizeof(float),(264-8)/8,(257-1)*sizeof(float),0}; \
  DataCopyPad((dst),(buf),correction); DataCopyPad((dst)[256],(buf)[256],decay); } while(0)
#define U2G_GROUPED_RECORDS(dst,buf,n) do { \
  DataCopyExtParams p={uint16_t(n),129*sizeof(float),(264-136)/8,0,0}; \
  DataCopyPad((dst),(buf),p); } while(0)
#define U2G_GROUPED_K(dst,buf,n) do { \
  DataCopyExtParams p={uint16_t(n),128*sizeof(float),(264-128)/8,0,0}; \
  DataCopyPad((dst),(buf),p); } while(0)
#define SYNC(event) do { SetFlag<HardEvent::event>(0); WaitFlag<HardEvent::event>(0); } while(0)
#define BV() PipeBarrier<PIPE_V>()
#define ACCUM(dst,src,scale) do { @ACCUM@ } while(0)
@PACK_DECL@
const int group = GetBlockIdx() / 2;
const int sid = GetSubBlockIdx();
@PACK_INIT@
@NORM_BINDINGS@
@NORM_INIT@
@GATE_INIT@
for (int wave=0, job=group; job<@JOBS@; ++wave,job+=24) {
    const int part=@PART@, first=part*2, owned=2-part, width=owned*128;
    const int row=job/8, head=job%8*3+first;
    const int page=sg.GetValue(row*@INDEX@), begin=lg.GetValue(row), end=lg.GetValue(row+1);
    const bool valid=begin>=0 && end<=@TOKENS@ && end>begin && end-begin<=4;
    const bool active=valid && page>0 && page<@PAGES@;
    if (active) {
        @MODEL_INPUTS_BEFORE_READY@
        G2U(values_bf16,vg[begin*@VALUE@+head*128],end-begin,width,@VALUE@,bfloat16_t);
    }
    @GATE_COMPUTE@
    CrossCoreWaitFlag(5);
    if (active) {
        @MODEL_INPUTS_AFTER_READY@
        const int ws=group*@DEPTH@+wave%@DEPTH@;
        G2U(proj,pg[ws*@PROJECTION_SLOT@+first*128],8,width,384,float);
        G2U(gram,gg[ws*256],1,256,256,float);
    }
    CrossCoreSetFlag<2,PIPE_MTE2>(6);
    @PACK_LOOKAHEAD@
    if (!valid) continue;
    const int count=end-begin;
    if (active) {
        SYNC(MTE2_S);
        SYNC(S_V);
        Cast(values,values_bf16,RoundMode::CAST_NONE,count*width);
        @MODEL_FIXED_BEGIN@
        @MODEL_NORMS@
        for (int h=0;h<owned;++h) {
            const int lane=first+h;
            float decay[4],beta[4];
            for (int t=0;t<count;++t) {
                @GATE_COEFF@
                records.SetValue(h*1056+t*264+@RECORD_DECAY_UB@,decay[t]);
            }
            SYNC(S_V);
            float prefix=1.0f;
            for (int t=0;t<count;++t) {
                prefix*=decay[t];
                Muls(pred,proj[t*width+h*128],prefix@KEY_FACTOR@,128);
                Muls(result,proj[(t+4)*width+h*128],prefix@QUERY_FACTOR@,128);
                BV();
                for (int j=0;j<t;++j) {
                    float weight=1.0f;
                    for (int k=j+1;k<=t;++k) weight*=decay[k];
                    const auto previous=records[h*1056+j*264];
                    ACCUM(pred,previous,weight*gram.GetValue(j*16+t)@PREV_KEY_FACTOR@);
                    ACCUM(result,previous,weight*gram.GetValue(j*16+t+4)@PREV_QUERY_FACTOR@); @AFTER_ACCUM@
                }
                auto c=records[h*1056+t*264];
                Sub(c,values[t*width+h*128],pred,128); BV();
                Muls(c,c,beta[t],128); BV();
                ACCUM(result,c,gram.GetValue(t*16+t+4)@CURR_QUERY_FACTOR@); @AFTER_ACCUM@
                Muls(result,result,0x1.6a09e6p-4f,128); BV();
                Cast(outputs[t*width+h*128],result,RoundMode::CAST_RINT,128);
                @RECORD_KEY_BEGIN@Cast(c[@RECORD_KEY_UB@],keys[t*128],RoundMode::CAST_NONE,128);
                @RECORD_KEY_NORM@@RECORD_KEY_END@
                BV();
            }
        }
        @MODEL_FIXED_END@
        PipeBarrier<PIPE_ALL>();
        @MODEL_REPLAY_BEGIN@@GROUPED_REPLAY_BEGIN@
        for (int h=0;h<owned;++h) {
            @REPLAY_STORE@
        }
        @MODEL_REPLAY_END@
    } else {
        Duplicate(outputs,bfloat16_t(0),count*width); SYNC(V_MTE3);
    }
    @NORM_APPLY@
    U2G(og[(begin*24+head)*128],outputs,count,width,3072,bfloat16_t);
    PipeBarrier<PIPE_ALL>();
}
#undef G2U
#undef U2G
#undef U2G_REPLAY_FIELDS
#undef U2G_GROUPED_RECORDS
#undef U2G_GROUPED_K
#undef SYNC
#undef BV
#undef ACCUM
#undef PACK_JOB
}
"""


_GATE_INIT = r"""
GlobalTensor<bfloat16_t> ar, br, bias_gm;
GlobalTensor<float> alog_gm;
ar.SetGlobalBuffer((__gm__ bfloat16_t*)a_raw_handle);
br.SetGlobalBuffer((__gm__ bfloat16_t*)b_raw_handle);
bias_gm.SetGlobalBuffer((__gm__ bfloat16_t*)bias_handle);
alog_gm.SetGlobalBuffer((__gm__ float*)log_a_handle);
G2U(bias_bf,bias_gm[0],1,24,24,bfloat16_t);
G2U(alog_ub,alog_gm[0],1,24,24,float);
SYNC(MTE2_V);
Cast(bias_ub,bias_bf,RoundMode::CAST_NONE,24);
Exp(alog_ub,alog_ub,24);
SYNC(V_S);
"""


_GATE_COMPUTE = r"""
float dec_lane[8],bet_lane[8];
if (active) {
    const int steps=end-begin;
    SYNC(MTE2_V);
    Cast(a_ub,a_bf,RoundMode::CAST_NONE,steps*32);
    Cast(b_ub,b_bf,RoundMode::CAST_NONE,steps*32);
    Duplicate(work,0.0f,16);
    SYNC(V_S);
    float x_lane[8];
    for (int t=0;t<steps;++t) {
        for (int h=0;h<owned;++h) {
            const int i=t*2+h;
            x_lane[i]=a_ub.GetValue(t*32+head+h)+bias_ub.GetValue(head+h);
            work.SetValue(i,x_lane[i]);
            work.SetValue(8+i,-b_ub.GetValue(t*32+head+h));
        }
    }
    SYNC(S_V); Exp(work,work,16); SYNC(V_S);
    for (int t=0;t<steps;++t) {
        for (int h=0;h<owned;++h) {
            const int i=t*2+h;
            bet_lane[i]=1.0f/(work.GetValue(8+i)+1.0f);
            work.SetValue(i,work.GetValue(i)+1.0f);
        }
    }
    SYNC(S_V); Ln(work,work,8); SYNC(V_S);
    for (int t=0;t<steps;++t) {
        for (int h=0;h<owned;++h) {
            const int i=t*2+h;
            work.SetValue(i,-alog_ub.GetValue(head+h)*(x_lane[i]<=20.0f ? work.GetValue(i) : x_lane[i]));
            work.SetValue(8+i,bet_lane[i]);
        }
    }
    SYNC(S_V); Exp(work,work,8); BV();
    Cast(round_ub,work[8],RoundMode::CAST_RINT,8); BV();
    Cast(work[8],round_ub,RoundMode::CAST_NONE,8); SYNC(V_S);
    for (int t=0;t<steps;++t) {
        for (int h=0;h<owned;++h) {
            const int i=t*2+h;
            dec_lane[i]=work.GetValue(i);
            bet_lane[i]=work.GetValue(8+i);
        }
    }
}
"""
