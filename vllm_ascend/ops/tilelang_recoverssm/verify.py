# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 Qwen MTP3 verify probe. Q and K arrive normalized by the model."""

import tilelang
from tilelang import language as T


def build(
    tile_v,
    *,
    rows=8,
    pages=12,
    tokens=32,
    replay_steps=4,
    value_heads=24,
    key_heads=8,
    state_stride=None,
    replay_stride=None,
    index_stride=1,
    gate_stride=None,
    b_token_stride=None,
    v_token_stride=None,
    dt_dtype="bfloat16",
    log_dtype="float32",
    output_dtype="bfloat16",
    native_reduce=True,
    rebase_pages=False,
    packed_replay=False,
):
    dim = 128
    if tile_v not in (64, 128):
        raise ValueError("expected 64 or 128 value rows per vector worker")
    if packed_replay and (tile_v != dim or not native_reduce):
        raise ValueError("a packed replay record requires one worker per complete value head")
    partitions = dim // tile_v
    jobs = rows * value_heads * partitions
    if jobs % 2 or value_heads % key_heads:
        raise ValueError("unsupported work partition")
    state_stride = state_stride or value_heads * dim * dim
    replay_stride = replay_stride or value_heads * replay_steps * 257
    gate_stride = gate_stride or value_heads
    b_token_stride = b_token_stride or gate_stride
    v_token_stride = v_token_stride or value_heads * dim
    descriptor_pages = 1 if rebase_pages else pages
    bf = "bfloat16"
    fp = "float32"

    # dav-2201 vector instructions process 64 FP32 elements per repeat.
    # A zero source repeat stride reuses the same key/query across value rows.
    # BiSheng compiles these inline in the vector kernel execution space.
    row_product = """
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::BinaryRepeatParams p(1, 1, 1, 16, 16, 0);
        for (int col = 0; col < 128; col += 64)
            AscendC::Mul<float>(dst[col], mat[col], vec[col], uint64_t(64), nr, p);
        AscendC::PipeBarrier<PIPE_V>();
    """
    row_sum = """
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::BinaryRepeatParams p(1, 1, 1, 16, 16, 16);
        AscendC::Add<float>(mat, mat, mat[64], uint64_t(64), nr, p);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(dst, mat, uint64_t(64), nr, 1, 1, 16);
        AscendC::PipeBarrier<PIPE_V>();
    """
    helpers = (
        "\n#define row_product(dst,mat,vec,nr) do { "
        + " ".join(row_product.split())
        + " } while(0)\n#define row_sum(dst,mat,nr) do { "
        + " ".join(row_sum.split())
        + " } while(0)\n"
    )

    @T.prim_func
    def verify(
        log_a: T.Tensor((value_heads,), log_dtype),
        a: T.Tensor((tokens, value_heads), bf, strides=(gate_stride, 1)),
        b: T.Tensor((tokens, value_heads), bf, strides=(b_token_stride, 1)),
        bias: T.Tensor((value_heads,), dt_dtype),
        q: T.Tensor((tokens, key_heads, dim), bf),
        k: T.Tensor((tokens, key_heads, dim), bf),
        v: T.Tensor((tokens, value_heads, dim), bf, strides=(v_token_stride, dim, 1)),
        state: T.Tensor((descriptor_pages, value_heads, dim, dim), bf, strides=(state_stride, dim * dim, dim, 1)),
        replay: T.Tensor(
            (descriptor_pages, value_heads, replay_steps, 257), fp, strides=(replay_stride, replay_steps * 257, 257, 1)
        ),
        starts: T.Tensor((rows + 1,), "int32"),
        slots: T.Tensor((rows,), "int32", strides=(index_stride,)),
        out: T.Tensor((tokens, value_heads, dim), output_dtype),
    ):
        with T.Kernel(jobs // 2, is_npu=True) as (core, sub):  # noqa: SIM117 - TileLang parses nested scopes
            with T.Scope("V"):
                if native_reduce:
                    T._src_code(helpers)
                job = core * 2 + sub
                row = job // (value_heads * partitions)
                head = job // partitions % value_heads
                kh = head // (value_heads // key_heads)
                lo = job % partitions * tile_v
                slot = slots[row]
                begin = starts[row]
                length = starts[row + 1] - begin
                h = T.alloc_ub((tile_v, dim), fp)
                half_h = T.alloc_ub((tile_v, dim), bf)
                work = T.alloc_ub((tile_v, dim), fp)
                column = T.alloc_ub((16, 1), fp)
                stripe = T.alloc_ub((16, dim), fp)
                key_row = T.alloc_ub((1, dim + tile_v * 8), fp)
                query_row = T.alloc_ub((1, dim), fp)
                half_key = T.alloc_ub((dim,), bf)
                half_query = T.alloc_ub((dim,), bf)
                half_v = T.alloc_ub((tile_v,), bf)
                val = T.alloc_ub((tile_v,), fp)
                proj = T.alloc_ub((tile_v, 1), fp)
                record = T.alloc_ub((264,), fp)
                correction = T.alloc_ub((tile_v,), fp)
                result = T.alloc_ub((tile_v,), output_dtype)
                a_half = T.alloc_ub((value_heads,), bf)
                b_half = T.alloc_ub((value_heads,), bf)
                a_float = T.alloc_ub((value_heads,), fp)
                b_float = T.alloc_ub((value_heads,), fp)
                bias_raw = T.alloc_ub((value_heads,), dt_dtype)
                bias_float = T.alloc_ub((value_heads,), fp)
                log_raw = T.alloc_ub((value_heads,), log_dtype)
                log_float = T.alloc_ub((value_heads,), fp)
                gates = T.alloc_ub((8,), fp)
                gate_bf = T.alloc_ub((8,), bf)
                if length > 0:
                    if slot > 0:
                        # A reduced service can have >2^32 addressable elements.
                        # Rebase on the selected page before issuing tile DMA.
                        if rebase_pages:
                            T._src_code(
                                f"state.SetGlobalBuffer((__gm__ bfloat16_t*)state_handle + ((int64_t)slot) * {state_stride}LL); "  # noqa: E501 - inline AscendC intrinsic
                                f"replay.SetGlobalBuffer((__gm__ float*)replay_handle + ((int64_t)slot) * {replay_stride}LL);"  # noqa: E501 - inline AscendC intrinsic
                            )
                        page = 0 if rebase_pages else slot
                        T.copy(state[page, head, lo : lo + tile_v, :], half_h)
                        T.tile.cast(h, half_h, "CAST_NONE", tile_v * dim)
                        T.copy(bias, bias_raw)
                        T.copy(log_a, log_raw)
                        T.copy(bias_raw, bias_float)
                        T.copy(log_raw, log_float)
                        T.tile.fill(gates, 0.0)
                        for step in T.serial(length):
                            token = begin + step
                            # Use vector math for exp/log and an explicit BF16 gate round trip.
                            T.copy(a[token, :], a_half)
                            T.copy(b[token, :], b_half)
                            T.tile.cast(a_float, a_half, "CAST_NONE", value_heads)
                            T.tile.cast(b_float, b_half, "CAST_NONE", value_heads)
                            gates[0] = a_float[head] + bias_float[head]
                            gates[1] = log_float[head]
                            gates[2] = -b_float[head]
                            x = gates[0]
                            T.tile.exp(gates, gates)
                            exp_a = gates[1]
                            inv_beta = gates[2] + 1.0
                            gates[0] = gates[0] + 1.0
                            T.tile.ln(gates, gates)
                            gates[0] = -exp_a * T.if_then_else(x <= 20.0, gates[0], x)
                            T.tile.exp(gates, gates)
                            decay = gates[0]
                            gates[1] = 1.0 / inv_beta
                            T.tile.cast(gate_bf, gates, "CAST_RINT", 8)
                            T.tile.cast(gates, gate_bf, "CAST_NONE", 8)
                            beta = gates[1]
                            T.copy(k[token, kh, :], half_key)
                            T.copy(q[token, kh, :], half_query)
                            T.copy(v[token, head, lo : lo + tile_v], half_v)
                            T.tile.cast(key_row, half_key, "CAST_NONE", dim)
                            T.tile.cast(query_row, half_query, "CAST_NONE", dim)
                            T.tile.cast(val, half_v, "CAST_NONE", tile_v)
                            # Preserve the exact float32 bits of 128**-0.5. The
                            # generic code generator prints only seven digits.
                            T._src_code(
                                "AscendC::PipeBarrier<PIPE_V>(); "
                                "AscendC::Muls(query_row, query_row, 0x1.6a09e6p-4f, 128); "
                                "AscendC::PipeBarrier<PIPE_V>();"
                            )
                            T.tile.mul(h, h, decay)
                            if native_reduce:
                                T._src_code(f"row_product(work,h,key_row,{tile_v});")
                                T._src_code(f"row_sum(proj,work,{tile_v});")
                            else:
                                T.tile.broadcast(work, key_row)
                                T.tile.mul(work, h, work)
                                T.reduce_sum(work, proj[:, 0], dim=-1)
                            for i in T.Parallel(tile_v):
                                correction[i] = (val[i] - proj[i, 0]) * beta
                            if native_reduce:
                                # Brcb expands eight FP32 lanes per correction.
                                # Each repeated VMUL reuses the packed key and one correction.
                                T._src_code(
                                    f"AscendC::PipeBarrier<PIPE_V>(); "
                                    f"AscendC::Brcb(key_row[128], correction, {tile_v // 8}, AscendC::BrcbRepeatParams(1,8)); "  # noqa: E501 - inline AscendC intrinsic
                                    f"AscendC::PipeBarrier<PIPE_V>(); "
                                    f"AscendC::Mul(work[0],key_row[128],key_row[0],uint64_t(64),{tile_v},AscendC::BinaryRepeatParams(1,0,1,16,1,0)); "  # noqa: E501 - inline AscendC intrinsic
                                    f"AscendC::Mul(work[64],key_row[128],key_row[64],uint64_t(64),{tile_v},AscendC::BinaryRepeatParams(1,0,1,16,1,0)); "  # noqa: E501 - inline AscendC intrinsic
                                    f"AscendC::PipeBarrier<PIPE_V>(); "
                                    f"AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(7); "
                                    f"AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(7);"
                                )
                                T.tile.add(h, h, work)
                            else:
                                T.tile.broadcast(work, key_row)
                                for block in T.serial(tile_v // 16):
                                    off = block * 16
                                    T.copy(correction[off : off + 16], column[:, 0])
                                    T.tile.broadcast(stripe, column)
                                    T.tile.mul(stripe, stripe, work[off : off + 16, :])
                                    T.tile.add(h[off : off + 16, :], h[off : off + 16, :], stripe)
                            if native_reduce:
                                T._src_code(f"row_product(work,h,query_row,{tile_v});")
                                T._src_code(f"row_sum(proj,work,{tile_v});")
                            else:
                                T.tile.broadcast(work, query_row)
                                T.tile.mul(work, h, work)
                                T.reduce_sum(work, proj[:, 0], dim=-1)
                            if output_dtype == "bfloat16":
                                T.tile.cast(result, proj, "CAST_RINT", tile_v)
                            else:
                                T.copy(proj[:, 0], result)
                            T.copy(result, out[token, head, lo : lo + tile_v])
                            if packed_replay:
                                # The aligned UB allocation holds 264 floats; the GM record has only 257.
                                T.copy(correction, record[:128])
                                T.copy(key_row[0, :128], record[128:256])
                                record[256] = decay
                                T.copy(record[:257], replay[page, head, step, :257])
                            else:
                                T.copy(correction, replay[page, head, step, lo : lo + tile_v])
                                if lo == 0:
                                    T.copy(key_row[0, :dim], replay[page, head, step, dim : 2 * dim])
                                    gates[0] = decay
                                    T.copy(gates[0:1], replay[page, head, step, 2 * dim : 2 * dim + 1])
                    else:
                        T.tile.fill(result, 0.0)
                        for step in T.serial(length):
                            T.copy(result, out[begin + step, head, lo : lo + tile_v])

    return tilelang.compile(
        verify,
        target="ascendc",
        platform="A3",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )
